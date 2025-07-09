from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv
import cv2
from datetime import datetime
from enum import Enum


class PassType(Enum):
    """Types of pass outcomes."""
    SUCCESSFUL = "successful"
    INTERCEPTED = "intercepted"  # Ball goes to opponent
    OUT_OF_BOUNDS = "out_of_bounds"  # Ball goes out of play
    MISSED = "missed"  # Ball doesn't reach any player


@dataclass
class Pass:
    """Represents a successful pass between players."""
    passer_id: int
    receiver_id: int
    passer_team: int
    receiver_team: int
    frame_number: int
    timestamp: float
    passer_position: np.ndarray
    receiver_position: np.ndarray
    ball_position: np.ndarray
    distance: float
    pass_type: PassType = PassType.SUCCESSFUL


@dataclass
class UnsuccessfulPass:
    """Represents an unsuccessful pass attempt."""
    passer_id: int
    passer_team: int
    frame_number: int
    timestamp: float
    passer_position: np.ndarray
    ball_position: np.ndarray
    pass_type: PassType
    intended_receiver_id: Optional[int] = None
    intended_receiver_team: Optional[int] = None
    intercepted_by_id: Optional[int] = None
    intercepted_by_team: Optional[int] = None
    distance: Optional[float] = None


@dataclass
class PlayerState:
    """Represents the state of a player at a given time."""
    player_id: int
    team_id: int
    position: np.ndarray
    frame_number: int
    timestamp: float
    has_ball: bool
    ball_distance: float


class PassTracker:
    """
    Tracks passes between players and detects ball proximity.
    
    This class maintains the state of players and the ball over time,
    detecting when passes occur between players of the same team and
    calculating the distance between the ball and each player.
    """
    
    def __init__(self, 
                 ball_proximity_threshold: float = 50.0,
                 pass_distance_threshold: float = 200.0,
                 min_pass_frames: int = 5,
                 max_player_history: int = 30,
                 out_of_bounds_timeout: int = 15):  # Frames to wait before considering ball out of bounds
        """
        Initialize the PassTracker.
        
        Args:
            ball_proximity_threshold: Distance in pixels to consider ball "close" to player
            pass_distance_threshold: Maximum distance for a valid pass
            min_pass_frames: Minimum frames between potential passes
            max_player_history: Maximum number of frames to keep player history
            out_of_bounds_timeout: Frames to wait before considering ball out of bounds
        """
        self.ball_proximity_threshold = ball_proximity_threshold
        self.pass_distance_threshold = pass_distance_threshold
        self.min_pass_frames = min_pass_frames
        self.max_player_history = max_player_history
        self.out_of_bounds_timeout = out_of_bounds_timeout
        
        # Player tracking
        self.player_history: Dict[int, deque] = {}
        self.current_players: Dict[int, PlayerState] = {}
        
        # Ball tracking
        self.ball_history: deque = deque(maxlen=max_player_history)
        self.current_ball_position: Optional[np.ndarray] = None
        self.ball_lost_frames: int = 0  # Track how long ball has been undetected
        
        # Pass tracking
        self.passes: List[Pass] = []
        self.unsuccessful_passes: List[UnsuccessfulPass] = []
        self.last_pass_frame = -1
        
        # Pass attempt tracking
        self.pending_pass_attempts: List[Dict] = []  # Track attempted passes
        
        # Frame tracking
        self.frame_number = 0
        self.fps = 30.0  # Default FPS, can be updated
        
    def update_fps(self, fps: float):
        """Update the FPS for timestamp calculations."""
        self.fps = fps
        
    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.linalg.norm(pos1 - pos2)
    
    def _is_same_team(self, team1: int, team2: int) -> bool:
        """Check if two players are on the same team."""
        return team1 == team2
    
    def _get_player_position(self, player_id: int) -> Optional[np.ndarray]:
        """Get the current position of a player."""
        if player_id in self.current_players:
            return self.current_players[player_id].position
        return None
    
    def update_ball_position(self, ball_detections: sv.Detections):
        """Update the current ball position."""
        if len(ball_detections) > 0:
            # Get the center of the ball detection
            ball_center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            if len(ball_center) > 0:
                self.current_ball_position = ball_center[0]
                self.ball_history.append(self.current_ball_position.copy())
                self.ball_lost_frames = 0  # Reset lost ball counter
        else:
            self.current_ball_position = None
            self.ball_lost_frames += 1
    
    def update_players(self, 
                      player_detections: sv.Detections, 
                      team_ids: np.ndarray,
                      player_ids: Optional[np.ndarray] = None):
        """
        Update player positions and states.
        
        Args:
            player_detections: Detections of players
            team_ids: Team IDs for each player
            player_ids: Optional player IDs for tracking
        """
        if len(player_detections) == 0:
            return
            
        # Get player positions (bottom center of bounding boxes)
        player_positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Generate player IDs if not provided
        if player_ids is None:
            player_ids = np.arange(len(player_detections))
        
        # Update current players
        self.current_players.clear()
        for i, (player_id, position, team_id) in enumerate(zip(player_ids, player_positions, team_ids)):
            # Calculate ball distance
            ball_distance = float('inf')
            has_ball = False
            
            if self.current_ball_position is not None:
                ball_distance = self._calculate_distance(position, self.current_ball_position)
                has_ball = ball_distance <= self.ball_proximity_threshold
            
            # Create player state
            player_state = PlayerState(
                player_id=int(player_id),
                team_id=int(team_id),
                position=position.copy(),
                frame_number=self.frame_number,
                timestamp=self.frame_number / self.fps,
                has_ball=has_ball,
                ball_distance=ball_distance
            )
            
            self.current_players[player_id] = player_state
            
            # Update player history
            if player_id not in self.player_history:
                self.player_history[player_id] = deque(maxlen=self.max_player_history)
            self.player_history[player_id].append(player_state)
    
    def _detect_pass_attempts(self) -> List[Dict]:
        """
        Detect when a player attempts to pass (has ball and moves it significantly).
        
        Returns:
            List of pass attempt dictionaries
        """
        attempts = []
        
        # Find players who currently have the ball
        players_with_ball = [
            player_id for player_id, state in self.current_players.items()
            if state.has_ball
        ]
        
        for player_id in players_with_ball:
            current_state = self.current_players[player_id]
            
            # Look for potential receivers (same team, not too far)
            potential_receivers = []
            for other_player_id, other_state in self.current_players.items():
                if other_player_id != player_id:
                    distance = self._calculate_distance(
                        current_state.position, other_state.position
                    )
                    
                    if distance <= self.pass_distance_threshold:
                        potential_receivers.append({
                            'player_id': other_player_id,
                            'team_id': other_state.team_id,
                            'distance': distance,
                            'is_same_team': self._is_same_team(current_state.team_id, other_state.team_id)
                        })
            
            if potential_receivers:
                attempts.append({
                    'passer_id': player_id,
                    'passer_team': current_state.team_id,
                    'passer_position': current_state.position.copy(),
                    'ball_position': self.current_ball_position.copy() if self.current_ball_position is not None else None,
                    'potential_receivers': potential_receivers,
                    'frame_number': self.frame_number,
                    'timestamp': current_state.timestamp
                })
        
        return attempts
    
    def _resolve_pass_attempts(self, new_attempts: List[Dict]):
        """
        Resolve pending pass attempts and determine their outcomes.
        """
        # Process each new attempt
        for attempt in new_attempts:
            self.pending_pass_attempts.append(attempt)
        
        # Check if any pending attempts have been resolved
        resolved_attempts = []
        for attempt in self.pending_pass_attempts[:]:  # Copy list to modify during iteration
            # Check if any potential receiver got the ball
            ball_received = False
            for receiver_info in attempt['potential_receivers']:
                receiver_id = receiver_info['player_id']
                if receiver_id in self.current_players:
                    if self.current_players[receiver_id].has_ball:
                        ball_received = True
                        # Determine if it was successful or intercepted
                        if receiver_info['is_same_team']:
                            # Successful pass
                            pass_record = Pass(
                                passer_id=attempt['passer_id'],
                                receiver_id=receiver_id,
                                passer_team=attempt['passer_team'],
                                receiver_team=receiver_info['team_id'],
                                frame_number=attempt['frame_number'],
                                timestamp=attempt['timestamp'],
                                passer_position=attempt['passer_position'],
                                receiver_position=self.current_players[receiver_id].position.copy(),
                                ball_position=attempt['ball_position'],
                                distance=receiver_info['distance'],
                                pass_type=PassType.SUCCESSFUL
                            )
                            self.passes.append(pass_record)
                        else:
                            # Intercepted pass
                            unsuccessful_pass = UnsuccessfulPass(
                                passer_id=attempt['passer_id'],
                                passer_team=attempt['passer_team'],
                                frame_number=attempt['frame_number'],
                                timestamp=attempt['timestamp'],
                                passer_position=attempt['passer_position'],
                                ball_position=attempt['ball_position'],
                                pass_type=PassType.INTERCEPTED,
                                intended_receiver_id=receiver_id,
                                intended_receiver_team=receiver_info['team_id'],
                                intercepted_by_id=receiver_id,
                                intercepted_by_team=receiver_info['team_id'],
                                distance=receiver_info['distance']
                            )
                            self.unsuccessful_passes.append(unsuccessful_pass)
                        break
            
            # If ball was received, mark attempt as resolved
            if ball_received:
                resolved_attempts.append(attempt)
                self.pending_pass_attempts.remove(attempt)
        
        # Check for out-of-bounds passes (ball lost for too long)
        for attempt in self.pending_pass_attempts[:]:
            frames_since_attempt = self.frame_number - attempt['frame_number']
            if frames_since_attempt > self.out_of_bounds_timeout:
                # Ball went out of bounds
                unsuccessful_pass = UnsuccessfulPass(
                    passer_id=attempt['passer_id'],
                    passer_team=attempt['passer_team'],
                    frame_number=attempt['frame_number'],
                    timestamp=attempt['timestamp'],
                    passer_position=attempt['passer_position'],
                    ball_position=attempt['ball_position'],
                    pass_type=PassType.OUT_OF_BOUNDS
                )
                self.unsuccessful_passes.append(unsuccessful_pass)
                resolved_attempts.append(attempt)
                self.pending_pass_attempts.remove(attempt)
    
    def detect_passes(self) -> Tuple[List[Pass], List[UnsuccessfulPass]]:
        """
        Detect both successful and unsuccessful passes.
        
        Returns:
            Tuple of (new_successful_passes, new_unsuccessful_passes)
        """
        # Detect new pass attempts
        new_attempts = self._detect_pass_attempts()
        
        # Resolve pending attempts
        self._resolve_pass_attempts(new_attempts)
        
        # Return new passes from this frame
        new_successful_passes = [p for p in self.passes if p.frame_number == self.frame_number]
        new_unsuccessful_passes = [p for p in self.unsuccessful_passes if p.frame_number == self.frame_number]
        
        return new_successful_passes, new_unsuccessful_passes
    
    def get_ball_proximity_stats(self) -> Dict[int, Dict]:
        """
        Get statistics about ball proximity for each player.
        
        Returns:
            Dictionary with player ID as key and proximity stats as value
        """
        stats = {}
        
        for player_id, state in self.current_players.items():
            stats[player_id] = {
                'team_id': state.team_id,
                'current_distance': state.ball_distance,
                'has_ball': state.has_ball,
                'position': state.position.tolist(),
                'frame_number': state.frame_number
            }
        
        return stats
    
    def get_pass_statistics(self) -> Dict:
        """
        Get statistics about passes.
        
        Returns:
            Dictionary with pass statistics
        """
        if not self.passes:
            return {
                'total_passes': 0,
                'passes_by_team': {},
                'average_pass_distance': 0.0,
                'passes_per_minute': 0.0
            }
        
        # Calculate statistics
        total_passes = len(self.passes)
        passes_by_team = {}
        total_distance = 0.0
        
        for pass_record in self.passes:
            team_key = f"Team {pass_record.passer_team}"
            if team_key not in passes_by_team:
                passes_by_team[team_key] = 0
            passes_by_team[team_key] += 1
            total_distance += pass_record.distance
        
        # Calculate time duration
        if self.passes:
            duration_minutes = (self.passes[-1].timestamp - self.passes[0].timestamp) / 60.0
            passes_per_minute = total_passes / max(duration_minutes, 1.0)
        else:
            passes_per_minute = 0.0
        
        return {
            'total_passes': total_passes,
            'passes_by_team': passes_by_team,
            'average_pass_distance': total_distance / total_passes,
            'passes_per_minute': passes_per_minute,
            'recent_passes': [
                {
                    'passer_id': p.passer_id,
                    'receiver_id': p.receiver_id,
                    'team': p.passer_team,
                    'distance': p.distance,
                    'frame': p.frame_number
                }
                for p in self.passes[-5:]  # Last 5 passes
            ]
        }
    
    def process_frame(self, 
                     ball_detections: sv.Detections,
                     player_detections: sv.Detections,
                     team_ids: np.ndarray,
                     player_ids: Optional[np.ndarray] = None) -> Tuple[List[Pass], List[UnsuccessfulPass], Dict]:
        """
        Process a single frame and return detected passes and proximity stats.
        
        Args:
            ball_detections: Ball detections for the frame
            player_detections: Player detections for the frame
            team_ids: Team IDs for each player
            player_ids: Optional player IDs for tracking
            
        Returns:
            Tuple of (new_successful_passes, new_unsuccessful_passes, proximity_stats)
        """
        self.frame_number += 1
        
        # Update ball and player positions
        self.update_ball_position(ball_detections)
        self.update_players(player_detections, team_ids, player_ids)
        
        # Detect passes
        new_successful_passes, new_unsuccessful_passes = self.detect_passes()
        
        # Get proximity statistics
        proximity_stats = self.get_ball_proximity_stats()
        
        return new_successful_passes, new_unsuccessful_passes, proximity_stats


class PassAnnotator:
    """
    Annotates frames with pass information and ball proximity.
    """
    
    def __init__(self, 
                 pass_color: Tuple[int, int, int] = (0, 255, 0),  # Green
                 proximity_color: Tuple[int, int, int] = (255, 0, 0),  # Red
                 text_color: Tuple[int, int, int] = (255, 255, 255),  # White
                 line_thickness: int = 2,
                 circle_radius: int = 5):
        """
        Initialize the PassAnnotator.
        
        Args:
            pass_color: Color for pass lines (BGR)
            proximity_color: Color for proximity indicators (BGR)
            text_color: Color for text (BGR)
            line_thickness: Thickness of pass lines
            circle_radius: Radius of proximity circles
        """
        self.pass_color = pass_color
        self.proximity_color = proximity_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.circle_radius = circle_radius
    
    def annotate_passes(self, 
                       frame: np.ndarray, 
                       passes: List[Pass],
                       max_passes_to_show: int = 3) -> np.ndarray:
        """
        Annotate frame with recent passes.
        
        Args:
            frame: Frame to annotate
            passes: List of passes to show
            max_passes_to_show: Maximum number of recent passes to display
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Show recent passes
        recent_passes = passes[-max_passes_to_show:]
        
        for i, pass_record in enumerate(recent_passes):
            # Draw pass line
            cv2.line(annotated_frame,
                    tuple(pass_record.passer_position.astype(int)),
                    tuple(pass_record.receiver_position.astype(int)),
                    self.pass_color,
                    self.line_thickness)
            
            # Draw player markers
            cv2.circle(annotated_frame,
                      tuple(pass_record.passer_position.astype(int)),
                      self.circle_radius,
                      self.pass_color,
                      -1)
            
            cv2.circle(annotated_frame,
                      tuple(pass_record.receiver_position.astype(int)),
                      self.circle_radius,
                      self.pass_color,
                      -1)
            
            # Add pass label
            label = f"Pass {len(passes) - max_passes_to_show + i + 1}"
            cv2.putText(annotated_frame,
                       label,
                       tuple(pass_record.passer_position.astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       self.text_color,
                       1)
        
        return annotated_frame
    
    def annotate_proximity(self, 
                          frame: np.ndarray, 
                          proximity_stats: Dict[int, Dict],
                          show_distance: bool = True) -> np.ndarray:
        """
        Annotate frame with ball proximity information.
        
        Args:
            frame: Frame to annotate
            proximity_stats: Proximity statistics for each player
            show_distance: Whether to show distance text
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for player_id, stats in proximity_stats.items():
            position = np.array(stats['position']).astype(int)
            
            # Draw proximity circle
            color = self.proximity_color if stats['has_ball'] else (128, 128, 128)
            cv2.circle(annotated_frame,
                      tuple(position),
                      self.circle_radius,
                      color,
                      -1)
            
            # Add distance text
            if show_distance:
                distance_text = f"{stats['current_distance']:.1f}px"
                cv2.putText(annotated_frame,
                           distance_text,
                           tuple(position + [10, 10]),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           self.text_color,
                           1)
        
        return annotated_frame 