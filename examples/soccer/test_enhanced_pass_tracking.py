#!/usr/bin/env python3
"""
Test script for enhanced pass tracking functionality.
This script demonstrates tracking of both successful and unsuccessful passes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import cv2
from sports.common.enhanced_pass_tracker import (
    EnhancedPassTracker, 
    EnhancedPassAnnotator, 
    PassType
)
import supervision as sv


def create_synthetic_data():
    """
    Create synthetic player and ball data to test pass tracking.
    This simulates a scenario with both successful and unsuccessful passes.
    """
    # Create synthetic player detections
    player_positions = [
        [100, 200],  # Player 0 (Team 0) - starts with ball
        [300, 200],  # Player 1 (Team 0) - teammate
        [500, 200],  # Player 2 (Team 1) - opponent
        [700, 200],  # Player 3 (Team 1) - opponent
    ]
    
    team_ids = [0, 0, 1, 1]  # Team assignments
    player_ids = [0, 1, 2, 3]
    
    # Ball positions over time (simulating different scenarios)
    ball_positions = [
        [100, 200],   # Frame 0: Ball with Player 0
        [150, 200],   # Frame 1: Ball moving towards Player 1 (successful pass)
        [300, 200],   # Frame 2: Ball with Player 1 (successful pass completed)
        [350, 200],   # Frame 3: Ball moving towards Player 2 (intercepted pass)
        [500, 200],   # Frame 4: Ball with Player 2 (intercepted pass completed)
        [600, 200],   # Frame 5: Ball moving out of bounds
        [800, 200],   # Frame 6: Ball out of bounds
        None,         # Frame 7: Ball lost (out of bounds)
        None,         # Frame 8: Ball still lost
        [100, 200],   # Frame 9: Ball back in play with Player 0
    ]
    
    return player_positions, team_ids, player_ids, ball_positions


def create_detections_from_positions(positions, class_ids=None):
    """Create supervision Detections from positions."""
    if not positions:
        return sv.Detections.empty()
    
    # Convert positions to bounding boxes (assuming 50x50 boxes)
    boxes = []
    for pos in positions:
        x, y = pos
        boxes.append([x - 25, y - 25, x + 25, y + 25])  # [x1, y1, x2, y2]
    
    boxes = np.array(boxes)
    
    if class_ids is None:
        class_ids = np.zeros(len(positions))
    
    return sv.Detections(
        xyxy=boxes,
        class_id=class_ids,
        confidence=np.ones(len(positions))
    )


def test_enhanced_pass_tracking():
    """Test the enhanced pass tracking system."""
    print("Testing Enhanced Pass Tracking System")
    print("=" * 50)
    
    # Initialize the enhanced pass tracker
    tracker = EnhancedPassTracker(
        ball_proximity_threshold=50.0,
        pass_distance_threshold=200.0,
        out_of_bounds_timeout=3  # Short timeout for testing
    )
    
    # Initialize the annotator
    annotator = EnhancedPassAnnotator()
    
    # Get synthetic data
    player_positions, team_ids, player_ids, ball_positions = create_synthetic_data()
    
    # Process each frame
    for frame_num, ball_pos in enumerate(ball_positions):
        print(f"\nFrame {frame_num}:")
        
        # Create player detections
        player_detections = create_detections_from_positions(player_positions)
        
        # Create ball detections
        if ball_pos is not None:
            ball_detections = create_detections_from_positions([ball_pos])
        else:
            ball_detections = sv.Detections.empty()
        
        # Process the frame
        new_successful_passes, new_unsuccessful_passes, proximity_stats = tracker.process_frame(
            ball_detections=ball_detections,
            player_detections=player_detections,
            team_ids=np.array(team_ids),
            player_ids=np.array(player_ids)
        )
        
        # Print results
        print(f"  Ball position: {ball_pos}")
        print(f"  Players with ball: {[pid for pid, stats in proximity_stats.items() if stats['has_ball']]}")
        
        if new_successful_passes:
            print(f"  New successful passes: {len(new_successful_passes)}")
            for p in new_successful_passes:
                print(f"    Player {p.passer_id} (Team {p.passer_team}) → Player {p.receiver_id} (Team {p.receiver_team})")
        
        if new_unsuccessful_passes:
            print(f"  New unsuccessful passes: {len(new_unsuccessful_passes)}")
            for p in new_unsuccessful_passes:
                print(f"    Player {p.passer_id} (Team {p.passer_team}) - {p.pass_type.value}")
                if p.intercepted_by_id is not None:
                    print(f"      Intercepted by Player {p.intercepted_by_id} (Team {p.intercepted_by_team})")
    
    # Get final statistics
    stats = tracker.get_pass_statistics()
    
    print("\n" + "=" * 50)
    print("FINAL STATISTICS:")
    print(f"Total pass attempts: {stats['total_attempts']}")
    print(f"Successful passes: {stats['successful_passes']}")
    print(f"Unsuccessful passes: {stats['unsuccessful_passes']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average pass distance: {stats['average_pass_distance']:.1f} pixels")
    
    if stats['unsuccessful_by_type']:
        print("\nUnsuccessful passes by type:")
        for pass_type, count in stats['unsuccessful_by_type'].items():
            print(f"  {pass_type}: {count}")
    
    print("\nRecent successful passes:")
    for p in stats['recent_successful']:
        print(f"  Player {p['passer_id']} → Player {p['receiver_id']} (Team {p['team']})")
    
    print("\nRecent unsuccessful passes:")
    for p in stats['recent_unsuccessful']:
        print(f"  Player {p['passer_id']} - {p['pass_type']} (Team {p['team']})")


def test_visual_annotation():
    """Test the visual annotation capabilities."""
    print("\n" + "=" * 50)
    print("Testing Visual Annotation")
    print("=" * 50)
    
    # Create a test frame
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    
    # Initialize tracker and annotator
    tracker = EnhancedPassTracker()
    annotator = EnhancedPassAnnotator()
    
    # Create some test passes
    from sports.common.enhanced_pass_tracker import Pass, UnsuccessfulPass
    
    # Successful pass
    successful_pass = Pass(
        passer_id=0,
        receiver_id=1,
        passer_team=0,
        receiver_team=0,
        frame_number=1,
        timestamp=1.0,
        passer_position=np.array([100, 200]),
        receiver_position=np.array([300, 200]),
        ball_position=np.array([200, 200]),
        distance=200.0
    )
    
    # Intercepted pass
    intercepted_pass = UnsuccessfulPass(
        passer_id=1,
        passer_team=0,
        frame_number=2,
        timestamp=2.0,
        passer_position=np.array([300, 200]),
        ball_position=np.array([400, 200]),
        pass_type=PassType.INTERCEPTED,
        intercepted_by_id=2,
        intercepted_by_team=1
    )
    
    # Out of bounds pass
    out_of_bounds_pass = UnsuccessfulPass(
        passer_id=2,
        passer_team=1,
        frame_number=3,
        timestamp=3.0,
        passer_position=np.array([500, 200]),
        ball_position=np.array([750, 200]),
        pass_type=PassType.OUT_OF_BOUNDS
    )
    
    # Test proximity stats
    proximity_stats = {
        0: {'team_id': 0, 'current_distance': 25.0, 'has_ball': True, 'position': [100, 200], 'frame_number': 1},
        1: {'team_id': 0, 'current_distance': 150.0, 'has_ball': False, 'position': [300, 200], 'frame_number': 1},
        2: {'team_id': 1, 'current_distance': 200.0, 'has_ball': False, 'position': [500, 200], 'frame_number': 1},
    }
    
    # Annotate the frame
    annotated_frame = annotator.annotate_passes(
        frame, 
        [successful_pass], 
        [intercepted_pass, out_of_bounds_pass]
    )
    
    annotated_frame = annotator.annotate_proximity(annotated_frame, proximity_stats)
    
    # Save the annotated frame
    output_path = "test_enhanced_annotation.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved to: {output_path}")
    
    # Print what should be visible
    print("\nVisual elements in the annotated frame:")
    print("- Green line: Successful pass (Player 0 → Player 1)")
    print("- Red line: Intercepted pass (Player 1 → Player 2)")
    print("- Magenta line: Out of bounds pass (Player 2)")
    print("- Blue circle: Player with ball (Player 0)")
    print("- Gray circles: Players without ball (Players 1, 2)")


if __name__ == "__main__":
    # Test the enhanced pass tracking
    test_enhanced_pass_tracking()
    
    # Test visual annotation
    test_visual_annotation()
    
    print("\n" + "=" * 50)
    print("Enhanced Pass Tracking Test Complete!")
    print("=" * 50) 