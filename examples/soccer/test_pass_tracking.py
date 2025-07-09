import os
import sys
import numpy as np
import supervision as sv

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sports.common.pass_tracker import PassTracker, PassAnnotator

def create_sample_detections():
    """Create sample player and ball detections for testing."""
    
    # Sample player positions (x, y coordinates)
    player_positions = np.array([
        [100, 200],  # Player 1
        [300, 200],  # Player 2 (same team)
        [500, 200],  # Player 3 (different team)
        [200, 400],  # Player 4 (same team as Player 1)
    ])
    
    # Sample team IDs (0 = team 1, 1 = team 2)
    team_ids = np.array([0, 0, 1, 0])
    
    # Sample player IDs
    player_ids = np.array([1, 2, 3, 4])
    
    # Create detections
    detections = sv.Detections(
        xyxy=np.column_stack([
            player_positions[:, 0] - 20,  # x1
            player_positions[:, 1] - 40,  # y1
            player_positions[:, 0] + 20,  # x2
            player_positions[:, 1] + 40,  # y2
        ]),
        tracker_id=player_ids
    )
    
    return detections, team_ids

def create_sample_ball_detections(ball_position):
    """Create sample ball detection."""
    detections = sv.Detections(
        xyxy=np.array([
            [ball_position[0] - 10, ball_position[1] - 10,
             ball_position[0] + 10, ball_position[1] + 10]
        ])
    )
    return detections

def test_pass_tracking():
    """Test the pass tracking functionality."""
    
    print("Testing Pass Tracking and Ball Proximity Detection")
    print("=" * 50)
    
    # Initialize pass tracker
    pass_tracker = PassTracker(
        ball_proximity_threshold=50.0,
        pass_distance_threshold=200.0,
        min_pass_frames=2,
        max_player_history=10
    )
    pass_tracker.update_fps(30.0)
    
    # Create sample data
    player_detections, team_ids = create_sample_detections()
    
    print(f"Created {len(player_detections)} players")
    print(f"Team IDs: {team_ids}")
    print()
    
    # Simulate ball movement and pass detection
    ball_positions = [
        [100, 200],  # Ball near Player 1
        [150, 200],  # Ball moving towards Player 2
        [300, 200],  # Ball near Player 2 (potential pass received)
        [350, 200],  # Ball moving towards Player 4
        [200, 400],  # Ball near Player 4 (potential pass received)
    ]
    
    for i, ball_pos in enumerate(ball_positions):
        print(f"Frame {i+1}: Ball at {ball_pos}")
        
        # Create ball detection
        ball_detections = create_sample_ball_detections(ball_pos)
        
        # Process frame
        new_passes, proximity_stats = pass_tracker.process_frame(
            ball_detections=ball_detections,
            player_detections=player_detections,
            team_ids=team_ids,
            player_ids=player_detections.tracker_id
        )
        
        # Print proximity stats
        print("  Ball proximity:")
        for player_id, stats in proximity_stats.items():
            status = "HAS BALL" if stats['has_ball'] else f"{stats['current_distance']:.1f}px away"
            print(f"    Player {player_id} (Team {stats['team_id']}): {status}")
        
        # Print new passes
        if new_passes:
            print("  NEW PASSES DETECTED:")
            for pass_record in new_passes:
                print(f"    Pass from Player {pass_record.passer_id} to Player {pass_record.receiver_id}")
                print(f"    Distance: {pass_record.distance:.1f}px")
        
        print()
    
    # Print final statistics
    stats = pass_tracker.get_pass_statistics()
    print("Final Statistics:")
    print(f"  Total passes: {stats['total_passes']}")
    print(f"  Passes by team: {stats['passes_by_team']}")
    print(f"  Average pass distance: {stats['average_pass_distance']:.1f}px")
    print(f"  Passes per minute: {stats['passes_per_minute']:.1f}")
    
    if stats['recent_passes']:
        print("  Recent passes:")
        for pass_info in stats['recent_passes']:
            print(f"    Pass {pass_info['passer_id']} -> {pass_info['receiver_id']} "
                  f"(Team {pass_info['team']}, {pass_info['distance']:.1f}px)")

if __name__ == "__main__":
    test_pass_tracking() 