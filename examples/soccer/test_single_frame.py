#!/usr/bin/env python3
"""
Simple test script for enhanced pass tracking on a single frame.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
from sports.common.pass_tracker import PassTracker, PassAnnotator
import supervision as sv


def create_test_frame():
    """Create a simple test frame with players and ball."""
    # Create a green soccer field background
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]  # Forest green
    
    # Add field lines
    cv2.line(img, (0, 200), (600, 200), (255, 255, 255), 2)  # Center line
    cv2.circle(img, (300, 200), 30, (255, 255, 255), 2)      # Center circle
    
    # Add players (red and blue teams)
    cv2.circle(img, (150, 150), 15, (255, 0, 0), -1)    # Red player (Team 0)
    cv2.circle(img, (450, 150), 15, (0, 0, 255), -1)    # Blue player (Team 1)
    cv2.circle(img, (150, 250), 15, (255, 0, 0), -1)    # Red player (Team 0)
    cv2.circle(img, (450, 250), 15, (0, 0, 255), -1)    # Blue player (Team 1)
    
    # Add ball (yellow)
    cv2.circle(img, (300, 200), 8, (255, 255, 0), -1)
    
    # Add player labels
    cv2.putText(img, "P0", (140, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "P1", (440, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "P2", (140, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "P3", (440, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def test_enhanced_pass_tracking():
    """Test enhanced pass tracking on a single frame."""
    print("Testing Enhanced Pass Tracking on Single Frame")
    print("=" * 50)
    
    # Create test frame
    print("Creating test frame...")
    frame = create_test_frame()
    
    # Save original frame
    cv2.imwrite("test_frame_original.jpg", frame)
    print("Original frame saved: test_frame_original.jpg")
    
    # Initialize pass tracker
    tracker = PassTracker(
        ball_proximity_threshold=50.0,
        pass_distance_threshold=200.0,
        out_of_bounds_timeout=3
    )
    
    annotator = PassAnnotator()
    
    # Create synthetic detections
    print("Creating synthetic detections...")
    
    # Player positions and bounding boxes
    player_positions = [
        [150, 150],  # P0 (Team 0)
        [450, 150],  # P1 (Team 1)
        [150, 250],  # P2 (Team 0)
        [450, 250],  # P3 (Team 1)
    ]
    
    # Create bounding boxes (30x30 around each player)
    player_boxes = []
    for pos in player_positions:
        x, y = pos
        player_boxes.append([x - 15, y - 15, x + 15, y + 15])
    
    player_boxes = np.array(player_boxes)
    
    # Ball position and bounding box
    ball_position = [300, 200]
    ball_boxes = np.array([[ball_position[0] - 8, ball_position[1] - 8, 
                           ball_position[0] + 8, ball_position[1] + 8]])
    
    # Create detections
    player_detections = sv.Detections(
        xyxy=player_boxes,
        class_id=[2, 2, 2, 2],  # All players
        confidence=[1.0, 1.0, 1.0, 1.0]
    )
    
    ball_detections = sv.Detections(
        xyxy=ball_boxes,
        class_id=[0],  # Ball
        confidence=[1.0]
    )
    
    # Team assignments
    team_ids = np.array([0, 1, 0, 1])  # Red, Blue, Red, Blue
    player_ids = np.array([0, 1, 2, 3])
    
    print(f"Created {len(player_detections)} player detections")
    print(f"Created {len(ball_detections)} ball detections")
    
    # Process frame
    print("Processing frame with enhanced pass tracker...")
    new_successful_passes, new_unsuccessful_passes, proximity_stats = tracker.process_frame(
        ball_detections=ball_detections,
        player_detections=player_detections,
        team_ids=team_ids,
        player_ids=player_ids
    )
    
    # Print results
    print(f"\nResults:")
    print(f"Ball position: {ball_position}")
    print(f"Players with ball: {[pid for pid, stats in proximity_stats.items() if stats['has_ball']]}")
    
    for player_id, stats in proximity_stats.items():
        print(f"  Player {player_id} (Team {stats['team_id']}): distance={stats['current_distance']:.1f}px, has_ball={stats['has_ball']}")
    
    if new_successful_passes:
        print(f"New successful passes: {len(new_successful_passes)}")
    
    if new_unsuccessful_passes:
        print(f"New unsuccessful passes: {len(new_unsuccessful_passes)}")
    
    # Get statistics
    stats = tracker.get_pass_statistics()
    print(f"\nStatistics:")
    print(f"Total pass attempts: {stats['total_attempts']}")
    print(f"Successful passes: {stats['successful_passes']}")
    print(f"Unsuccessful passes: {stats['unsuccessful_passes']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    
    # Annotate frame
    print("\nAnnotating frame...")
    annotated_frame = frame.copy()
    
    # Draw player detections with team colors
    color_lookup = team_ids
    labels = [f"P{i}" for i in range(len(player_detections))]
    
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF']),
        thickness=2
    )
    ellipse_label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF']),
        text_color=sv.Color.from_hex('#FFFFFF'),
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    
    annotated_frame = ellipse_annotator.annotate(
        annotated_frame, player_detections, custom_color_lookup=color_lookup)
    annotated_frame = ellipse_label_annotator.annotate(
        annotated_frame, player_detections, labels, custom_color_lookup=color_lookup)
    
    # Annotate ball proximity
    annotated_frame = annotator.annotate_proximity(
        annotated_frame, proximity_stats, show_distance=True)
    
    # Add statistics overlay
    stats_text = [
        f"Players Detected: {len(player_detections)}",
        f"Balls Detected: {len(ball_detections)}",
        f"Players with Ball: {sum(1 for stats in proximity_stats.values() if stats['has_ball'])}",
        f"Ball Proximity Threshold: 50px"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add legend
    legend_text = [
        "LEGEND:",
        "Red circles: Team 0 players",
        "Blue circles: Team 1 players",
        "Yellow circle: Ball",
        "Blue circles: Players with ball proximity"
    ]
    
    for i, text in enumerate(legend_text):
        cv2.putText(annotated_frame, text, (10, 350 + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save annotated frame
    cv2.imwrite("test_frame_annotated.jpg", annotated_frame)
    print("Annotated frame saved: test_frame_annotated.jpg")
    
    # Display frame
    print("\nDisplaying frame (press any key to close)...")
    cv2.imshow("Enhanced Pass Tracking - Single Frame", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("Enhanced Pass Tracking on Single Frame - Complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_enhanced_pass_tracking() 