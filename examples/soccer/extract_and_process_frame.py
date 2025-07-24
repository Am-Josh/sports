#!/usr/bin/env python3
"""
Extract a single frame from a video and process it with enhanced pass tracking.
This is much faster than processing the entire video.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.pass_tracker import PassTracker, PassAnnotator


def extract_frame_from_video(video_path, frame_number=0):
    """Extract a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps} FPS")
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None
    
    print(f"Successfully extracted frame {frame_number}")
    return frame


def process_single_frame_with_pass_tracking(frame, output_path="processed_frame.jpg"):
    """Process a single frame with enhanced pass tracking."""
    print("Processing frame with pass tracking...")
    
    # Load YOLO models
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
    BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')
    
    try:
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
        ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH)
        print("YOLO models loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return None
    
    # Detect players and ball
    print("Detecting players and ball...")
    player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
    ball_result = ball_detection_model(frame, verbose=False)[0]
    
    player_detections = sv.Detections.from_ultralytics(player_result)
    ball_detections = sv.Detections.from_ultralytics(ball_result)
    
    print(f"Detected {len(player_detections)} players")
    print(f"Detected {len(ball_detections)} balls")
    
    # For demonstration, create synthetic team assignments
    # In a real scenario, you'd use team classification
    if len(player_detections) > 0:
        team_ids = np.array([0, 0, 1, 1] * (len(player_detections) // 4 + 1))[:len(player_detections)]
        player_ids = np.arange(len(player_detections))
    else:
        print("No players detected!")
        return None
    
    # Initialize pass tracker
    tracker = PassTracker(
        ball_proximity_threshold=50.0,
        pass_distance_threshold=200.0,
        out_of_bounds_timeout=3
    )
    
    annotator = PassAnnotator()
    
    # Process the frame
    new_successful_passes, new_unsuccessful_passes, proximity_stats = tracker.process_frame(
        ball_detections=ball_detections,
        player_detections=player_detections,
        team_ids=team_ids,
        player_ids=player_ids
    )
    
    # Annotate the frame
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
    
    # Add statistics
    stats = tracker.get_pass_statistics()
    stats_text = [
        f"Players Detected: {len(player_detections)}",
        f"Balls Detected: {len(ball_detections)}",
        f"Players with Ball: {sum(1 for stats in proximity_stats.values() if stats['has_ball'])}",
        f"Ball Proximity Threshold: 50px"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save the annotated frame
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved to: {output_path}")
    
    # Display the frame
    print("Displaying frame (press any key to close)...")
    cv2.imshow("Pass Tracking - Single Frame", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return annotated_frame


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and process a single frame with pass tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to extract (default: 0)')
    parser.add_argument('--output', type=str, default='processed_frame.jpg', help='Output image path')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found!")
        return
    
    print(f"Extracting frame {args.frame} from {args.video}")
    
    # Extract frame
    frame = extract_frame_from_video(args.video, args.frame)
    if frame is None:
        return
    
    # Process frame
    annotated_frame = process_single_frame_with_pass_tracking(frame, args.output)
    
    if annotated_frame is not None:
        print("Frame processing completed successfully!")
    else:
        print("Frame processing failed!")


if __name__ == "__main__":
    main() 