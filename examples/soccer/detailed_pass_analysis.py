#!/usr/bin/env python3
"""
Detailed Pass Analysis - Comprehensive Statistics
This script provides detailed statistics for all pass types you requested:

✅ SUCCESSFUL PASSES: Same team → same team
❌ INTERCEPTED PASSES: Same team → opponent team  
❌ OUT OF BOUNDS: Ball goes out of play
❌ MISSED PASSES: Ball doesn't reach any player
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import json
from datetime import datetime

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sports.common.pass_tracker import (
    PassTracker, 
    PassAnnotator, 
    PassType
)
import supervision as sv

# Model paths
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

def main():
    print("🎯 Detailed Pass Analysis")
    print("=" * 50)
    print("📊 Tracking all pass types:")
    print("   ✅ SUCCESSFUL: Same team → same team")
    print("   ❌ INTERCEPTED: Same team → opponent")
    print("   ❌ OUT OF BOUNDS: Ball goes out of play")
    print("   ❌ MISSED: Ball doesn't reach any player")
    print("=" * 50)
    
    # Initialize tracker and annotator
    tracker = PassTracker(
        ball_proximity_threshold=60.0,
        pass_distance_threshold=300.0,
        min_pass_frames=8,
        out_of_bounds_timeout=20
    )
    annotator = PassAnnotator()
    
    # Initialize models
    print("Loading the Ai sys.")
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH)
    
    # Video paths
    source_video_path = os.path.join(PARENT_DIR, "data/2e57b9_0.mp4")
    target_video_path = os.path.join(PARENT_DIR, "data/2e57b9_0-detailed-analysis.mp4")
    
    # Get video properties
    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Update tracker with video FPS
    tracker.update_fps(fps)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, (width, height))
    
    # Statistics tracking
    pass_history = []
    total_passes = 0
    successful_passes = 0
    intercepted_passes = 0
    out_of_bounds_passes = 0
    missed_passes = 0
    
    # Process frames
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    print("Processing frames...")
    for frame_num, frame in enumerate(tqdm(frame_generator, total=frame_count)):
        try:
            # Detect players and ball
            player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            player_detections = sv.Detections.from_ultralytics(player_result)
            
            ball_result = ball_detection_model(frame, imgsz=640, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            
            # Simple team assignment
            if len(player_detections) > 0:
                team_ids = np.array([i % 2 for i in range(len(player_detections))])
                player_ids = np.arange(len(player_detections))
            else:
                team_ids = np.array([])
                player_ids = np.array([])
            
            # Process frame with enhanced pass tracker
            new_successful_passes, new_unsuccessful_passes, proximity_stats = tracker.process_frame(
                ball_detections=ball_detections,
                player_detections=player_detections,
                team_ids=team_ids,
                player_ids=player_ids
            )
            
            # Update statistics
            for pass_obj in new_successful_passes:
                successful_passes += 1
                total_passes += 1
                pass_history.append({
                    'frame': frame_num,
                    'type': 'SUCCESSFUL',
                    'passer_id': pass_obj.passer_id,
                    'receiver_id': pass_obj.receiver_id,
                    'passer_team': pass_obj.passer_team,
                    'receiver_team': pass_obj.receiver_team,
                    'distance': pass_obj.distance
                })
            
            for pass_obj in new_unsuccessful_passes:
                total_passes += 1
                if pass_obj.pass_type == PassType.INTERCEPTED:
                    intercepted_passes += 1
                elif pass_obj.pass_type == PassType.OUT_OF_BOUNDS:
                    out_of_bounds_passes += 1
                elif pass_obj.pass_type == PassType.MISSED:
                    missed_passes += 1
                
                pass_history.append({
                    'frame': frame_num,
                    'type': pass_obj.pass_type.value.upper(),
                    'passer_id': pass_obj.passer_id,
                    'passer_team': pass_obj.passer_team,
                    'intercepted_by_id': pass_obj.intercepted_by_id,
                    'intercepted_by_team': pass_obj.intercepted_by_team,
                    'distance': pass_obj.distance
                })
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            # Draw player detections
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            if len(player_detections) > 0:
                annotated_frame = box_annotator.annotate(annotated_frame, player_detections)
                
                # Add labels with team info and ball proximity
                labels = []
                for i, (team_id, player_id) in enumerate(zip(team_ids, player_ids)):
                    has_ball = proximity_stats.get(player_id, {}).get('has_ball', False)
                    distance = proximity_stats.get(player_id, {}).get('ball_distance', float('inf'))
                    
                    if has_ball:
                        label = f"P{player_id} T{team_id} 🏀"
                    else:
                        label = f"P{player_id} T{team_id} {distance:.0f}px"
                    labels.append(label)
                
                annotated_frame = label_annotator.annotate(annotated_frame, player_detections, labels)
            
            # Annotate passes
            annotated_frame = annotator.annotate_passes(
                frame=annotated_frame,
                successful_passes=new_successful_passes,
                unsuccessful_passes=new_unsuccessful_passes,
                max_passes_to_show=5
            )
            
            # Annotate ball proximity
            annotated_frame = annotator.annotate_proximity(
                frame=annotated_frame,
                proximity_stats=proximity_stats,
                show_distance=True
            )
            
            # Add comprehensive statistics overlay
            success_rate = (successful_passes / max(total_passes, 1) * 100)
            stats_text = [
                f"Frame: {frame_num}/{frame_count}",
                f"Total Passes: {total_passes}",
                f"✅ Successful: {successful_passes} ({success_rate:.1f}%)",
                f"❌ Intercepted: {intercepted_passes}",
                f"❌ Out of Bounds: {out_of_bounds_passes}",
                f"❌ Missed: {missed_passes}"
            ]
            
            # Draw statistics with color coding
            for i, text in enumerate(stats_text):
                color = (255, 255, 255)  # White
                if "✅" in text:
                    color = (0, 255, 0)  # Green for successful
                elif "❌" in text:
                    color = (0, 0, 255)  # Red for unsuccessful
                
                cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Print real-time updates
            if new_successful_passes or new_unsuccessful_passes:
                print(f"\n🎯 Frame {frame_num}:")
                for pass_obj in new_successful_passes:
                    print(f"  ✅ SUCCESSFUL: Player {pass_obj.passer_id} (T{pass_obj.passer_team}) → Player {pass_obj.receiver_id} (T{pass_obj.receiver_team})")
                for pass_obj in new_unsuccessful_passes:
                    if pass_obj.pass_type == PassType.INTERCEPTED:
                        print(f"  ❌ INTERCEPTED: Player {pass_obj.passer_id} (T{pass_obj.passer_team}) → Player {pass_obj.intercepted_by_id} (T{pass_obj.intercepted_by_team})")
                    elif pass_obj.pass_type == PassType.OUT_OF_BOUNDS:
                        print(f"  ❌ OUT OF BOUNDS: Player {pass_obj.passer_id} (T{pass_obj.passer_team})")
                    elif pass_obj.pass_type == PassType.MISSED:
                        print(f"  ❌ MISSED: Player {pass_obj.passer_id} (T{pass_obj.passer_team})")
                
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            out.write(frame)
            continue
    
    # Release video writer
    out.release()
    
    # Generate final comprehensive statistics
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE PASS STATISTICS")
    print("=" * 60)
    print(f"📈 TOTAL PASSES: {total_passes}")
    print(f"✅ SUCCESSFUL PASSES: {successful_passes} ({(successful_passes/max(total_passes,1)*100):.1f}%)")
    print(f"❌ INTERCEPTED PASSES: {intercepted_passes} ({(intercepted_passes/max(total_passes,1)*100):.1f}%)")
    print(f"❌ OUT OF BOUNDS: {out_of_bounds_passes} ({(out_of_bounds_passes/max(total_passes,1)*100):.1f}%)")
    print(f"❌ MISSED PASSES: {missed_passes} ({(missed_passes/max(total_passes,1)*100):.1f}%)")
    
    # Player-specific statistics
    print("\n👥 PLAYER-SPECIFIC STATISTICS:")
    player_stats = {}
    for pass_event in pass_history:
        passer_id = pass_event['passer_id']
        if passer_id not in player_stats:
            player_stats[passer_id] = {'successful': 0, 'unsuccessful': 0, 'total': 0}
        
        player_stats[passer_id]['total'] += 1
        if pass_event['type'] == 'SUCCESSFUL':
            player_stats[passer_id]['successful'] += 1
        else:
            player_stats[passer_id]['unsuccessful'] += 1
    
    for player_id, stats in player_stats.items():
        success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  Player {player_id}: {stats['successful']}/{stats['total']} successful ({success_rate:.1f}%)")
    
    # Team statistics
    print("\n🏆 TEAM PERFORMANCE:")
    team_stats = {}
    for pass_event in pass_history:
        team_id = pass_event['passer_team']
        if team_id not in team_stats:
            team_stats[team_id] = {'successful': 0, 'unsuccessful': 0, 'total': 0}
        
        team_stats[team_id]['total'] += 1
        if pass_event['type'] == 'SUCCESSFUL':
            team_stats[team_id]['successful'] += 1
        else:
            team_stats[team_id]['unsuccessful'] += 1
    
    for team_id, stats in team_stats.items():
        success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  Team {team_id}: {stats['successful']}/{stats['total']} successful ({success_rate:.1f}%)")
    
    # Save detailed statistics
    detailed_stats = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_passes': total_passes,
            'successful_passes': successful_passes,
            'intercepted_passes': intercepted_passes,
            'out_of_bounds_passes': out_of_bounds_passes,
            'missed_passes': missed_passes,
            'success_rate': (successful_passes / max(total_passes, 1) * 100)
        },
        'player_statistics': player_stats,
        'team_statistics': team_stats,
        'pass_history': pass_history
    }
    
    stats_file = "detailed_pass_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2, default=str)
    
    print(f"\n💾 Detailed statistics saved to: {stats_file}")
    print(f"📹 Output video saved to: {target_video_path}")
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main() 