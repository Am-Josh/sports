import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the main script
from main import main, Mode

if __name__ == "__main__":
    print("Soccer Analysis with Pass Tracking and Ball Proximity Detection")
    print("=" * 60)
    print("Available modes:")
    print("1. PLAYER_DETECTION - Detect players, goalkeepers, referees, and ball")
    print("2. BALL_DETECTION - Track the ball specifically")
    print("3. PITCH_DETECTION - Detect soccer field boundaries")
    print("4. PLAYER_TRACKING - Track players across frames")
    print("5. TEAM_CLASSIFICATION - Classify players into teams")
    print("6. RADAR - Create radar-like visualization")
    print("7. PASS_TRACKING - Track passes and ball proximity (NEW!)")
    print()
    
    # You can change the mode here
    selected_mode = Mode.PASS_TRACKING  # Change this to test different modes
    
    print(f"Running mode: {selected_mode.value}")
    print("Processing video: data/2e57b9_0.mp4")
    print("Output: data/2e57b9_0-pass-tracking.mp4")
    print()
    
    if selected_mode == Mode.PASS_TRACKING:
        print("PASS_TRACKING mode features:")
        print("- Tracks ball proximity to each player")
        print("- Detects successful passes between same-team players")
        print("- Shows real-time statistics (total passes, passes/min, avg distance)")
        print("- Visual indicators for players with ball")
        print("- Pass lines showing recent successful passes")
        print()
    
    # Run the analysis
    main(
        source_video_path="data/2e57b9_0.mp4",
        target_video_path=f"data/2e57b9_0-{selected_mode.value.lower()}.mp4",
        device="cpu",
        mode=selected_mode
    )
    
    print("Analysis complete!")
    print(f"Output saved to: data/2e57b9_0-{selected_mode.value.lower()}.mp4") 