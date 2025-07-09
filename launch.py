#!/usr/bin/env python3
"""
Simple launcher script for the Soccer Analysis Project
Run this file directly to execute the soccer analysis
"""

import os
import sys

def main():
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Import and run the main script using absolute import
    from examples.soccer.main import main, Mode
    
    print("Starting Soccer Analysis...")
    print("Processing video: examples/soccer/data/2e57b9_0.mp4")
    print("Output: examples/soccer/data/2e57b9_0-player-detection.mp4")
    
    # Run player detection
    main(
        source_video_path="examples/soccer/data/2e57b9_0.mp4",
        target_video_path="examples/soccer/data/2e57b9_0-player-detection.mp4",
        device="cpu",
        mode=Mode.PLAYER_DETECTION
    )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 