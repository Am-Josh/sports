#!/usr/bin/env python3
"""
Test script to debug import issues
"""

import os
import sys

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
print("Project root:", project_root)
sys.path.insert(0, project_root)

print("Updated Python path:", sys.path)

try:
    print("Trying to import sports module...")
    import sports
    print("✓ sports module imported successfully")
    
    print("Trying to import sports.annotators.soccer...")
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    print("✓ sports.annotators.soccer imported successfully")
    
    print("Trying to import sports.common.ball...")
    from sports.common.ball import BallTracker, BallAnnotator
    print("✓ sports.common.ball imported successfully")
    
    print("Trying to import sports.common.team...")
    from sports.common.team import TeamClassifier
    print("✓ sports.common.team imported successfully")
    
    print("Trying to import sports.common.view...")
    from sports.common.view import ViewTransformer
    print("✓ sports.common.view imported successfully")
    
    print("Trying to import sports.configs.soccer...")
    from sports.configs.soccer import SoccerPitchConfiguration
    print("✓ sports.configs.soccer imported successfully")
    
    print("All imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Error type: {type(e)}")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    print(f"Error type: {type(e)}") 