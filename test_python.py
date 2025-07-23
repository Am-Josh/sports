#!/usr/bin/env python3
"""
Simple test script to verify Python installation and basic functionality
"""

import sys
import os

def main():
    print("=" * 50)
    print("Python Installation Test")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we can import basic packages
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy not available")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV not available")
    
    try:
        import supervision
        print("✓ Supervision imported successfully")
    except ImportError:
        print("✗ Supervision not available")
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main() 