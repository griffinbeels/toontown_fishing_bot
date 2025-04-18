#!/usr/bin/env python3
"""
DG Fish Detection Enhancement Tool

This script applies the optimized DG fish detection parameters to your screen_capture.py file.
Run this before using screen_capture.py if you want to use the enhanced DG detection.
"""

import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime

def main():
    # Check if screen_capture.py exists
    screen_capture_path = Path("screen_capture.py")
    if not screen_capture_path.exists():
        print("Error: screen_capture.py not found in the current directory.")
        print("This tool must be run from the same directory as screen_capture.py.")
        return False
        
    # Create backup
    backup_path = f"screen_capture.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(screen_capture_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the screen_capture.py file
    with open(screen_capture_path, "r") as f:
        content = f.read()
    
    # Check if this appears to be the right file
    if "fish_detection" not in content or "FishDetector" not in content:
        print("Warning: screen_capture.py does not appear to contain fish detection code.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return False
    
    # Apply optimizations
    modified_content = apply_optimizations(content)
    
    # Write the modified file
    with open(screen_capture_path, "w") as f:
        f.write(modified_content)
    
    print("Successfully enhanced DG fish detection in screen_capture.py!")
    print("Run your script as usual, and it should have better fish detection in DG.")
    return True

def apply_optimizations(content):
    # Find the initialization of the FishDetector
    detector_init_pattern = r"(\s*)(fish_detector\s*=\s*FishDetector\(\))"
    
    # Replacement to add DG-specific parameter overrides
    replacement = r"""\1\2
\1# Apply optimized DG parameters - added by enhance_dg_detection.py
\1def optimize_dg_detection(detector):
\1    # Optimize DG fish detection
\1    detector.dg_params = {
\1        "center_x_ratio": 0.5,    # Center X position relative to width
\1        "center_y_ratio": 0.45,   # Center Y position relative to height
\1        "h_axis_ratio": 0.58,     # Wider horizontal axis for better DG pond coverage
\1        "v_axis_ratio": 0.42,     # Slightly taller vertical axis for better coverage
\1        "shadow_threshold": 0.81,  # Lower threshold to catch more fish shadows
\1        "shadow_min_threshold": 55, # Lower minimum threshold for better detection
\1    }
\1    
\1    # DG-specific Hough parameters - optimized for real usage
\1    if playground == "dg":
\1        detector.hough_dp = 1.1           # Lower DP for better resolution
\1        detector.hough_min_dist = 18      # Lower min distance to catch closely positioned fish
\1        detector.hough_param1 = 55        # Higher edge detection sensitivity
\1        detector.hough_param2 = 14        # Lower threshold to detect more circles
\1        detector.max_fish_count = 4       # DG has exactly 4 fish
\1        detector.max_match_distance = 40  # More flexible matching for real-world tracking
\1    return detector
\1
\1# Check if we're in DG to apply optimizations
\1if "playground" in locals() and playground.lower() == "dg":
\1    fish_detector = optimize_dg_detection(fish_detector)"""
    
    # Apply the modification
    content = re.sub(detector_init_pattern, replacement, content)
    
    # Find the set_playground_params call
    playground_set_pattern = r"(\s*)(fish_detector\.set_playground_params\(playground\))"
    
    # Add a comment after the existing call (we don't replace it)
    replacement = r"\1\2  # Basic playground params, enhanced for DG if detected"
    
    # Apply the modification
    content = re.sub(playground_set_pattern, replacement, content)
    
    return content

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 