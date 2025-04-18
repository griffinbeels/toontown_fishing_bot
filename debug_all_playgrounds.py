import cv2
import numpy as np
from pathlib import Path
from fish_detection import FishDetector
import time
import os

def debug_playground(detector, playground, test_images_dir="tests"):
    """Run detailed debug on a specific playground."""
    print(f"\n{'='*50}")
    print(f"Testing {playground.upper()} playground")
    print(f"{'='*50}")
    
    # Set playground-specific parameters
    detector.set_playground_params(playground)
    
    # Path to test images
    test_dir = Path(test_images_dir)
    playground_dir = test_dir / playground
    
    if not playground_dir.exists():
        print(f"Error: Test directory for {playground} not found at {playground_dir}")
        return False
        
    # Process each test image in the playground directory
    test_files = list(playground_dir.glob("*.png"))
    if not test_files:
        print(f"No test images found in {playground_dir}")
        return False
        
    for img_path in test_files:
        test_case = img_path.stem  # Use filename as test case ID
        
        print(f"\nProcessing {test_case}...")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Could not load test image at {img_path}")
            continue
            
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run fish detection with detailed debugging
        fish = detector.detect_fish(img_rgb, playground, test_case)
        
        # Print detection results
        print(f"Detected {len(fish)} fish:")
        high_conf_fish = [f for f in fish if f.confidence > 0.7]
        
        for i, f in enumerate(high_conf_fish):
            print(f"  Fish {i+1}: position=({f.x}, {f.y}), radius={f.radius:.1f}, confidence={f.confidence:.2f}")
        
        print(f"High confidence fish: {len(high_conf_fish)}")
        
        # Show image path where debug output is stored
        debug_dir = detector.debug_history.get_debug_dir(playground, test_case)
        print(f"Debug output saved to: {debug_dir}")
    
    return True

def main():
    """Run fish detection debug on all playgrounds."""
    # Initialize detector
    detector = FishDetector()
    
    # Define playgrounds to test
    playgrounds = ["ttc", "dg", "bb"]
    
    # Process each playground
    for playground in playgrounds:
        debug_playground(detector, playground)
    
    print("\nDebug completed. Check the debug_history directory for results.")

if __name__ == "__main__":
    main() 