import cv2
import numpy as np
import os
from pathlib import Path
from fish_detection import FishDetector

def test_ttc_fish_detection():
    """Test fish detection on TTC image with proper visualization of steps."""
    # Setup detector
    detector = FishDetector()
    
    # Path to test image
    test_dir = Path("tests")
    img_path = test_dir / "ttc" / "test_1.png"
    
    # Ensure debug output directory exists
    debug_dir = test_dir / "debug_output"
    debug_dir.mkdir(exist_ok=True)
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load test image at {img_path}")
        return False
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Save original input
    cv2.imwrite(str(debug_dir / "1_original_input.png"), img)
    
    # Run fish detection
    fish = detector.detect_fish(img_rgb)
    
    # Print debug information
    print("\n--- Debug Information ---")
    
    if "pond_brightness" in detector.debug_info:
        print(f"Pond Brightness: {detector.debug_info['pond_brightness']:.1f}")
    
    if "shadow_threshold_value" in detector.debug_info:
        print(f"Shadow Threshold: {detector.debug_info['shadow_threshold_value']}")
    
    if "pond_center" in detector.debug_info:
        print(f"Pond Center: {detector.debug_info['pond_center']}")
    
    if "pond_axes" in detector.debug_info:
        print(f"Pond Axes: {detector.debug_info['pond_axes']}")
    
    # Check if contour metrics were recorded
    if "contour_metrics" in detector.debug_info:
        print(f"\nFound {len(detector.debug_info['contour_metrics'])} contours during processing")
        for i, metrics in enumerate(detector.debug_info["contour_metrics"]):
            print(f"\nContour #{i+1}:")
            print(f"  Position: {metrics['position']}")
            print(f"  Radius: {metrics['radius']:.1f}")
            print(f"  Area: {metrics['area']:.1f}")
            print(f"  Circularity: {metrics['circularity']:.2f} (threshold: {detector.circularity_threshold:.2f})")
            print(f"  Solidity: {metrics['solidity']:.2f} (threshold: {detector.solidity_threshold:.2f})")
            print(f"  Aspect Ratio: {metrics['aspect_ratio']:.2f} (threshold: {detector.min_aspect_ratio:.2f})")
            print(f"  Center Score: {metrics['center_score']:.2f} (threshold: 0.3)")
            print(f"  Confidence: {metrics['confidence']:.2f}")
            print(f"  Is Valid: {metrics['is_valid']}")
    else:
        print("\nNo contours found in the shadow mask!")
    
    # Generate incremental debug images
    debug_images = detector.draw_incremental_debug(img_rgb)
    
    # Save all incremental debug images
    for img_name, debug_img in debug_images.items():
        # Convert RGB to BGR for saving
        if len(debug_img.shape) == 3 and debug_img.shape[2] == 3:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(debug_dir / f"{img_name}.png"), debug_img)
    
    # Print info about detected fish
    print("\n--- Detection Results ---")
    print(f"Detected {len(fish)} fish:")
    for i, f in enumerate(fish):
        print(f"  Fish {i+1}: position=({f.x}, {f.y}), radius={f.radius:.1f}, confidence={f.confidence:.2f}")
    
    # Count high-confidence detections (>0.7)
    high_conf_detections = len([f for f in fish if f.confidence > 0.7])
    
    # Check if we detected exactly two fish with high confidence
    expected_count = 2
    result = high_conf_detections == expected_count
    
    print(f"Expected {expected_count} fish, found {high_conf_detections} high-confidence detections")
    print(f"Test result: {'PASS' if result else 'FAIL'}")
    
    return result

if __name__ == "__main__":
    test_ttc_fish_detection() 