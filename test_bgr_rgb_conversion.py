#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Import from our files
sys.path.insert(0, '.')
from debug_dg_detection import SimpleFishDetector, DGTuner
from fish_detection import FishDetector

def test_bgr_rgb_conversion():
    """
    Test to identify where RGB and BGR conversions are happening incorrectly.
    Creates a test image with pure RGB colors and traces them through various conversion paths.
    """
    # Create a test image with pure RGB colors
    test_colors = [
        (255, 0, 0),    # Red in RGB / Blue in BGR
        (0, 255, 0),    # Green in both
        (0, 0, 255),    # Blue in RGB / Red in BGR
        (255, 255, 0),  # Yellow in RGB / Cyan in BGR
        (0, 255, 255),  # Cyan in RGB / Yellow in BGR
        (255, 0, 255),  # Magenta in both
        (255, 255, 255) # White in both
    ]
    
    color_names_rgb = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
    color_names_bgr = ["Blue", "Green", "Red", "Cyan", "Yellow", "Magenta", "White"]
    
    # Create test image with RGB ordering
    test_image_rgb = np.zeros((100, 70*len(test_colors), 3), dtype=np.uint8)
    for i, color in enumerate(test_colors):
        test_image_rgb[:, i*70:(i+1)*70] = color
    
    # Same image but with BGR ordering
    test_image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
    
    print("=== RGB/BGR Conversion Test ===")
    
    # Compare values in RGB and BGR images at same positions
    print("\nRGB image values:")
    for i, name in enumerate(color_names_rgb):
        x = i*70 + 35
        color = test_image_rgb[50, x]
        print(f"{name}: {color}")
    
    print("\nBGR image values:")
    for i, name in enumerate(color_names_bgr):
        x = i*70 + 35
        color = test_image_bgr[50, x]
        print(f"{name}: {color}")
    
    # Test SimpleFishDetector with RGB
    print("\n=== Testing with SimpleFishDetector ===")
    detector = SimpleFishDetector()
    detector.set_playground("dg")
    
    # Test with clearly RGB input
    print("\nTesting with RGB input...")
    debug_frame_from_rgb = detector.draw_debug(test_image_rgb)
    print("After draw_debug:")
    for i, name in enumerate(color_names_rgb):
        x = i*70 + 35
        color = debug_frame_from_rgb[50, x]
        print(f"{color_names_rgb[i]}: {color}")
    
    # Test with BGR input
    print("\nTesting with BGR input...")
    debug_frame_from_bgr = detector.draw_debug(test_image_bgr)
    print("After draw_debug:")
    for i, name in enumerate(color_names_bgr):
        x = i*70 + 35
        color = debug_frame_from_bgr[50, x]
        print(f"{color_names_bgr[i]}: {color}")
    
    # Test in the full DGTuner display_frame method
    print("\n=== Testing display_frame method ===")
    
    # Create a test function based on display_frame method
    def test_display_frame(frame, expects_rgb=True):
        """Simulate the display_frame method processing"""
        try:
            # Convert to PIL
            if expects_rgb:
                print("Input assumed to be RGB")
                pil_img = Image.fromarray(frame)
            else:
                print("Input assumed to be BGR, converting to RGB first")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
            
            print(f"PIL image mode: {pil_img.mode}")
            
            # Convert back for testing
            array_after_pil = np.array(pil_img)
            return array_after_pil
        except Exception as e:
            print(f"Error in display_frame simulation: {e}")
            return None
    
    # Test both paths
    print("\nTesting display_frame with RGB input...")
    display_result_rgb = test_display_frame(debug_frame_from_rgb, expects_rgb=True)
    if display_result_rgb is not None:
        for i, name in enumerate(color_names_rgb):
            x = i*70 + 35
            color = display_result_rgb[50, x]
            print(f"{name}: {color}")
    
    print("\nTesting display_frame with BGR input...")
    display_result_bgr = test_display_frame(debug_frame_from_bgr, expects_rgb=False)
    if display_result_bgr is not None:
        for i, name in enumerate(color_names_bgr):
            x = i*70 + 35
            color = display_result_bgr[50, x]
            print(f"{name}: {color}")
    
    # Create visual comparisons
    plt.figure(figsize=(16, 12))
    
    plt.subplot(331)
    plt.title("Original RGB")
    plt.imshow(test_image_rgb)
    
    plt.subplot(332)
    plt.title("Original BGR (displayed as RGB)")
    plt.imshow(test_image_bgr)
    
    plt.subplot(333)
    plt.title("BGR→RGB Conversion")
    plt.imshow(cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2RGB))
    
    plt.subplot(334)
    plt.title("draw_debug with RGB input")
    plt.imshow(debug_frame_from_rgb)
    
    plt.subplot(335)
    plt.title("draw_debug with BGR input")
    plt.imshow(debug_frame_from_bgr)
    
    plt.subplot(336)
    plt.title("draw_debug with BGR input, converted to RGB")
    plt.imshow(cv2.cvtColor(debug_frame_from_bgr, cv2.COLOR_BGR2RGB))
    
    plt.subplot(337)
    plt.title("display_frame with RGB input")
    plt.imshow(display_result_rgb)
    
    plt.subplot(338)
    plt.title("display_frame with BGR input")
    plt.imshow(display_result_bgr)
    
    plt.subplot(339)
    plt.title("Corrected Flow")
    # Convert BGR to RGB, call detection, convert back to RGB for display
    corrected_path = cv2.cvtColor(detector.draw_debug(cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)
    plt.imshow(corrected_path)
    
    plt.tight_layout()
    plt.savefig("bgr_rgb_test.png")
    print("\nVisual comparison saved to bgr_rgb_test.png")
    print("Check the image file to understand where RGB/BGR mismatches are occurring")

if __name__ == "__main__":
    test_bgr_rgb_conversion() 