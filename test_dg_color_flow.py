#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt
import sys
import os

# Import from our files
sys.path.insert(0, '.')
from debug_dg_detection import SimpleFishDetector

def test_color_flow_in_dg_detection():
    """
    Test the color transformations specifically in the SimpleFishDetector and 
    display_frame methods to identify where color problems occur.
    """
    # Create test image with known RGB colors
    test_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 255) # White
    ]
    
    color_names = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
    
    # Create RGB test image
    test_image_rgb = np.zeros((100, 70*len(test_colors), 3), dtype=np.uint8)
    for i, color in enumerate(test_colors):
        test_image_rgb[:, i*70:(i+1)*70] = color
    
    print("=== SimpleFishDetector Color Flow Test ===")
    print("Original RGB test image created")
    
    print("\nOriginal RGB values:")
    for i, name in enumerate(color_names):
        x = i*70 + 35  # Center of color band
        color = test_image_rgb[50, x]
        print(f"{name}: {color}")
    
    # 1. Check what happens in SimpleFishDetector.draw_debug
    detector = SimpleFishDetector()
    detector.set_playground("dg")
    
    # 2. Pass to draw_debug
    # This assumes frame is RGB as documented
    print("\nPassing RGB image to draw_debug...")
    debug_frame = detector.draw_debug(test_image_rgb)
    
    # 3. Check colors in debug_frame
    print("\nColors after draw_debug (should still be RGB):")
    for i, name in enumerate(color_names):
        x = i*70 + 35  # Center of color band
        color = debug_frame[50, x]
        print(f"{name}: {color}")
    
    # 4. Display using display_frame-like code
    print("\nSimulating display_frame conversion process...")
    
    # Convert to PIL (as in display_frame)
    try:
        pil_img = Image.fromarray(debug_frame)
        print("Successfully converted to PIL Image")
        
        # Check what PIL thinks about the image
        print(f"PIL image mode: {pil_img.mode}")
        
        # Convert back for testing
        back_to_array = np.array(pil_img)
        
        print("\nColors after PIL conversion:")
        for i, name in enumerate(color_names):
            x = i*70 + 35
            color = back_to_array[50, x]
            print(f"{name}: {color}")
    except Exception as e:
        print(f"Error in PIL conversion: {e}")
    
    # 5. Create validation visualizations
    plt.figure(figsize=(12, 12))
    
    plt.subplot(311)
    plt.title("Original RGB Test Image")
    plt.imshow(test_image_rgb)
    
    plt.subplot(312)
    plt.title("After SimpleFishDetector.draw_debug")
    plt.imshow(debug_frame)
    
    plt.subplot(313)
    plt.title("After PIL conversion (display_frame)")
    plt.imshow(back_to_array)
    
    plt.tight_layout()
    plt.savefig("dg_color_test.png")
    print("\nVisual comparison saved to dg_color_test.png")
    print("Check the image to see if colors match or are inverted (RGB vs BGR issue)")

if __name__ == "__main__":
    test_color_flow_in_dg_detection() 