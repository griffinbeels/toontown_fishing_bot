#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def test_color_conversion():
    """
    Test the color conversion pipeline to verify colors are preserved correctly.
    Creates a test image with known RGB colors and traces how they're transformed.
    """
    # Create a test image with pure RGB colors
    # Red, Green, Blue, Yellow, Cyan, Magenta, White
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
    
    # Create RGB numpy array (100x700x3) with color bands
    test_image_rgb = np.zeros((100, 70*len(test_colors), 3), dtype=np.uint8)
    for i, color in enumerate(test_colors):
        test_image_rgb[:, i*70:(i+1)*70] = color
    
    print("=== Color Test Results ===")
    print("Original RGB values:")
    for i, name in enumerate(color_names):
        x = i*70 + 35  # Center of color band
        color = test_image_rgb[50, x]
        print(f"{name}: {color} (RGB)")
    
    # 1. Simulate the draw_debug function flow
    debug_image = test_image_rgb.copy()  # This would be the frame passed to draw_debug
    
    # Draw some circles with explicit RGB colors to simulate the debug drawing
    cv2.circle(debug_image, (35, 50), 20, (0, 255, 0), 2)  # Green circle on red
    cv2.circle(debug_image, (105, 50), 20, (255, 255, 0), 2)  # Yellow circle on green
    
    # Print color values at test points
    print("\nAfter debug drawing (should still be RGB):")
    for i, name in enumerate(color_names):
        x = i*70 + 35  # Center of color band
        color = debug_image[50, x]
        print(f"{name}: {color} (RGB)")
    
    # 2. Test conversion to PIL Image and back (as in display_frame)
    pil_img = Image.fromarray(debug_image)
    
    # 3. Test conversion back to numpy array (to check what PIL is doing)
    back_to_array = np.array(pil_img)
    
    print("\nAfter PIL conversion and back to numpy:")
    for i, name in enumerate(color_names):
        x = i*70 + 35  # Center of color band
        color = back_to_array[50, x]
        print(f"{name}: {color} (RGB)")
    
    # 4. Visual comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.title("Original RGB Test Image")
    plt.imshow(test_image_rgb)
    
    plt.subplot(212)
    plt.title("After debug drawing and PIL conversion")
    plt.imshow(back_to_array)
    
    plt.tight_layout()
    plt.savefig("color_test_result.png")
    print("\nVisual comparison saved to color_test_result.png")
    print("Check the image file to see if colors are correct or inverted (RGB vs BGR issue)")

if __name__ == "__main__":
    test_color_conversion() 