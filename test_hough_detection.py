import cv2
import numpy as np
import os
import time
from pathlib import Path
from fish_detection import FishDetector

def test_detection_comparison():
    """
    Compare traditional contour detection with the new Hough Circle Transform method
    on test images from different playgrounds.
    """
    # Initialize detector
    detector = FishDetector()
    
    # Define test directories
    test_dir = Path("tests")
    playgrounds = ["ttc", "dg", "bb"]
    
    # Create output directory for comparison results
    output_dir = test_dir / "hough_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing fish detection with Hough Circle Transform...")
    
    for playground in playgrounds:
        playground_dir = test_dir / playground
        
        if not playground_dir.exists():
            print(f"Playground directory not found: {playground_dir}")
            continue
        
        # Create playground output directory
        playground_output_dir = output_dir / playground
        playground_output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in playground_dir.glob("test_*.png"):
            test_case = img_path.stem
            print(f"Processing {playground}/{test_case}")
            
            # Load test image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run detection with contours only
            detector.use_hough_transform = False
            detector.set_playground_params(playground)
            contour_fish = detector.detect_fish(img, playground, f"{test_case}_contour")
            contour_debug = detector.draw_debug(img)
            
            # Run detection with Hough + contours
            detector.use_hough_transform = True
            detector.set_playground_params(playground)
            hough_fish = detector.detect_fish(img, playground, f"{test_case}_hough")
            hough_debug = detector.draw_debug(img)
            
            # Create side-by-side comparison
            h, w = img.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w, :] = contour_debug
            comparison[:, w:, :] = hough_debug
            
            # Add labels
            cv2.putText(
                comparison, 
                "Contour Only", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                comparison, 
                "Hough + Contour", 
                (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Add fish counts
            cv2.putText(
                comparison, 
                f"Fish: {len(contour_fish)}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                comparison, 
                f"Fish: {len(hough_fish)}", 
                (w + 10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Save comparison image
            comparison_path = playground_output_dir / f"{test_case}_comparison.png"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            # Save individual debug images to inspect each step
            for method, name in [("contour", "Contour Only"), ("hough", "Hough + Contour")]:
                # Get debug images from debug history
                debug_dir = detector.debug_history.get_debug_dir(playground, f"{test_case}_{method}")
                
                # Copy summary to output directory
                summary_file = debug_dir / "detection_summary.txt"
                if summary_file.exists():
                    with open(summary_file, "r") as src:
                        summary_content = src.read()
                        
                    with open(playground_output_dir / f"{test_case}_{method}_summary.txt", "w") as dst:
                        dst.write(f"Method: {name}\n\n")
                        dst.write(summary_content)
    
    print("Testing completed! Results saved to tests/hough_comparison directory.")

if __name__ == "__main__":
    test_detection_comparison() 