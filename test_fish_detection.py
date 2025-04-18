import unittest
import cv2
import numpy as np
import os
from pathlib import Path
from fish_detection import FishDetector

class TestFishDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FishDetector()
        self.test_dir = Path(__file__).parent
        
        # Expected fish counts for each test image
        self.expected_counts = {
            'dg': 4,  # Donald's Dock should have 4 fish
            'ttc': 2  # Toontown Central should have 2 fish
        }
        
        # Load test images
        self.test_images = {}
        for location in ['dg', 'ttc']:
            img_dir = self.test_dir / location
            if img_dir.exists():
                for img_path in img_dir.glob('test_1.png'):  # Use test_1.png for consistency
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.test_images[location] = img
    
    def test_pond_detection(self):
        """Test that pond area is correctly detected in test images."""
        for location, image in self.test_images.items():
            pond_mask = self.detector.detect_pond_area(image)
            
            # Pond mask should not be empty
            self.assertTrue(np.any(pond_mask > 0), f"No pond detected in {location} image")
            
            # Pond should cover reasonable portion of image (5-50%)
            pond_coverage = np.sum(pond_mask > 0) / (pond_mask.shape[0] * pond_mask.shape[1])
            self.assertTrue(0.05 <= pond_coverage <= 0.5,
                          f"Pond coverage in {location} is {pond_coverage:.2%}, expected 5-50%")
    
    def test_fish_detection_count(self):
        """Test that correct number of fish are detected in each image."""
        for location, image in self.test_images.items():
            if location not in self.expected_counts:
                continue
                
            fish = self.detector.detect_fish(image)
            expected = self.expected_counts[location]
            
            # Count high-confidence detections (confidence > 0.7)
            high_conf_detections = len([f for f in fish if f.confidence > 0.7])
            
            self.assertEqual(high_conf_detections, expected,
                           f"Expected {expected} fish in {location}, found {high_conf_detections} "
                           f"high-confidence detections")
    
    def test_fish_characteristics(self):
        """Test that detected fish have reasonable characteristics."""
        for location, image in self.test_images.items():
            fish = self.detector.detect_fish(image)
            
            for f in fish:
                # Test position is within image bounds
                self.assertTrue(0 <= f.x < image.shape[1], f"Fish x position {f.x} out of bounds")
                self.assertTrue(0 <= f.y < image.shape[0], f"Fish y position {f.y} out of bounds")
                
                # Test radius is within expected range
                self.assertTrue(self.detector.min_radius <= f.radius <= self.detector.max_radius,
                              f"Fish radius {f.radius} outside expected range")
                
                # Test area is within expected range
                self.assertTrue(self.detector.min_area <= f.area <= self.detector.max_area,
                              f"Fish area {f.area} outside expected range")
    
    def test_detection_stability(self):
        """Test that detection is stable across multiple runs."""
        for location, image in self.test_images.items():
            # Run detection multiple times
            results = []
            for _ in range(3):  # Reduced to 3 runs for faster testing
                fish = self.detector.detect_fish(image)
                results.append(len([f for f in fish if f.confidence > 0.7]))
            
            # All runs should give same number of detections
            self.assertEqual(len(set(results)), 1,
                           f"Detection not stable in {location}. Got varying counts: {results}")
    
    def test_visualization(self):
        """Test that debug visualization runs without errors."""
        for location, image in self.test_images.items():
            try:
                # First detect fish
                self.detector.detect_fish(image)
                # Then try to draw debug visualization
                debug_frame = self.detector.draw_debug(image)
                
                # Check debug frame has same dimensions as input
                self.assertEqual(image.shape, debug_frame.shape,
                               "Debug visualization changed image dimensions")
                
                # Save debug output for manual inspection
                debug_dir = self.test_dir / "debug_output"
                debug_dir.mkdir(exist_ok=True)
                debug_img = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"{location}_debug.png"), debug_img)
                
            except Exception as e:
                self.fail(f"Visualization failed for {location}: {str(e)}")
    
    def test_incremental_debug(self):
        """Test that incremental debug visualization works properly."""
        for location, image in self.test_images.items():
            try:
                # First detect fish
                fish = self.detector.detect_fish(image)
                
                # Generate and check incremental debug images
                debug_images = self.detector.draw_incremental_debug(image)
                
                # Should have at least 5 debug images
                self.assertGreaterEqual(len(debug_images), 5, 
                                     f"Too few incremental debug images for {location}")
                
                # All images should have same dimensions as input or be valid binary masks
                for name, debug_img in debug_images.items():
                    self.assertEqual(len(debug_img.shape), 3, 
                                   f"Debug image {name} doesn't have 3 dimensions")
                    self.assertEqual(debug_img.shape[0], image.shape[0], 
                                   f"Debug image {name} has wrong height")
                    self.assertEqual(debug_img.shape[1], image.shape[1], 
                                   f"Debug image {name} has wrong width")
                
            except Exception as e:
                self.fail(f"Incremental debug failed for {location}: {str(e)}")

if __name__ == '__main__':
    unittest.main() 