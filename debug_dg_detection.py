#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import argparse
import tkinter as tk
from tkinter import ttk, messagebox # Import messagebox
from pathlib import Path
import json # Import json
import sys
from fish_detection import FishDetector
from collections import deque
from PIL import Image, ImageTk
from enum import Enum, auto
# Import the new ScreenCapturer class and exceptions
from screen_capture import ScreenCapturer, WindowNotFoundError, CaptureError

# Define visualization modes
class VisualizationMode(Enum):
    # Add START mode
    START = auto()        # Raw captured frame
    # Input preprocessing
    GRAYSCALE = auto()    # Grayscale conversion (first processing step)
    
    # Region masking
    ROI_MASK = auto()     # Region of interest mask (pond area)
    NAME_TAG_MASK = auto()  # Name tag masking
    
    # Image processing
    BACKGROUND = auto()   # Background subtraction
    POND_TREATMENT = auto()  # Pond water treatment adjustments
    THRESHOLD = auto()    # Thresholded image after preprocessing
    
    # Contour detection
    CONTOURS = auto()     # All detected contours
    VALID_CONTOURS = auto()  # Filtered valid contours
    
    # Final detection
    CIRCLES = auto()      # Hough circles detection
    FINAL_MASK = auto()   # Final processed mask
    FINAL = auto()        # Final output: Original frame + detection overlay
    
    # Composite view
    ALL_STAGES = auto()   # Grid of all processing stages

# --- Mappings --- 
# Define Parameter Configuration Structure (Moved BEFORE mappings)
PARAMETER_CONFIG = [
    # --- Region Masking ---
    {
        'category': 'Region Masking',
        'step': 'ROI Mask', # Corresponds to VisualizationMode.ROI_MASK
        'params': [
            # Currently ROI mask dimensions are playground-specific, but center weight affects confidence
            {'name': 'center_pond_weight', 'label': "Center Weight", 'min': 0.0, 'max': 1.0, 'default': 0.2, 'precision': 2},
            # Re-add ellipse axis ratios for direct tuning
            {'name': 'h_axis_ratio', 'label': "H Axis Ratio", 'min': 0.01, 'max': 1.0, 'default': 0.58, 'precision': 2},
            {'name': 'v_axis_ratio', 'label': "V Axis Ratio", 'min': 0.01, 'max': 1.0, 'default': 0.42, 'precision': 2},
        ]
    },
    {
        'category': 'Region Masking',
        'step': 'Name Tag Mask', # Corresponds to VisualizationMode.NAME_TAG_MASK
        'params': [
            {'name': 'name_tag_x_offset', 'label': "X Offset", 'min': -1000, 'max': 1000, 'default': 0, 'precision': 0},
            {'name': 'name_tag_y_offset', 'label': "Y Offset", 'min': -1000, 'max': 100, 'default': -195, 'precision': 0},
            {'name': 'name_tag_width', 'label': "Width", 'min': 10, 'max': 1000, 'default': 114, 'precision': 0},
            {'name': 'name_tag_height', 'label': "Height", 'min': 5, 'max': 1000, 'default': 79, 'precision': 0},
        ]
    },
    # --- Image Processing ---
    {
        'category': 'Image Processing',
        'step': 'Background', # Corresponds to VisualizationMode.BACKGROUND
        'params': [
            # No specific parameters here for now, relies on background model toggle
        ]
    },
    {
        'category': 'Image Processing',
        'step': 'Pond Treatment', # Corresponds to VisualizationMode.POND_TREATMENT
        'params': [
            {'name': 'pond_brightness_adjust', 'label': "Brightness Adjust", 'min': -100, 'max': 100, 'default': 50, 'precision': 0},
            {'name': 'pond_contrast_adjust', 'label': "Contrast Adjust", 'min': 0.1, 'max': 3.0, 'default': 2.0, 'precision': 2},
        ]
    },
    {
        'category': 'Image Processing',
        'step': 'Threshold', # Corresponds to VisualizationMode.THRESHOLD
        'params': [
             {'name': 'shadow_threshold', 'label': "Shadow Ratio", 'min': 0.1, 'max': 1.0, 'default': 0.8, 'precision': 2 },
             {'name': 'shadow_min_threshold', 'label': "Min Shadow Value", 'min': 0, 'max': 255, 'default': 55, 'precision': 0},
             {'name': 'adaptive_block_size', 'label': "Adaptive Block Size", 'min': 3, 'max': 500, 'default': 101, 'precision': 0},
             {'name': 'adaptive_c_value', 'label': "Adaptive C Value", 'min': -20, 'max': 50, 'default': 20, 'precision': 0},
        ]
    },
    # --- Contour Detection --- (Validation params applied here)
    {
        'category': 'Contour Detection',
        'step': 'Validation', # Related to CONTOURS/VALID_CONTOURS view
        'params': [
             # These are part of FishDetector's core validation logic
             {'name': 'min_area', 'label': 'Min Area', 'min': 0, 'max': 600, 'default': 10, 'precision': 0}, # Updated max range
             {'name': 'max_area', 'label': 'Max Area', 'min': 50, 'max': 1900, 'default': 2500, 'precision': 0}, # Updated max range
             {'name': 'circularity_threshold', 'label': 'Min Circularity', 'min': 0.0, 'max': 1.0, 'default': 0.3, 'precision': 2}, # Increased default to 0.3
             {'name': 'solidity_threshold', 'label': 'Min Solidity', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'precision': 2}, # Increased default to 0.5
             {'name': 'min_aspect_ratio', 'label': 'Min Aspect Ratio', 'min': 0.0, 'max': 1.0, 'default': 0.3, 'precision': 2}, # Keep 0.3
        ]
    },
    # --- Final Detection (Hough) ---
     {
        'category': 'Final Detection (Hough)',
        'step': 'Circles', # Corresponds to VisualizationMode.CIRCLES
        'params': [
            # NEW Hough Preprocessing Params
            {'name': 'hough_blur_ksize', 'label': "Blur Kernel Size", 'min': 1, 'max': 21, 'default': 5, 'precision': 0}, # Keep 5
            {'name': 'hough_adapt_block', 'label': "Adapt Thresh Block", 'min': 3, 'max': 401, 'default': 21, 'precision': 0}, # Keep 21
            {'name': 'hough_adapt_c', 'label': "Adapt Thresh C", 'min': -20, 'max': 20, 'default': 5, 'precision': 0}, # Keep 5
            {'name': 'hough_morph_ksize', 'label': "Morph Kernel Size", 'min': 1, 'max': 15, 'default': 5, 'precision': 0}, # Increased default to 5
            # MOVED Min/Max Radius Here
            {'name': 'min_radius', 'label': 'Min Radius', 'min': 0.1, 'max': 200, 'default': 1.0, 'precision': 1}, 
            {'name': 'max_radius', 'label': 'Max Radius', 'min': 1, 'max': 400, 'default': 50.0, 'precision': 1}, 
            # Existing Hough Circle Params
            {'name': 'hough_dp', 'label': "DP (Resolution)", 'min': 1.0, 'max': 3.0, 'default': 1.2, 'precision': 1}, # Keep 1.2
            {'name': 'hough_min_dist', 'label': "Min Distance", 'min': 1, 'max': 200, 'default': 25, 'precision': 0}, # Increased default to 25
            {'name': 'hough_param1', 'label': "Canny Edge Thresh", 'min': 1, 'max': 300, 'default': 50, 'precision': 0}, # Keep 50
            {'name': 'hough_param2', 'label': "Accumulator Thresh", 'min': 1, 'max': 100, 'default': 45, 'precision': 0}, # Increased default to 45
        ]
    },
    # FINAL Mode has no specific parameters
]

# Link Visualization Modes to their corresponding Parameter section titles
# Use the 'step' name from PARAMETER_CONFIG and append ' Parameters'
VIS_MODE_TO_PARAM_SECTION = {
    "ROI_MASK": "ROI Mask Parameters",
    "NAME_TAG_MASK": "Name Tag Mask Parameters",
    "BACKGROUND": None, # No specific params currently
    "POND_TREATMENT": "Pond Treatment Parameters",
    "THRESHOLD": "Threshold Parameters",
    "CONTOURS": "Validation Parameters", # Validation params affect contour filtering
    "VALID_CONTOURS": "Validation Parameters",
    "CIRCLES": "Circles Parameters",
    "FINAL_MASK": None, # No specific params currently
    "FINAL": None, # No specific params currently
    "START": None,
    "ALL_STAGES": None,
}
# Map step section titles back to their parent category section titles
PARAM_STEP_TO_CATEGORY = {item['step'] + " Parameters": item['category'] 
                          for item in PARAMETER_CONFIG if item.get('params')}
# Add top-level category titles for expansion
PARAM_STEP_TO_CATEGORY.update({item['category']: "Parameters" 
                                for item in PARAMETER_CONFIG})
PARAM_STEP_TO_CATEGORY["Parameters"] = None # Top level

# --- End Mappings ---

# Define the default file path for saving/loading parameters
DEFAULT_PARAMS_FILE = "saved_params.json"

class ParameterSlider:
    """A slider with a label to adjust a parameter."""
    def __init__(self, parent, name, label, min_val, max_val, initial, precision=2, callback=None):
        self.name = name
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=5, pady=2)
        
        self.label = ttk.Label(self.frame, text=label, width=20)
        self.label.pack(side="left")
        
        self.value_var = tk.DoubleVar(value=initial)
        self.precision = precision
        
        self.slider = ttk.Scale(
            self.frame, 
            from_=min_val, 
            to=max_val, 
            variable=self.value_var,
            orient="horizontal"
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # Create a label to show the current value
        self.value_label = ttk.Label(self.frame, width=10)
        self.value_label.pack(side="right")
        
        # Set up callback and update value display
        self.callback = callback
        self.value_var.trace_add("write", self._on_value_changed)
        self._update_value_label()
    
    def _on_value_changed(self, *args):
        self._update_value_label()
        if self.callback:
            self.callback(self.name, self.get_value())
    
    def _update_value_label(self):
        # Format value with appropriate precision
        format_str = f"{{:.{self.precision}f}}"
        self.value_label.config(text=format_str.format(self.value_var.get()))
    
    def get_value(self):
        return self.value_var.get()
    
    def set_value(self, value):
        self.value_var.set(value)

class SimpleFishDetector:
    """Simplified wrapper for FishDetector to improve performance."""
    def __init__(self):
        self.detector = FishDetector()
        self.params = {}
        self.use_background_model = False
        self.mask_name_tag = True  # Whether to mask out the player's name tag
        
        # Pond water treatment parameters
        self.pond_brightness_adjust = 0  # Adjustment to pond brightness (-50 to 50)
        self.pond_contrast_adjust = 1.0  # Contrast multiplier for pond (0.5 to 2.0)
        self.adaptive_block_size = 21  # Block size for adaptive thresholding (odd number)
        self.adaptive_c_value = 5      # C value for adaptive thresholding
        
        # Store debug visuals for inspection
        self.debug_images = {}
        # Position and size for name tag masking
        self.name_tag_params = {
            "x_offset": 0,     # Horizontal offset from player center 
            "y_offset": -195,   # Vertical offset from player center (negative = above)
            "width": 114,       # Width of name tag mask
            "height": 79       # Height of name tag mask
        }
    
    def set_playground(self, playground):
        """Set the playground and update detector parameters."""
        # Apply the playground parameters from FishDetector directly
        self.detector.set_playground_params(playground)
        
        # Don't modify the parameters further - just use FishDetector's defaults
        # This prevents double application of parameters
        
        # Store the playground for debug info
        self.current_playground = playground
    
    def set_parameters(self, params):
        """Set specific detection parameters manually.
        
        This allows overriding the default playground parameters.
        """
        self.params.update(params)
        
        # Apply parameters directly to the detector
        # Apply common parameters
        if 'center_pond_weight' in params:
            self.detector.center_pond_weight = params['center_pond_weight']
            
        # Apply Hough parameters
        if 'hough_dp' in params:
            self.detector.hough_dp = params['hough_dp']
        if 'hough_min_dist' in params:
            self.detector.hough_min_dist = params['hough_min_dist']
        if 'hough_param1' in params:
            self.detector.hough_param1 = params['hough_param1']
        if 'hough_param2' in params:
            self.detector.hough_param2 = params['hough_param2']
            
        # Apply ellipse parameters - directly to the detector, not to playground params
        # This prevents double application of parameters
        if 'h_axis_ratio' in params:
            self.detector.h_axis_ratio = params['h_axis_ratio']
        if 'v_axis_ratio' in params:
            self.detector.v_axis_ratio = params['v_axis_ratio']
            
        # Apply shadow parameters - directly to the detector, not to playground params
        if 'shadow_threshold' in params:
            self.detector.shadow_threshold = params['shadow_threshold']
        if 'shadow_min_threshold' in params:
            self.detector.shadow_min_threshold = int(params['shadow_min_threshold'])
            
        # Apply size constraints
        if 'min_radius' in params:
            self.detector.min_radius = params['min_radius']
        if 'max_radius' in params:
            self.detector.max_radius = params['max_radius']
        if 'min_area' in params:
            self.detector.min_area = params['min_area']
        if 'max_area' in params:
            self.detector.max_area = params['max_area']
            
        # Apply pond water treatment parameters
        if 'pond_brightness_adjust' in params:
            self.pond_brightness_adjust = int(params['pond_brightness_adjust'])
        if 'pond_contrast_adjust' in params:
            self.pond_contrast_adjust = float(params['pond_contrast_adjust'])
        if 'adaptive_block_size' in params:
            # Ensure block size is odd
            block_size = int(params['adaptive_block_size'])
            if block_size % 2 == 0:
                block_size += 1
            self.adaptive_block_size = block_size
        if 'adaptive_c_value' in params:
            self.adaptive_c_value = int(params['adaptive_c_value'])
            
        # Apply name tag masking parameters if present
        if 'name_tag_x_offset' in params:
            self.name_tag_params["x_offset"] = params['name_tag_x_offset']
        if 'name_tag_y_offset' in params:
            self.name_tag_params["y_offset"] = params['name_tag_y_offset']
        if 'name_tag_width' in params:
            self.name_tag_params["width"] = params['name_tag_width']
        if 'name_tag_height' in params:
            self.name_tag_params["height"] = params['name_tag_height']
    
    def create_name_tag_mask(self, frame_shape):
        """Create a mask to exclude the player's name tag.
        
        Args:
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Binary mask where 0 = name tag area to exclude
        """
        height, width = frame_shape[:2]
        
        # Create base mask (all ones - everything included)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Player position is typically at the bottom center
        player_center_x = int(width * 0.5)
        player_center_y = int(height * 0.8)
        
        # Calculate name tag position
        tag_x = int(player_center_x + self.name_tag_params["x_offset"])
        tag_y = int(player_center_y + self.name_tag_params["y_offset"])
        tag_width = int(self.name_tag_params["width"])
        tag_height = int(self.name_tag_params["height"])
        
        # Create rectangle for name tag
        x1 = max(0, int(tag_x - tag_width // 2))
        y1 = max(0, int(tag_y - tag_height // 2))
        x2 = min(width, int(tag_x + tag_width // 2))
        y2 = min(height, int(tag_y + tag_height // 2))
        
        # Set name tag area to 0 (excluded area)
        mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def detect_fish(self, frame):
        """Detect fish in frame with current parameters."""
        # Manually disable background model to prevent freezing
        original_background_frames = self.detector.background_frames
        original_median_background = self.detector.median_background
        
        # Disable background model if not using it
        if not self.use_background_model:
            self.detector.background_frames = deque(maxlen=10)
            self.detector.median_background = None
            self.detector.use_hough_transform = True
        
        # Disable morphology if needed - REMOVED, now controlled by flag
        # original_morphology = self.detector.apply_morphology if hasattr(self.detector, 'apply_morphology') else True
        # self.detector.apply_morphology = self.apply_morphology
        
        try:
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            # Prepare the additional mask to pass to the detector
            additional_mask = None
            
            # If name tag masking is enabled, create and prepare the mask
            if self.mask_name_tag:
                # Create name tag mask (0 = exclude)
                name_tag_mask = self.create_name_tag_mask(frame.shape)
                additional_mask = name_tag_mask # Assign it to be passed
                
                # Store for debugging - create a visualization of the mask over the frame
                mask_viz = frame.copy()
                name_tag_inv = cv2.bitwise_not(name_tag_mask)
                mask_viz[name_tag_inv > 0] = [0, 0, 255] # Red in areas to be masked
                self.debug_images["name_tag_mask"] = name_tag_mask
                self.debug_images["name_tag_overlay"] = mask_viz
                # No need to store combined_mask here anymore
            
            # Apply pond water treatment - modify how we process the image
            # This step does not affect the mask directly but prepares the image for thresholding/hough
            processed_frame = self.apply_pond_water_treatment(processed_frame)
            
            # Call the detector with the processed frame AND the additional mask
            fish = self.detector.detect_fish(
                processed_frame, 
                playground=self.current_playground, 
                additional_exclude_mask=additional_mask # Pass the mask here
            )
            
            # Get contours for visualization (Now uses the correct mask source)
            self._extract_contours_for_debug(processed_frame) 
            
            # REMOVED: Post-detection filtering based on name tag mask is no longer needed
            # as the mask is applied *during* detection by FishDetector
            # if name_tag_mask is not None: ...
            
            # Save debug images for different visualization modes
            self.debug_images.update({
                "grayscale": self.detector.debug_info.get("grayscale", None),
                "global_threshold": self.detector.debug_info.get("global_threshold", None),
                "adaptive_threshold": self.detector.debug_info.get("adaptive_threshold", None),
                "combined_threshold": self.detector.debug_info.get("combined_threshold", None),
                "final_mask": self.detector.debug_info.get("final_shadow_mask", None),
                "roi_mask": self.detector.debug_info.get("roi_mask", None),
                "background_subtraction": self.detector.debug_info.get("background_subtraction", None),
                "hough_circles": self.detector.debug_info.get("hough_circles", None)
            })
            
            return fish
        except Exception as e:
            print(f"Error in fish detection: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Restore original values
            if not self.use_background_model:
                self.detector.background_frames = original_background_frames
                self.detector.median_background = original_median_background
            # self.detector.apply_morphology = original_morphology # REMOVED
            
            # Restore original character mask if we modified it
            # This logic is incorrect as the wrapper doesn't directly modify the core detector's mask
            # if self.mask_name_tag and hasattr(self.detector, 'character_mask'):
            #     self.detector.character_mask = original_character_mask
    
    def _extract_contours_for_debug(self, frame):
        """Extract contours from the FINAL SHADOW MASK for debugging purposes."""
        # Get the actual final shadow mask generated by detect_pond_area
        # This mask IS affected by the shadow parameters
        shadow_mask = self.detector.debug_info.get("final_shadow_mask")
        
        if shadow_mask is None:
            print("Warning: final_shadow_mask not found in debug_info for contour extraction.")
            # Fallback: Create a blank mask of the same size
            if frame is not None:
                 shadow_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            else: # Cannot proceed without a frame size reference
                 self.all_contours = []
                 self.valid_contours = []
                 return []
        
        # Ensure mask is binary (it should be, but good practice)
        _, binary_mask = cv2.threshold(shadow_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the final shadow mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store all contours
        self.all_contours = contours
        
        # Store binary mask for potential display if needed (though FINAL_MASK mode shows it)
        self.processed_binary = binary_mask # Reuse this variable name
        
        # Filter contours based on shape and size (Using FishDetector's validation)
        valid_contours = []
        # Directly use self.detector which is the FishDetector instance
        if hasattr(self.detector, 'is_valid_fish'): # Check if the method exists on the detector instance
            for contour in contours:
                area = cv2.contourArea(contour)
                # Basic area check first (using self.detector attributes directly)
                if area < self.detector.min_area or area > self.detector.max_area:
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                # Check if validation passes (ignoring confidence score here, just shape/size)
                # Call the method directly on self.detector
                is_valid, _ = self.detector.is_valid_fish(contour, area, int(x), int(y), radius, frame.shape)
                if is_valid:
                    valid_contours.append(contour)
        else:
             # This case should ideally not happen if self.detector is initialized correctly
             print("Warning: Cannot perform full contour validation - is_valid_fish method not found on detector.")
             valid_contours = contours # Fallback: use all contours if validation fails

        # Store valid contours
        self.valid_contours = valid_contours
        
        return valid_contours # Though not strictly necessary to return here
    
    def apply_pond_water_treatment(self, frame):
        """Apply pond water treatment to the frame based on current parameters.
        
        This modifies how we process the pond water to better detect fish shadows.
        """
        # Make a copy of the frame to avoid modifying the original
        processed = frame.copy()
        
        # Get ROI mask for the pond area
        roi_mask = self.detector.create_roi_mask(frame.shape)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Apply brightness and contrast adjustments to the pond area
        if self.pond_brightness_adjust != 0 or self.pond_contrast_adjust != 1.0:
            # Create mask for pond area
            pond_mask = np.zeros_like(gray)
            pond_mask[roi_mask > 0] = 255
            
            # Extract pond area
            pond_area = cv2.bitwise_and(gray, gray, mask=pond_mask)
            
            # Apply contrast and brightness adjustments
            alpha = self.pond_contrast_adjust  # Contrast control
            beta = self.pond_brightness_adjust  # Brightness control
            adjusted_pond = cv2.convertScaleAbs(pond_area, alpha=alpha, beta=beta)
            
            # Put the adjusted pond back into the frame
            # First create inverted mask
            inv_mask = cv2.bitwise_not(pond_mask)
            # Get everything outside pond
            outside_pond = cv2.bitwise_and(gray, gray, mask=inv_mask)
            # Combine outside and adjusted pond
            gray = cv2.add(outside_pond, adjusted_pond)
            
            # Save for debugging
            self.debug_images["adjusted_pond"] = gray.copy()
        
        # Apply adaptive thresholding with custom parameters
        if self.adaptive_block_size > 3:  # Minimum valid block size
            # Ensure block size is an integer and odd
            block_size_int = int(self.adaptive_block_size)
            if block_size_int % 2 == 0:
                block_size_int += 1 
            # Ensure C value is an integer
            c_value_int = int(self.adaptive_c_value)
            
            # Apply adaptive threshold to grayscale image (mask to pond area)
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size_int, # Pass integer block size
                c_value_int     # Pass integer C value
            )
            # Mask to pond area
            thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
            # Save for debugging
            self.debug_images["custom_threshold"] = thresh.copy()
            
            # Replace the shadows in the processed frame
            # First convert processed back to grayscale if it isn't already
            if len(processed.shape) == 3:
                processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                processed_gray = processed.copy()
                
            # Create a mask of shadows from thresholding
            shadow_mask = thresh.copy()
            
            # Modify the pond area's shadows in the processed frame
            processed_gray[shadow_mask > 0] = 0  # Make shadows pure black
            
            # Convert back to color if original was color
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
            else:
                processed = processed_gray
        
        # Return the processed frame
        return processed
    
    def draw_debug(self, frame, visualization_mode=VisualizationMode.START):
        """Draw debug visualization based on selected mode."""
        try:
            # Explicitly handle START mode first
            if visualization_mode == VisualizationMode.START:
                return self._draw_start_debug(frame)
            elif visualization_mode == VisualizationMode.GRAYSCALE:
                return self._draw_grayscale_debug(frame)
            elif visualization_mode == VisualizationMode.THRESHOLD:
                return self._draw_threshold_debug(frame)
            elif visualization_mode == VisualizationMode.CONTOURS:
                return self._draw_contours_debug(frame)
            elif visualization_mode == VisualizationMode.VALID_CONTOURS:
                return self._draw_valid_contours_debug(frame)
            elif visualization_mode == VisualizationMode.CIRCLES:
                return self._draw_circles_debug(frame)
            elif visualization_mode == VisualizationMode.ROI_MASK:
                return self._draw_roi_mask_debug(frame)
            elif visualization_mode == VisualizationMode.FINAL_MASK:
                return self._draw_final_mask_debug(frame)
            elif visualization_mode == VisualizationMode.BACKGROUND:
                return self._draw_background_debug(frame)
            elif visualization_mode == VisualizationMode.POND_TREATMENT:
                return self._draw_pond_water_debug(frame)
            elif visualization_mode == VisualizationMode.NAME_TAG_MASK:
                return self._draw_name_tag_mask_debug(frame)
            elif visualization_mode == VisualizationMode.ALL_STAGES:
                return self._draw_all_stages_debug(frame)
            elif visualization_mode == VisualizationMode.FINAL:
                return self._draw_final_debug(frame) # Add final mode handler
            else:
                # Default to START mode if unknown
                print(f"Warning: Unknown visualization mode '{visualization_mode}'. Defaulting to START.")
                return self._draw_start_debug(frame) 
        except Exception as e:
            print(f"Error drawing debug: {e}")
            # Return original frame if drawing fails to prevent crash
            return frame.copy()

    def _draw_start_debug(self, frame):
        """Return the original frame without modifications."""
        return frame.copy() # Keep only one return
    
    def _draw_grayscale_debug(self, frame):
        """Draw grayscale visualization."""
        if "grayscale" in self.debug_images and self.debug_images["grayscale"] is not None:
            gray = self.debug_images["grayscale"]
            # Convert to color for viewing
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame.copy()
    
    def _draw_threshold_debug(self, frame):
        """Draw thresholded visualization."""
        if "combined_threshold" in self.debug_images and self.debug_images["combined_threshold"] is not None:
            thresh = self.debug_images["combined_threshold"]
            # Convert to color for viewing
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return frame.copy()
    
    def _draw_contours_debug(self, frame):
        """Draw all detected contours on a black background."""
        if hasattr(self, 'all_contours') and self.all_contours:
            # Create black background
            result = np.zeros_like(frame)
            # Draw contours in green
            cv2.drawContours(result, self.all_contours, -1, (0, 255, 0), 1) 
            return result
        # Return black frame if no contours
        return np.zeros_like(frame)
    
    def _draw_valid_contours_debug(self, frame):
        """Draw valid contours on a black background."""
        if hasattr(self, 'valid_contours') and self.valid_contours:
            # Create black background
            result = np.zeros_like(frame)
            # Draw valid contours in red
            cv2.drawContours(result, self.valid_contours, -1, (0, 0, 255), 1) 
            return result
        # Return black frame if no valid contours
        return np.zeros_like(frame)
    
    def _draw_circles_debug(self, frame):
        """Draw Hough circles visualization."""
        if "hough_circles" in self.debug_images and self.debug_images["hough_circles"] is not None:
            circles = self.debug_images["hough_circles"]
            # Return the circles image
            return circles
        return frame.copy()
    
    def _draw_roi_mask_debug(self, frame):
        """Draw ROI mask visualization."""
        if "roi_mask" in self.debug_images and self.debug_images["roi_mask"] is not None:
            roi = self.debug_images["roi_mask"]
            # Convert to color for viewing
            return cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        return frame.copy()
    
    def _draw_final_mask_debug(self, frame):
        """Draw final mask visualization."""
        if "final_mask" in self.debug_images and self.debug_images["final_mask"] is not None:
            final = self.debug_images["final_mask"]
            # Convert to color for viewing
            return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        return frame.copy()
    
    def _draw_background_debug(self, frame):
        """Draw background subtraction visualization."""
        if "background_subtraction" in self.debug_images and self.debug_images["background_subtraction"] is not None:
            bg = self.debug_images["background_subtraction"]
            # Convert to color for viewing
            return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        return frame.copy()
    
    def _draw_pond_water_debug(self, frame):
        """Draw pond water treatment visualization."""
        # Check if we have adjusted pond or custom threshold
        if "adjusted_pond" in self.debug_images and self.debug_images["adjusted_pond"] is not None:
            # Use the adjusted pond image
            adjusted = self.debug_images["adjusted_pond"]
            # Convert to color for viewing
            if len(adjusted.shape) == 2:
                return cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
            return adjusted
        elif "custom_threshold" in self.debug_images and self.debug_images["custom_threshold"] is not None:
            # Use the custom threshold image
            threshold = self.debug_images["custom_threshold"]
            # Convert to color for viewing
            if len(threshold.shape) == 2:
                return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            return threshold
        
        # If no pond water treatment images, return grayscale
        return self._draw_grayscale_debug(frame)
    
    def _draw_name_tag_mask_debug(self, frame):
        """Draw name tag mask visualization."""
        if "name_tag_overlay" in self.debug_images and self.debug_images["name_tag_overlay"] is not None:
            # Use the pre-created overlay with colored mask areas
            return self.debug_images["name_tag_overlay"].copy()
        elif "name_tag_mask" in self.debug_images and self.debug_images["name_tag_mask"] is not None:
            name_tag_mask = self.debug_images["name_tag_mask"]
            # Create a better visualization
            viz = frame.copy()
            # Invert the mask (0=name tag area, 255=normal area)
            inverted = cv2.bitwise_not(name_tag_mask)
            # Create red overlay for masked areas
            red_areas = np.zeros_like(viz)
            red_areas[:, :, 2] = 255  # Red channel
            # Apply the inverted mask to the red overlay
            red_masked = cv2.bitwise_and(red_areas, red_areas, mask=inverted)
            # Blend with original frame
            result = cv2.addWeighted(viz, 0.7, red_masked, 0.5, 0)
            # Draw contour around the masked area
            contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
            return result
        return frame.copy()
    
    def _draw_all_stages_debug(self, frame):
        """Create a grid visualization of all processing stages."""
        # Create a blank canvas
        height, width = frame.shape[:2]
        grid_height = height * 3  # 3 rows
        grid_width = width * 3    # 3 columns
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Define the order for the grid
        grid_layout = [
            # Row 1
            (self._draw_start_debug, "Original (Start)"),
            (self._draw_grayscale_debug, "Grayscale"),
            (self._draw_roi_mask_debug, "ROI Mask"),
            # Row 2
            (self._draw_pond_water_debug, "Pond Treatment"),
            (self._draw_final_mask_debug, "Final Mask (Morph)"),
            (self._draw_circles_debug, "Circles"),
            # Row 3
            (self._draw_contours_debug, "All Contours"),
            (self._draw_valid_contours_debug, "Valid Contours"),
            (self._draw_final_debug, "Final Output")
        ]

        # Collect images for the grid
        images = []
        labels = []
        for draw_func, label in grid_layout:
            try:
                img = draw_func(frame) 
                # Ensure image is BGR and correct size
                if img is None:
                    img = np.zeros((height, width, 3), dtype=np.uint8) # Black fallback
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Final check for 3 channels
                if len(img.shape) != 3 or img.shape[2] != 3:
                    print(f"Warning: Image for '{label}' is not 3-channel after processing. Creating black fallback.")
                    img = np.zeros((height, width, 3), dtype=np.uint8) # Black fallback
                    
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error generating image for '{label}' in grid: {e}")
                images.append(np.zeros((height, width, 3), dtype=np.uint8)) # Black fallback on error
                labels.append(label + " (Error)")

        # Place images in grid
        idx = 0
        for r in range(3):
            for c in range(3):
                if idx < len(images):
                    grid[r*height:(r+1)*height, c*width:(c+1)*width] = images[idx]
                    # Add labels
                    cv2.putText(grid, labels[idx], (c*width + 10, r*height + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1) # Smaller font
                    idx += 1

        return grid

    def _draw_final_debug(self, frame):
        """Draw final detection overlay (fish circles/confidence) on original frame."""
        debug_frame = frame.copy()
        # Use the currently tracked fish from the detector
        fish_list = self.detector.tracked_fish if hasattr(self.detector, 'tracked_fish') else []

        # Draw circles with brighter colors
        for fish in fish_list:
            # Draw a bright green circle
            cv2.circle(debug_frame, (fish.x, fish.y), int(fish.radius),
                      (0, 255, 0), 2)  # BGR green

            # Draw confidence text with better visibility
            text = f"{fish.confidence:.2f}"
            cv2.putText(debug_frame, text, (fish.x - 20, fish.y - int(fish.radius) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # BGR yellow

        # Add bright annotation showing total fish count
        cv2.putText(debug_frame, f"Fish: {len(fish_list)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # BGR yellow

        return debug_frame

    def _draw_normal_debug(self, frame):
        """Draw standard debug visualization."""
        # This mode is removed, call _draw_final_debug instead for similar output
        return self._draw_final_debug(frame)

class FishDetectionTuner(tk.Tk):
    """Application for tuning fish detection parameters for any playground."""
    def __init__(self):
        super().__init__()
        self.title("Fish Detection Tuner")
        self.geometry("1000x700")
        
        # Available playgrounds
        self.playgrounds = ["dg", "ttc", "bb"]
        self.current_playground = "dg"  # Default playground
        
        # Create detector (Renamed)
        self.simple_detector = SimpleFishDetector()
        self.simple_detector.set_playground(self.current_playground)
        
        # Configure for performance
        self.enable_visualization = True
        self.frame_interval = 50  # Slower frame rate for better stability (~20 FPS)
        self.is_running = False
        self.processing = False  # Flag to prevent concurrent processing
        
        # Set current visualization mode
        self.current_visualization_mode = VisualizationMode.START
                
        # --- NEW: Track parameter changes ---
        self.params_changed = False
        # --- End NEW ---
        
        # Set up the main interface
        self.setup_ui() 
        
        # Connect to game window using ScreenCapturer
        try:
            self.capturer = ScreenCapturer("Corporate Clash")
            initial_dims = self.capturer.get_capture_dimensions()
            game_width = initial_dims[2]
            game_height = initial_dims[3]
            
            # Calculate initial window size
            control_panel_width = 420 # UPDATED: Match actual control panel width + padding/scrollbar
            info_area_height = 120 # Estimated height for info text + status bar + padding
            total_width = game_width + control_panel_width
            total_height = max(game_height, 400) + info_area_height # Ensure minimum height + space for controls/info
            self.geometry(f"{total_width}x{int(total_height)}") # Set initial size, ensure height is int

            # Update display canvas size if needed
            if hasattr(self, 'display_canvas'):
                self.display_canvas.config(width=game_width, height=game_height)
                
            self.is_running = True
            self.status_var.set("Connected to game window")
        except (WindowNotFoundError, CaptureError, Exception) as e:
            self.status_var.set(f"Error connecting to game: {e}")
            self.is_running = False # Ensure loop doesn't start
        
        # Register exit handler and start capture loop if running
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.is_running:
            self.capture_loop()
    
    def setup_ui(self):
        """Set up the user interface with parameter controls."""
        # ... (Keep existing setup for main_frame, control_frame_outer, canvas, scrollbar, display_frame, display_canvas, status_bar, info_text) ...
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        self.control_frame_outer = ttk.Frame(self.main_frame, width=400, padding=5) # Increased width
        self.control_frame_outer.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        self.control_canvas = tk.Canvas(self.control_frame_outer, width=400) # Increased width
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.control_frame_outer, orient=tk.VERTICAL, command=self.control_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.control_frame = ttk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor=tk.NW)
        
        # Right panel for visualization
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.display_canvas = tk.Canvas(self.display_frame, bg="black")
        self.display_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.info_text_frame = ttk.Frame(self)
        self.info_text_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)
        ttk.Label(self.info_text_frame, text="Detection Info:").pack(side=tk.TOP, anchor=tk.W)
        self.info_text = tk.Text(self.info_text_frame, height=8, wrap=tk.WORD) 
        info_scrollbar = ttk.Scrollbar(self.info_text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.info_text.bind("<MouseWheel>", self._on_text_mousewheel)

        # --- Top Level Controls --- 
        self.sections = {} # Initialize sections dict
        self.sliders = {} # Initialize sliders dict
        
        # Playground section
        playground_content_frame = self.create_section(self.control_frame, "Playground", expanded=True, level=0)
        ttk.Label(playground_content_frame, text="Playground").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.playground_var = tk.StringVar(value="dg")
        self.playground_dropdown = ttk.Combobox(playground_content_frame, textvariable=self.playground_var, 
                                               values=["ttc", "dg", "bb"], state="readonly")
        self.playground_dropdown.grid(row=0, column=1, sticky=tk.EW, pady=2)
        self.playground_dropdown.bind("<<ComboboxSelected>>", self.change_playground)
        
        # Performance Options section
        perf_content_frame = self.create_section(self.control_frame, "Performance Options", expanded=True, level=0)
        self.visualization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_content_frame, text="Enable Visualization", variable=self.visualization_var, 
                      command=self.toggle_visualization).grid(row=0, column=0, sticky=tk.W, pady=2, columnspan=2)
        self.background_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(perf_content_frame, text="Use Background Model", variable=self.background_var, 
                      command=self.toggle_background).grid(row=1, column=0, sticky=tk.W, pady=2, columnspan=2)
        self.morphology_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_content_frame, text="Apply Morphology", variable=self.morphology_var, 
                      command=self.toggle_morphology).grid(row=2, column=0, sticky=tk.W, pady=2, columnspan=2)
        self.name_tag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_content_frame, text="Mask Name Tag", variable=self.name_tag_var, 
                      command=self.toggle_name_tag_mask).grid(row=3, column=0, sticky=tk.W, pady=2, columnspan=2)
        
        # Visualization Mode section
        # Start Visualization Mode expanded
        vis_mode_content_frame = self.create_section(self.control_frame, "Visualization Mode", expanded=True, level=0)
        self.visualization_mode_var = tk.StringVar(value="START") 
        # Create radio buttons for visualization modes (adjust rows as needed)
        vis_modes = [
            ("Input Preprocessing", [
                ("START", "Start"), ("GRAYSCALE", "Grayscale")
            ]),
            ("Region Masking", [
                ("ROI_MASK", "ROI Mask"), ("NAME_TAG_MASK", "Name Tag Mask")
            ]),
            ("Image Processing", [
                ("BACKGROUND", "Background"), ("POND_TREATMENT", "Pond Treatment"), ("THRESHOLD", "Threshold")
            ]),
            ("Contour Detection", [
                ("CONTOURS", "Contours"), ("VALID_CONTOURS", "Valid Contours")
            ]),
            ("Final Detection", [
                ("CIRCLES", "Circles"), ("FINAL_MASK", "Final Mask"), ("FINAL", "Final Output")
            ]),
            ("Composite View", [
                ("ALL_STAGES", "All Stages")
            ])
        ]
        row_idx = 0
        for cat_label, modes in vis_modes:
            ttk.Label(vis_mode_content_frame, text=cat_label, font=("TkDefaultFont", 9, "bold")).grid(
                row=row_idx, column=0, sticky=tk.W, pady=(5,2), columnspan=2)
            row_idx += 1
            for mode_val, mode_label in modes:
                ttk.Radiobutton(vis_mode_content_frame, text=mode_label, variable=self.visualization_mode_var,
                               value=mode_val, command=self.change_visualization_mode).grid(
                               row=row_idx, column=0, sticky=tk.W, pady=1, padx=10)
                row_idx += 1
        
        # --- New Parameters Section --- 
        # Start Parameters section expanded
        parameters_main_frame = self.create_section(self.control_frame, "Parameters", expanded=True, level=0)
        
        # Group parameters by category and then step
        grouped_params = {}
        for item in PARAMETER_CONFIG:
            cat = item['category']
            step = item['step']
            if cat not in grouped_params:
                grouped_params[cat] = {}
            if step not in grouped_params[cat]:
                grouped_params[cat][step] = []
            grouped_params[cat][step].extend(item['params'])

        # Create nested sections for parameters
        for category, steps in grouped_params.items():
            # Create Category Section (Level 1)
            category_content_frame = self.create_section(parameters_main_frame, category, expanded=False, level=1)
            
            for step, params in steps.items():
                if not params: continue # Skip steps with no parameters
                # Create Step Section (Level 2)
                step_title = f"{step} Parameters"
                step_content_frame = self.create_section(category_content_frame, step_title, expanded=False, level=2)
                
                # Create sliders within the step section
                for param_config in params:
                    param_name = param_config['name']
                    # Get initial value from detector if possible, else use config default
                    initial_value = param_config['default'] # Default fallback
                    # Access attributes via self.simple_detector or self.simple_detector.detector
                    if hasattr(self.simple_detector, param_name):
                        initial_value = getattr(self.simple_detector, param_name)
                    elif hasattr(self.simple_detector.detector, param_name):
                         initial_value = getattr(self.simple_detector.detector, param_name)
                    elif param_name.startswith("name_tag_") and hasattr(self.simple_detector, 'name_tag_params'):
                         # Correctly access name_tag_params on simple_detector
                         initial_value = self.simple_detector.name_tag_params.get(param_name.replace("name_tag_",""), initial_value)
                    # Add check for new Hough preprocessing params on the core detector
                    elif param_name in ['hough_blur_ksize', 'hough_adapt_block', 'hough_adapt_c', 'hough_morph_ksize'] and hasattr(self.simple_detector.detector, param_name):
                         initial_value = getattr(self.simple_detector.detector, param_name)
                         
                    slider = ParameterSlider(
                        step_content_frame,
                        param_name,
                        param_config['label'],
                        param_config['min'],
                        param_config['max'],
                        initial_value, # Use potentially updated initial value
                        precision=param_config['precision'],
                        callback=self.update_parameter # Use the existing callback
                    )
                    self.sliders[param_name] = slider # Store slider instance

        # REMOVE OLD Parameter Section Creation Calls
        # self.create_section("Hough Circle Detection", expanded=False) 
        # self.create_section("Size Constraints", expanded=False)
        # self.create_section("Pond Detection", expanded=False)
        # self.create_section("Shadow Detection", expanded=False)
        # self.create_section("Pond Water Treatment", expanded=False)
        # self.create_section("Name Tag Masking", expanded=False)
        # self.create_parameter_sliders() # REMOVE THIS CALL

        # Reset and Save Buttons (Pack at the bottom of the control_frame)
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM) # Pack at bottom
        
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.RIGHT, padx=5)
        
        # Configure scrolling for the left panel canvas
        self.control_frame.bind("<Configure>", self._on_frame_configure)
        self.control_canvas.bind("<Configure>", self._on_canvas_configure)
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.control_frame.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Set initial slider values based on current playground defaults (or loaded params)
        # self.update_sliders_from_detector() # This will be called within load_parameters or after fallback

        # Load parameters after UI is set up but before starting loop
        self.load_parameters()
    
    def create_section(self, parent_frame, title, expanded=False, level=0):
        """Create a collapsible section with header.
        Args:
            parent_frame: The parent Tkinter frame.
            title: The title for the section header.
            expanded: Whether the section starts expanded.
            level: Indentation level (0 for top-level).
        Returns:
            The content frame for this section.
        """
        # Create section frame with padding based on level
        section_frame = ttk.Frame(parent_frame)
        section_frame.pack(fill=tk.X, pady=(2, 0), padx=(level * 15, 0))

        # Style for header based on level (optional)
        style_name = f"Level{level}.TFrame" if level > 0 else "Section.TFrame"
        # Ensure style exists (basic implementation)
        s = ttk.Style()
        s.configure(style_name, background="lightgrey" if level == 0 else "SystemButtonFace") 

        # Create header frame
        header_frame = ttk.Frame(section_frame, style=style_name)
        header_frame.pack(fill=tk.X)

        # Store section state
        if title not in self.sections:
            self.sections[title] = {
                "expanded": expanded,
                "icon_var": tk.StringVar(value="▼" if expanded else "►"),
                "content_frame": None # Will be assigned below
            }
        section_state = self.sections[title]
        section_state["icon_var"].set("▼" if section_state["expanded"] else "►") # Ensure icon matches state

        # Create header with toggle icon
        toggle_label = ttk.Label(header_frame, textvariable=section_state["icon_var"], width=2)
        toggle_label.pack(side=tk.LEFT, padx=(5, 0))

        # Section title (adjust font size for sub-levels)
        font_weight = "bold" if level == 0 else "normal"
        font_size = 10 if level == 0 else 9
        title_label = ttk.Label(header_frame, text=title, font=("TkDefaultFont", font_size, font_weight))
        title_label.pack(side=tk.LEFT, padx=5, pady=(1 if level == 0 else 0))

        # Content frame
        content_frame = ttk.Frame(section_frame, padding=(15, 2, 0, 2)) # Consistent padding for content
        section_state["content_frame"] = content_frame # Store reference
        if section_state["expanded"]:
            content_frame.pack(fill=tk.X)
        
        # Make the entire header clickable to toggle
        # Important: Pass unique identifier (title) to lambda
        toggle_func = lambda e, t=title: self.toggle_section(t)
        for widget in [toggle_label, title_label, header_frame]:
            widget.bind("<Button-1>", toggle_func)

        return content_frame

    def toggle_section(self, title, force_expand=False):
        """Toggle the visibility of a section, optionally force expand and expand parents."""
        if title not in self.sections:
            print(f"Warning: Attempted to toggle non-existent section '{title}'")
            return

        section = self.sections[title]
        original_state = section["expanded"]
        
        # Determine new state
        if force_expand:
            new_state = True
        else:
            new_state = not section["expanded"]
            
        section["expanded"] = new_state
        section["icon_var"].set("▼" if section["expanded"] else "►")

        # Show/hide content frame
        content_frame = section["content_frame"]
        if content_frame:
            if section["expanded"]:
                content_frame.pack(fill=tk.X)
            else:
                content_frame.pack_forget()
        
        # Update scrollregion after UI changes settle
        self.control_frame.update_idletasks()
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def change_playground(self, event=None):
        """Change the active playground."""
        playground = self.playground_var.get()
        if playground in self.playgrounds:
            # Update current playground
            self.current_playground = playground
            
            # Set the playground in the detector (use renamed variable)
            self.simple_detector.set_playground(playground)
            
            # Update slider values to match playground defaults
            self.update_sliders_from_detector()
            
            # Update status
            self.status_var.set(f"Switched to {playground.upper()} playground")
    
    def update_sliders_from_detector(self):
        """Update slider values based on current detector parameters."""
        # Get current parameters from the core detector instance
        core_detector = self.simple_detector.detector 
        
        # Update Hough parameters
        if hasattr(core_detector, 'hough_dp') and 'hough_dp' in self.sliders:
            self.sliders['hough_dp'].set_value(core_detector.hough_dp)
        if hasattr(core_detector, 'hough_min_dist') and 'hough_min_dist' in self.sliders:
            self.sliders['hough_min_dist'].set_value(core_detector.hough_min_dist)
        if hasattr(core_detector, 'hough_param1') and 'hough_param1' in self.sliders:
            self.sliders['hough_param1'].set_value(core_detector.hough_param1)
        if hasattr(core_detector, 'hough_param2') and 'hough_param2' in self.sliders:
            self.sliders['hough_param2'].set_value(core_detector.hough_param2)
            
        # Update size constraints
        if hasattr(core_detector, 'min_radius') and 'min_radius' in self.sliders:
            self.sliders['min_radius'].set_value(core_detector.min_radius)
        if hasattr(core_detector, 'max_radius') and 'max_radius' in self.sliders:
            self.sliders['max_radius'].set_value(core_detector.max_radius)
            
        # Update ellipse parameters for pond
        if hasattr(core_detector, 'h_axis_ratio') and 'h_axis_ratio' in self.sliders:
            self.sliders['h_axis_ratio'].set_value(core_detector.h_axis_ratio)
        if hasattr(core_detector, 'v_axis_ratio') and 'v_axis_ratio' in self.sliders:
            self.sliders['v_axis_ratio'].set_value(core_detector.v_axis_ratio)
            
        # Update center weight
        if hasattr(core_detector, 'center_pond_weight') and 'center_pond_weight' in self.sliders:
            self.sliders['center_pond_weight'].set_value(core_detector.center_pond_weight)
            
        # Update shadow detection parameters
        if hasattr(core_detector, 'shadow_threshold') and 'shadow_threshold' in self.sliders:
            self.sliders['shadow_threshold'].set_value(core_detector.shadow_threshold)
        if hasattr(core_detector, 'shadow_min_threshold') and 'shadow_min_threshold' in self.sliders:
            self.sliders['shadow_min_threshold'].set_value(core_detector.shadow_min_threshold)
        
        # Update validation params (if sliders exist)
        if hasattr(core_detector, 'min_area') and 'min_area' in self.sliders:
            self.sliders['min_area'].set_value(core_detector.min_area)
        if hasattr(core_detector, 'max_area') and 'max_area' in self.sliders:
            self.sliders['max_area'].set_value(core_detector.max_area)
        if hasattr(core_detector, 'circularity_threshold') and 'circularity_threshold' in self.sliders:
            self.sliders['circularity_threshold'].set_value(core_detector.circularity_threshold)
        if hasattr(core_detector, 'solidity_threshold') and 'solidity_threshold' in self.sliders:
            self.sliders['solidity_threshold'].set_value(core_detector.solidity_threshold)
        if hasattr(core_detector, 'min_aspect_ratio') and 'min_aspect_ratio' in self.sliders:
            self.sliders['min_aspect_ratio'].set_value(core_detector.min_aspect_ratio)

        # Update SimpleFishDetector specific params from self.simple_detector
        if hasattr(self.simple_detector, 'name_tag_params'):
             if 'name_tag_x_offset' in self.sliders: self.sliders['name_tag_x_offset'].set_value(self.simple_detector.name_tag_params.get('x_offset', 0))
             if 'name_tag_y_offset' in self.sliders: self.sliders['name_tag_y_offset'].set_value(self.simple_detector.name_tag_params.get('y_offset', -195))
             if 'name_tag_width' in self.sliders: self.sliders['name_tag_width'].set_value(self.simple_detector.name_tag_params.get('width', 114))
             if 'name_tag_height' in self.sliders: self.sliders['name_tag_height'].set_value(self.simple_detector.name_tag_params.get('height', 79))
        if hasattr(self.simple_detector, 'pond_brightness_adjust') and 'pond_brightness_adjust' in self.sliders:
            self.sliders['pond_brightness_adjust'].set_value(self.simple_detector.pond_brightness_adjust)
        if hasattr(self.simple_detector, 'pond_contrast_adjust') and 'pond_contrast_adjust' in self.sliders:
            self.sliders['pond_contrast_adjust'].set_value(self.simple_detector.pond_contrast_adjust)
        if hasattr(self.simple_detector, 'adaptive_block_size') and 'adaptive_block_size' in self.sliders:
             self.sliders['adaptive_block_size'].set_value(self.simple_detector.adaptive_block_size)
        if hasattr(self.simple_detector, 'adaptive_c_value') and 'adaptive_c_value' in self.sliders:
             self.sliders['adaptive_c_value'].set_value(self.simple_detector.adaptive_c_value)

        # Update NEW Hough preprocessing params (from core_detector)
        if hasattr(core_detector, 'hough_blur_ksize') and 'hough_blur_ksize' in self.sliders:
             self.sliders['hough_blur_ksize'].set_value(core_detector.hough_blur_ksize)
        if hasattr(core_detector, 'hough_adapt_block') and 'hough_adapt_block' in self.sliders:
             self.sliders['hough_adapt_block'].set_value(core_detector.hough_adapt_block)
        if hasattr(core_detector, 'hough_adapt_c') and 'hough_adapt_c' in self.sliders:
             self.sliders['hough_adapt_c'].set_value(core_detector.hough_adapt_c)
        if hasattr(core_detector, 'hough_morph_ksize') and 'hough_morph_ksize' in self.sliders:
             self.sliders['hough_morph_ksize'].set_value(core_detector.hough_morph_ksize)

    def change_visualization_mode(self):
        """Change the visualization mode and expand relevant parameters."""
        mode_str = self.visualization_mode_var.get()
        try:
            self.current_visualization_mode = VisualizationMode[mode_str]
            self.status_var.set(f"Visualization mode changed to {mode_str}")
            
            # Expand corresponding parameter section if mapping exists
            target_section_title = VIS_MODE_TO_PARAM_SECTION.get(mode_str)
            if target_section_title:
                # print(f"Expanding section '{target_section_title}' for mode '{mode_str}'") # Debug print
                self.toggle_section(target_section_title, force_expand=True)
            # else: # Debug print
                # print(f"No parameter section mapped for mode '{mode_str}'")

        except KeyError:
            self.status_var.set(f"Invalid visualization mode: {mode_str}")

    def toggle_name_tag_mask(self):
        """Toggle name tag masking on/off."""
        self.simple_detector.mask_name_tag = self.name_tag_var.get()
    
    def update_parameter(self, param_name, value):
        """Update a detector parameter and re-detect."""
        try:
            self.status_var.set("")
            core_detector = self.simple_detector.detector # Reference to the actual FishDetector
            
            # Special handling for min_radius and max_radius
            if param_name == 'min_radius':
                core_detector.min_radius = value
                # Ensure max_radius is greater than min_radius
                if core_detector.max_radius <= value:
                    core_detector.max_radius = value + 1
                    self.sliders['max_radius'].set_value(value + 1)
            elif param_name == 'max_radius':
                # Ensure max_radius is greater than min_radius
                if value > core_detector.min_radius:
                    core_detector.max_radius = value
                else:
                    value = core_detector.min_radius + 1
                    core_detector.max_radius = value
                    self.sliders['max_radius'].set_value(value)
            else:
                # Set attribute on the correct object:
                # Core FishDetector params go to core_detector (self.simple_detector.detector)
                # Wrapper params (like pond adjustments, name tags) go to self.simple_detector
                core_detector_params = [
                    'center_pond_weight', 'h_axis_ratio', 'v_axis_ratio',
                    'shadow_threshold', 'shadow_min_threshold',
                    'min_area', 'max_area', 'min_radius', 'max_radius',
                    'circularity_threshold', 'solidity_threshold', 'min_aspect_ratio',
                    'hough_dp', 'hough_min_dist', 'hough_param1', 'hough_param2'
                ]
                
                # Add new Hough preprocessing params to core list
                hough_preprocessing_params = ['hough_blur_ksize', 'hough_adapt_block', 'hough_adapt_c', 'hough_morph_ksize']
                core_detector_params.extend(hough_preprocessing_params)
                
                if param_name in core_detector_params:
                    # Ensure target object exists
                    if core_detector: # Check if core_detector is not None
                        # Cast integer parameters correctly
                        if param_name in ['shadow_min_threshold', 'min_area', 'max_area', 
                                         'hough_min_dist', 'hough_param1', 'hough_param2',
                                         'hough_blur_ksize', 'hough_adapt_block', 'hough_adapt_c', 'hough_morph_ksize']: # Add new params here
                             # Ensure odd values for kernel/block sizes
                             int_value = int(value)
                             if param_name in ['hough_blur_ksize', 'hough_adapt_block'] and int_value % 2 == 0:
                                 int_value += 1
                                 # Update slider visually if value changed
                                 if self.sliders[param_name].get_value() != int_value:
                                     self.sliders[param_name].set_value(int_value)
                                     # No need to call setattr again, slider callback handles it
                                     # Return here because the slider callback will trigger this method again
                                     return 
                             setattr(core_detector, param_name, int_value)
                        # Cast float parameters correctly (adjust precision if needed)
                        elif param_name in ['center_pond_weight', 'h_axis_ratio', 'v_axis_ratio', 'shadow_threshold', 
                                          'min_radius', 'max_radius', 'circularity_threshold', 'solidity_threshold', 
                                          'min_aspect_ratio', 'hough_dp']:
                             current_val = getattr(core_detector, param_name, 'Not Found')
                             new_val = float(value)
                             setattr(core_detector, param_name, new_val)
                        else: # Default case (should ideally cover all)
                             setattr(core_detector, param_name, value) 
                    else:
                        print(f"Warning: core_detector (self.simple_detector.detector) not found when setting {param_name}")
                elif param_name.startswith('name_tag_'):
                     # Handle name tag params (stored in a dict on SimpleFishDetector)
                     key = param_name.replace("name_tag_", "")
                     # Ensure the dict exists on self.simple_detector
                     if not hasattr(self.simple_detector, 'name_tag_params'):
                         self.simple_detector.name_tag_params = {}
                     # Set integer value
                     self.simple_detector.name_tag_params[key] = int(value) 
                elif param_name in ['pond_brightness_adjust', 'adaptive_block_size', 'adaptive_c_value']:
                    # Handle integer params on SimpleFishDetector
                    setattr(self.simple_detector, param_name, int(value))
                elif param_name == 'pond_contrast_adjust':
                     # Handle float param on SimpleFishDetector
                     setattr(self.simple_detector, param_name, float(value))
                else:
                    # Fallback for any other parameters (should not happen with PARAMETER_CONFIG)
                    print(f"Warning: Unknown parameter '{param_name}' encountered in update_parameter.")
                    setattr(self.simple_detector, param_name, value) # Set on wrapper by default

            # Update status
            self.status_var.set(f"Updated {param_name} to {value:.{self.sliders[param_name].precision}f}") # Use precision from slider
                
            # REMOVED: Re-detection logic. The main loop will use the updated param naturally.
            # if self.current_image is not None: 
            #     self.detect_on_current_image()
                
        except Exception as e:
            # Print the specific error for debugging
            print(f"Error during parameter update for '{param_name}': {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error updating parameter: {str(e)}")
    
    def _on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame."""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Update the inner frame's width to fill the canvas."""
        width = event.width
        # Update the canvas's scroll region to fit the inner frame
        self.control_canvas.itemconfig(
            self.control_canvas.find_withtag("all")[0] if self.control_canvas.find_withtag("all") else 1, 
            width=width
        )
    
    def _on_mousewheel(self, event):
        """Scroll the canvas or text widget on mousewheel."""
        target_widget = event.widget
        
        # Determine the scroll amount
        if event.delta:  # Windows
            delta = -1 * (event.delta // 120)
        elif hasattr(event, 'num'): # Linux
            delta = -1 if event.num == 4 else 1 if event.num == 5 else 0
        else: # Fallback or other OS
            return "break" # Indicate event handled

        # Check if the event occurred over the control canvas or its children
        widget = target_widget
        is_control_scroll = False
        while widget is not None:
            if widget == self.control_canvas:
                is_control_scroll = True
                break
            # Check if the widget is a direct child of control_frame (which is inside control_canvas)
            if widget.master == self.control_frame:
                 is_control_scroll = True
                 break
            # Go up the widget hierarchy
            try:
                widget = widget.master
            except AttributeError:
                # Reached the top-level window or a widget without a master
                break
                
        if is_control_scroll:
            self.control_canvas.yview_scroll(delta, "units")
            return "break" # Stop the event from propagating

        # If not control scroll, let the text widget handle it if necessary
        # (The text widget's specific binding should already handle this, but 
        # this ensures we don't interfere if the event wasn't over the control canvas area)

        return # Allow event propagation for other widgets
    
    def toggle_visualization(self):
        """Toggle visualization on/off."""
        self.enable_visualization = self.visualization_var.get()
        # Update display based on toggle state
        if not hasattr(self, 'display_canvas'):
            return
            
        if self.enable_visualization:
            self.display_canvas.pack(fill="both", expand=True)
        else:
            self.display_canvas.pack_forget()
    
    def toggle_background(self):
        """Toggle background model on/off."""
        # self.simple_detector.use_background_model = self.background_var.get() # Old way
        # Set the flag on the core detector
        if hasattr(self.simple_detector, 'detector'):
            self.simple_detector.detector.use_background_model_flag = self.background_var.get()
            if not self.background_var.get():
                # Explicitly clear the core detector's background model when disabling
                self.simple_detector.detector.median_background = None
                self.simple_detector.detector.background_frames.clear()
                self.status_var.set("Background model disabled and cleared.")
            else:
                self.status_var.set("Background model enabled (needs frames to compute).")
        else:
             self.status_var.set("Error: Core detector not found to toggle background model.")
    
    def toggle_morphology(self):
        """Toggle morphology operations on/off."""
        # self.simple_detector.apply_morphology = self.morphology_var.get() # Old way
        # Set the flag on the core detector
        if hasattr(self.simple_detector, 'detector'):
            self.simple_detector.detector.apply_morphology = self.morphology_var.get()
            self.status_var.set(f"Morphology {'enabled' if self.morphology_var.get() else 'disabled'}.")
        else:
             self.status_var.set("Error: Core detector not found to toggle morphology.")
             
    def capture_loop(self):
        """Main capture and processing loop using ScreenCapturer."""
        if not self.is_running or self.processing:
            # Check again after delay
            self.after(self.frame_interval, self.capture_loop)
            return

        self.processing = True
        frame = None # Initialize frame
        capture_time = 0.0

        try:
            # Capture frame using the ScreenCapturer instance
            frame_data = self.capturer.capture_frame()
            
            # If capture failed (e.g., window minimized), skip processing this cycle
            if frame_data is None or frame_data[0] is None:
                # Update status slightly differently for skipped frames
                # Keep last known fish count visible if desired
                self.status_var.set(f"[{self.current_playground.upper()}] Capture Skipped | Viz: {self.current_visualization_mode.name}")
                self.processing = False
                self.after(self.frame_interval, self.capture_loop)
                return
                
            frame, capture_time = frame_data

            # Process the successfully captured frame
            if frame:
                self.process_frame(frame) # process_frame now handles display
            else:
                # Handle cases where capture_frame might return None unexpectedly
                 self.status_var.set(f"[{self.current_playground.upper()}] Capture returned None | Viz: {self.current_visualization_mode.name}")

        except WindowNotFoundError as e:
            self.status_var.set(f"ERROR: Game window lost ({e}). Stopping.")
            self.is_running = False # Stop the loop
        except CaptureError as e:
            # Log capture errors but try to continue
            self.status_var.set(f"Capture Error: {e}")
            # Optionally add a small delay here if capture errors are frequent
            # time.sleep(0.1)
        except Exception as e:
            # Catch other unexpected errors during capture/processing call
            self.status_var.set(f"ERROR in loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False # Stop on unexpected errors

        # Schedule next iteration regardless of processing outcome (unless stopped)
        self.processing = False
        if self.is_running:
            self.after(self.frame_interval, self.capture_loop)
    
    def process_frame(self, frame: Image.Image): 
        """Process a captured frame (PIL Image) and update display."""
        # ... (Keep existing logic: convert PIL->BGR, detect, visualize, call display_frame) ...
        try:
            # Convert PIL Image (RGB) to numpy array (BGR for OpenCV)
            frame_np = np.array(frame)
            # Ensure it's RGB before converting
            if frame.mode == 'RGB':
                 frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            elif frame.mode == 'RGBA': # Handle potential alpha channel
                 frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
            elif frame.mode == 'L': # Handle grayscale
                 frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            else:
                 # Assume BGR or other format OpenCV might handle, or log warning
                 print(f"Warning: Unexpected PIL image mode '{frame.mode}'. Assuming BGR-like.")
                 frame_bgr = frame_np # Use as is, hope for the best

        except Exception as e:
            self.status_var.set(f"Image conversion error: {e}")
            self.display_frame(None) # Clear display on conversion error
            return

        fish = []
        process_time = 0
        display_image = frame_bgr.copy() # Default to original frame
        mode_name = "N/A"

        try:
            # Perform fish detection
            start_time = time.time()
            fish = self.simple_detector.detect_fish(frame_bgr)
            process_time = (time.time() - start_time) * 1000

            self.update_fish_info(fish) # Update info text regardless of visualization

            # Generate the visualization frame if enabled
            if self.enable_visualization:
                try:
                    # Generate the debug frame based on the selected mode
                    display_image = self.simple_detector.draw_debug(frame_bgr, self.current_visualization_mode)
                    mode_name = self.current_visualization_mode.name
                except Exception as e:
                    self.status_var.set(f"Visualization error: {e}")
                    display_image = frame_bgr.copy() # Fallback to original frame on viz error
                    mode_name = "ERROR"
            else:
                # If visualization is off, display_image remains the original frame_bgr
                mode_name = "Disabled"
                # Explicitly set display_image to ensure it's BGR
                display_image = frame_bgr.copy() 

            # Update status bar
            self.status_var.set(f"[{self.current_playground.upper()}] Detected {len(fish)} fish | Process: {process_time:.1f}ms | Viz: {mode_name}")

        except Exception as e:
            self.status_var.set(f"Detection error: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            display_image = frame_bgr.copy() # Fallback to original on detection error
            mode_name = "Detection Failed"
            self.status_var.set(f"[{self.current_playground.upper()}] Detection Failed | Process: {process_time:.1f}ms | Viz: {mode_name}")

        # Display the determined frame (original or visualized)
        # Rename method call to avoid potential Tkinter conflicts
        self.update_display_canvas(display_image)
    
    def update_fish_info(self, fish):
        """Update the fish information display."""
        self.info_text.delete(1.0, tk.END)
        
        if not fish:
            self.info_text.insert(tk.END, "No fish detected\n")
            return
        
        self.info_text.insert(tk.END, f"Detected {len(fish)} fish in {self.current_playground.upper()}:\n")
        for i, f in enumerate(fish):
            self.info_text.insert(
                tk.END, 
                f"Fish #{i+1}: position=({f.x}, {f.y}), radius={f.radius:.1f}, confidence={f.confidence:.2f}\n"
            )
    
    def update_display_canvas(self, frame_to_display):
        """Display the provided frame (already processed/visualized) on the canvas."""
        if frame_to_display is not None and isinstance(frame_to_display, np.ndarray):
            try:
                # --- Input Validation --- 
                if frame_to_display.size == 0: 
                    print("Warning: Received empty frame for display.")
                    self.display_canvas.delete("all") # Clear canvas
                    return
                if frame_to_display.dtype != np.uint8:
                    print(f"Warning: Unexpected frame dtype {frame_to_display.dtype}. Attempting conversion.")
                    try:
                        frame_to_display = frame_to_display.astype(np.uint8)
                    except ValueError:
                        print("Error: Could not convert frame to uint8.")
                        self.display_canvas.delete("all")
                        return

                viz_frame = frame_to_display

                # --- Format Conversion (Ensure BGR) ---
                if len(viz_frame.shape) == 2:
                    # Convert grayscale to BGR
                    viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_GRAY2BGR)
                elif len(viz_frame.shape) != 3 or viz_frame.shape[2] != 3:
                     print(f"Warning: Unexpected frame shape {viz_frame.shape} received in display_frame. Cannot display.")
                     self.display_canvas.delete("all")
                     return
                
                # --- BGR to RGB Conversion for Tkinter ---
                if len(viz_frame.shape) == 3 and viz_frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
                else:
                    print(f"Error: Frame is not 3-channel BGR after processing: {viz_frame.shape}")
                    self.display_canvas.delete("all")
                    return 

                # --- Resizing --- 
                h, w = frame_rgb.shape[:2]
                canvas_width = self.display_canvas.winfo_width()
                canvas_height = self.display_canvas.winfo_height()

                if canvas_width > 10 and canvas_height > 10 and w > 0 and h > 0:
                    scale = min(canvas_width / w, canvas_height / h)
                    new_size = (int(w * scale), int(h * scale))
                    if new_size[0] > 0 and new_size[1] > 0:
                         frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_NEAREST)
                    else:
                        print(f"Warning: Calculated resize dimension is zero ({new_size}). Skipping resize.")
                elif w == 0 or h == 0:
                    print(f"Warning: Cannot resize frame with zero dimension ({w}x{h}).")
                    self.display_canvas.delete("all")
                    return

                # --- Display on Canvas ---
                pil_img = Image.fromarray(frame_rgb)
                self.tk_img = ImageTk.PhotoImage(image=pil_img)
                self.display_canvas.delete("all")
                self.display_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status_var.set(f"Display error: {e}")
                self.display_canvas.delete("all")
        else:
            print(f"Warning: Received invalid type for display: {type(frame_to_display)}")
            self.display_canvas.delete("all")
            self.status_var.set("Received invalid frame for display.")
    
    def save_parameters(self):
        """Save current parameters to a file."""
        # Collect current parameters
        params = {}
        for name, slider in self.sliders.items():
            params[name] = slider.get_value()
        
        # Add playground info
        params["playground"] = self.current_playground
        
        # Ensure output directory exists (optional, could save in root)
        # os.makedirs("debug_output", exist_ok=True) 
        
        # Generate filename with timestamp and playground (OLD METHOD)
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # filename = f"debug_output/{self.current_playground}_params_{timestamp}.json"
        
        # --- NEW: Save to default file ---
        filename = DEFAULT_PARAMS_FILE
        # --- End NEW ---
        
        # Save to JSON
        try:
            with open(filename, 'w') as f:
                json.dump(params, f, indent=4)
            
            # --- NEW: Mark params as saved ---
            self.params_changed = False
            # --- End NEW ---
            
            self.status_var.set(f"Parameters saved to {filename}")
            
            # Removed saving to txt file for simplicity, can be re-added if needed
            # Also save to a formatted text file
            # txt_filename = f"debug_output/{self.current_playground}_params_{timestamp}.txt"
            # with open(txt_filename, 'w') as f:
            #     f.write(f"# {self.current_playground.upper()} Fish Detection Parameters\n")
            #     f.write(f"# Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
            #     f.write("# Hough Circle Detection Parameters\n")
            #     for param in ['dp', 'min_distance', 'param1', 'param2']:
            #         if param in params:
            #             f.write(f"detector.{param} = {params[param]}\n")
            #     f.write("\n")
                
            #     f.write("# Size Parameters\n")
            #     for param in ['min_radius', 'max_radius']:
            #         if param in params:
            #             f.write(f"detector.{param} = {params[param]}\n")
            #     f.write("\n")
                
            #     f.write("# Important: Center weight (low value = less center bias)\n")
            #     if 'center_pond_weight' in params:
            #         f.write(f"detector.center_pond_weight = {params['center_pond_weight']}\n")
            #     f.write("\n")
                
            #     f.write(f"# {self.current_playground.upper()} Pond Detection Parameters\n")
            #     f.write(f"detector.{self.current_playground}_params = {{\n")
                
            #     # Get center ratios from current playground parameters
            #     center_x_ratio = 0.5
            #     center_y_ratio = 0.45
            #     if self.current_playground == "dg":
            #         center_x_ratio = 0.5
            #         center_y_ratio = 0.45
            #     elif self.current_playground == "ttc":
            #         center_x_ratio = 0.55
            #         center_y_ratio = 0.35
            #     elif self.current_playground == "bb":
            #         center_x_ratio = 0.5
            #         center_y_ratio = 0.4
                    
            #     f.write(f'    "center_x_ratio": {center_x_ratio},\n')
            #     f.write(f'    "center_y_ratio": {center_y_ratio},\n')
            #     for param in ['h_axis_ratio', 'v_axis_ratio', 'shadow_threshold', 'shadow_min_threshold']:
            #         if param in params:
            #             f.write(f'    "{param}": {params[param]},\n')
            #     f.write("}\n\n")
                
            #     f.write("# Name Tag Masking Parameters\n")
            #     f.write("detector.name_tag_params = {\n")
            #     for param in ['name_tag_x_offset', 'name_tag_y_offset', 'name_tag_width', 'name_tag_height']:
            #         if param in params:
            #             f.write(f'    "{param.replace("name_tag_", "")}": {params[param]},\n')
            #     f.write("}\n")
        except Exception as e:
            self.status_var.set(f"Error saving parameters: {e}")
    
    def reset_parameters(self):
        """Reset parameters to default values for the current playground."""
        # Reset to default parameters for current playground
        pg_name = self.current_playground
        simple_detector = self.simple_detector # Use renamed variable
        core_detector = simple_detector.detector # Reference to core FishDetector
        
        # Reset parameters specific to SimpleFishDetector
        simple_detector.pond_brightness_adjust = 0
        simple_detector.pond_contrast_adjust = 1.0
        simple_detector.adaptive_block_size = 21
        simple_detector.adaptive_c_value = 5
        simple_detector.mask_name_tag = True
        simple_detector.name_tag_params = {
            "x_offset": 0,     # Horizontal offset from player center 
            "y_offset": -195,   # Vertical offset from player center (negative = above)
            "width": 114,       # Width of name tag mask
            "height": 79       # Height of name tag mask
        }
        
        # Reset parameters specific to FishDetector (inside core_detector)
        if core_detector:
            # Call set_playground_params on the core detector
            core_detector.set_playground_params(pg_name)
            # Reset the background flag based on UI toggle state (important!)
            core_detector.use_background_model_flag = self.background_var.get()
            # Reset morphology flag based on UI toggle
            core_detector.apply_morphology = self.morphology_var.get()
            
            # Attempt to reset other core attributes to defaults from FishDetector's structure
            # Need to access FishDetector's internal defaults (e.g., self.ttc_params, self.dg_params etc.)
            default_params = {}
            if pg_name == 'ttc' and hasattr(core_detector, 'ttc_params'):
                default_params = core_detector.ttc_params
            elif pg_name == 'dg' and hasattr(core_detector, 'dg_params'):
                default_params = core_detector.dg_params
            elif pg_name == 'bb' and hasattr(core_detector, 'bb_params'):
                default_params = core_detector.bb_params
            else: # Fallback to general defaults if specific playground params missing
                 default_params = { # Replicate some known defaults as fallback
                      "center_pond_weight": 0.2,
                      "h_axis_ratio": 0.58, 
                      "v_axis_ratio": 0.42, 
                      "shadow_threshold": 0.81, 
                      "shadow_min_threshold": 55, 
                      "min_radius": 10, 
                      "max_radius": 40,
                      "min_area": 120,
                      "max_area": 3700,
                      "circularity_threshold": 0.4,
                      "solidity_threshold": 0.65,
                      "min_aspect_ratio": 0.35,
                      "hough_dp": 1.2,
                      "hough_min_dist": 30,
                      "hough_param1": 50,
                      "hough_param2": 30,
                 }

            # Reset attributes on core_detector using defaults
            core_detector.center_pond_weight = default_params.get("center_pond_weight", 0.2)
            # core_detector.h_axis_ratio = default_params.get("h_axis_ratio", 0.58) # These are set by set_playground_params
            # core_detector.v_axis_ratio = default_params.get("v_axis_ratio", 0.42)
            # core_detector.shadow_threshold = default_params.get("shadow_threshold", 0.81)
            # core_detector.shadow_min_threshold = default_params.get("shadow_min_threshold", 55)
            core_detector.min_radius = default_params.get("min_radius", 10)
            core_detector.max_radius = default_params.get("max_radius", 40)
            core_detector.min_area = default_params.get("min_area", 120)
            core_detector.max_area = default_params.get("max_area", 3700)
            core_detector.circularity_threshold = default_params.get("circularity_threshold", 0.4)
            core_detector.solidity_threshold = default_params.get("solidity_threshold", 0.65)
            core_detector.min_aspect_ratio = default_params.get("min_aspect_ratio", 0.35)
            # Hough parameters are also reset by set_playground_params usually
            # core_detector.hough_dp = default_params.get("hough_dp", 1.2)
            # core_detector.hough_min_dist = default_params.get("hough_min_dist", 30)
            # core_detector.hough_param1 = default_params.get("hough_param1", 50)
            # core_detector.hough_param2 = default_params.get("hough_param2", 30)
            # Reset NEW Hough preprocessing parameters
            core_detector.hough_blur_ksize = default_params.get("hough_blur_ksize", 5)
            core_detector.hough_adapt_block = default_params.get("hough_adapt_block", 21)
            core_detector.hough_adapt_c = default_params.get("hough_adapt_c", 5)
            core_detector.hough_morph_ksize = default_params.get("hough_morph_ksize", 3)

        # Update all sliders with the current values from the detector
        self.update_sliders_from_detector()
        
        # --- NEW: Mark params as unchanged ---
        self.params_changed = False
        # --- End NEW ---
        
        # Update status
        self.status_var.set(f"Parameters for {pg_name} reset to default values")
    
    # --- NEW: Load Parameters Method ---
    def load_parameters(self):
        """Load parameters from the default file if it exists."""
        filename = DEFAULT_PARAMS_FILE
        loaded_successfully = False
        try:
            print(f"Attempting to load parameters from: {filename}") # DEBUG
            if Path(filename).is_file():
                with open(filename, 'r') as f:
                    loaded_params = json.load(f)
                print(f"Successfully read JSON from {filename}") # DEBUG
                
                # Set playground first if it exists in the file
                loaded_pg = loaded_params.get('playground', self.current_playground)
                if loaded_pg in self.playgrounds:
                    if loaded_pg != self.current_playground:
                         print(f"Setting playground from save file: {loaded_pg}") # DEBUG
                         self.current_playground = loaded_pg
                         self.playground_var.set(loaded_pg)
                         # This call resets detector params (including Hough) to the loaded playground's defaults
                         self.simple_detector.set_playground(loaded_pg) 
                    else:
                         print(f"Playground in file matches current: {loaded_pg}") # DEBUG
                else:
                    print(f"Invalid playground '{loaded_pg}' in save file. Using current: {self.current_playground}") # DEBUG
                    # Ensure current playground defaults are set if loaded one was invalid
                    self.simple_detector.set_playground(self.current_playground)

                # Apply loaded parameters to sliders and detector
                applied_count = 0
                skipped_count = 0
                print("Applying loaded parameters to UI/detector...") # DEBUG
                # Important: Temporarily disable params_changed flag during loading
                original_changed_flag = self.params_changed
                self.params_changed = False
                
                for name, value in loaded_params.items():
                    if name == 'playground': continue # Already handled
                    
                    if name in self.sliders:
                        slider_min = self.sliders[name].slider.cget("from")
                        slider_max = self.sliders[name].slider.cget("to")
                        if slider_min <= value <= slider_max:
                             # Use update_parameter to set value in detector first
                             self.update_parameter(name, value)
                             # Then explicitly set slider value
                             self.sliders[name].set_value(value)
                             applied_count += 1
                        else:
                             print(f"Warning: Loaded value {value} for '{name}' outside slider range [{slider_min}-{slider_max}]. Skipping.")
                             skipped_count += 1
                    else:
                         print(f"Warning: Parameter '{name}' found in save file but not in current UI. Ignoring.")
                         skipped_count += 1
                
                # Restore original changed flag state if needed (though it should be false after load)
                self.params_changed = original_changed_flag 
                 
                # Check for parameters in config but missing from file (these will retain their initial default from PARAMETER_CONFIG)
                missing_count = 0
                for name in self.sliders:
                    if name not in loaded_params:
                         missing_count += 1
                print(f"Load complete. Applied: {applied_count}, Skipped/Ignored: {skipped_count}, Missing (used default): {missing_count}") # DEBUG
                         
                # Now, update sliders based on the *actual* detector state 
                # This syncs sliders for params *not* in the save file or skipped due to range issues
                print("Syncing sliders with final detector state after load...") # DEBUG
                self.update_sliders_from_detector() 
                         
                self.status_var.set(f"Loaded parameters from {filename}")
                self.params_changed = False # Parameters are synced 
                loaded_successfully = True
            else:
                 # File not found
                 print(f"Parameter file '{filename}' not found.") # DEBUG
                 self.status_var.set(f"No saved parameters file ({filename}) found. Using defaults.")
                 # Fall through to apply defaults...
                 
        except json.JSONDecodeError as e:
            print(f"ERROR: JSONDecodeError loading {filename}: {e}") # DEBUG
            self.status_var.set(f"Error: Could not decode {filename}. Using defaults.")
            messagebox.showerror("Load Error", f"Failed to parse parameter file '{filename}'.\nIt might be corrupted. Using default values.")
        except Exception as e:
            print(f"ERROR: Unexpected error loading parameters: {e}") # DEBUG
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error loading parameters: {e}")
            messagebox.showerror("Load Error", f"An unexpected error occurred while loading parameters:\n{e}\nUsing default values.")
            
        # --- Fallback Logic --- 
        if not loaded_successfully:
            print("Load failed or file not found. Applying default parameters for current playground and syncing UI.") # DEBUG
            # Ensure detector has defaults for the *current* playground
            self.simple_detector.set_playground(self.current_playground) 
            # Sync the UI sliders to match the detector's actual default state
            self.update_sliders_from_detector() 
            self.params_changed = False # Start with unchanged state
            
    # --- End NEW Method --- 
    
    def on_close(self):
        """Handle window closing."""
        # --- NEW: Check for unsaved changes ---
        if self.params_changed:
            response = messagebox.askyesnocancel("Quit", "You have unsaved parameter changes. Save before quitting?")
            if response is True: # User clicked Yes
                self.save_parameters()
                # Check if saving failed (though unlikely here)
                if self.params_changed: 
                    # Saving might have failed, don't close yet
                    messagebox.showwarning("Save Failed", "Could not save parameters. Please check console for errors.")
                    return # Abort close
                else:
                    self.is_running = False
                    self.destroy()
            elif response is False: # User clicked No
                self.is_running = False
                self.destroy()
            else: # User clicked Cancel or closed the dialog
                return # Abort close
        else:
             # No changes, close directly
             # self.save_parameters() # Removed auto-save if no changes
             self.is_running = False
             self.destroy()
    
    def start(self):
        """Start the application."""
        self.mainloop()

    # Add this new method to handle text area scrolling
    def _on_text_mousewheel(self, event):
        """Handle mousewheel scrolling for the info text widget."""
        # Determine scroll direction based on platform
        if event.delta:  # Windows
            delta = -1 * (event.delta // 120)
        elif hasattr(event, 'num'): # Linux
            delta = -1 if event.num == 4 else 1 if event.num == 5 else 0
        else: # Fallback or other OS
            return
        self.info_text.yview_scroll(delta, "units")
        return "break" # Stop the event from propagating further

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fish Detection Tuner")
    parser.add_argument('--playground', default='dg', help='Playground to tune (dg, ttc, bb)')
    parser.add_argument('--test', action='store_true', help='Enable test mode with additional debugging')
    args = parser.parse_args()
    
    # Start the tuner
    tuner = FishDetectionTuner()
    
    # Configure based on arguments
    if args.playground != 'dg':
        tuner.simple_detector.set_playground(args.playground)
        tuner.playground_var.set(args.playground.upper())
        tuner.change_playground()
        
    # Default to visualization enabled
    tuner.visualization_var.set(True)
    tuner.enable_visualization = True
    tuner.toggle_visualization()
    
    # Enable name tag masking by default
    tuner.name_tag_var.set(True)
    tuner.simple_detector.mask_name_tag = True
    
    tuner.start()

if __name__ == "__main__":
    main() 