# Windows API imports for window manipulation and screen capture
import win32gui  # Core Windows GUI functions
import win32ui   # Windows UI functions for device contexts
import win32con  # Windows constants
import win32api  # Windows API functions
import time      # For performance timing
from PIL import Image, ImageDraw, ImageTk  # Image processing and GUI display
import tkinter as tk  # GUI framework
from typing import Tuple, Optional, List  # Type hints
from collections import deque  # Efficient queue for performance metrics
import statistics  # For calculating averages
from contextlib import contextmanager  # For resource management
import numpy as np  # For image processing
import cv2  # For computer vision tasks
from fish_detection import FishDetector  # Import our fish detection module

# Performance and display constants
PERFORMANCE_BUFFER_SIZE = 100  # Number of frames to keep for performance metrics
TARGET_FPS = 30  # Target frames per second for smooth display
CURSOR_DRAW_MODE = 0x0003  # Windows constant for normal cursor drawing

class WindowNotFoundError(Exception):
    """Raised when target window cannot be found."""
    pass

class CaptureError(Exception):
    """Raised when screen capture fails."""
    pass

@contextmanager
def win32_dc_resources():
    """Context manager for handling win32 DC resources."""
    resources = []  # Track all created resources for cleanup
    try:
        yield resources
    finally:
        # Clean up all resources in reverse order
        for resource in resources:
            try:
                if isinstance(resource, int):
                    win32gui.DeleteObject(resource)  # Clean up bitmap handles
                else:
                    try:
                        resource.DeleteDC()  # Clean up device contexts
                    except Exception:
                        # If DeleteDC fails, try to release the DC
                        try:
                            win32gui.ReleaseDC(0, resource)
                        except Exception:
                            pass  # Ignore cleanup errors
            except Exception:
                pass  # Ignore cleanup errors to ensure all resources are attempted

# --- Core Capture Logic (Refactored into ScreenCapturer) ---

class ScreenCapturer:
    """Handles finding and capturing content from a specific window."""
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.window_handle = self._find_game_window()
        self.last_capture_time = 0.0

    def _find_game_window(self) -> int:
        """Find game window by title and return its handle."""
        def window_callback(handle: int, results: List[Optional[int]]) -> None:
            if win32gui.IsWindowVisible(handle) and self.window_title in win32gui.GetWindowText(handle):
                results[0] = handle

        found_handle = [None]
        win32gui.EnumWindows(window_callback, found_handle)

        if found_handle[0] is None:
            raise WindowNotFoundError(f"Could not find window with title '{self.window_title}'")
        return found_handle[0]

    def get_capture_dimensions(self) -> Tuple[int, int, int, int]:
        """Get game window client area dimensions and position."""
        try:
            # Get client area rectangle relative to window (top-left is 0,0)
            client_rect = win32gui.GetClientRect(self.window_handle)
            width = client_rect[2] - client_rect[0]
            height = client_rect[3] - client_rect[1]

            # Convert client area top-left corner to screen coordinates
            client_origin = win32gui.ClientToScreen(self.window_handle, (client_rect[0], client_rect[1]))

            return (client_origin[0], client_origin[1], width, height)

        except Exception as e:
            # Check if window still exists
            if not win32gui.IsWindow(self.window_handle):
                raise WindowNotFoundError(f"Window '{self.window_title}' no longer exists.")
            raise CaptureError(f"Failed to get window dimensions: {e}")

    def capture_frame(self) -> Tuple[Optional[Image.Image], float]:
        """Capture the client area of the window with the cursor."""
        capture_start = time.perf_counter()
        window_dc = None

        try:
            capture_region = self.get_capture_dimensions()
            capture_left, capture_top, capture_width, capture_height = capture_region

            # Ensure valid dimensions
            if capture_width <= 0 or capture_height <= 0:
                # print(f"Warning: Invalid capture dimensions {capture_width}x{capture_height}. Skipping frame.")
                return None, 0.0

            with win32_dc_resources() as resources:
                # Get DC for the entire window
                # Using GetWindowDC captures the whole window including title bar/borders
                # We need this to capture the cursor correctly even if it's slightly outside client area
                window_dc = win32gui.GetWindowDC(self.window_handle)
                if window_dc == 0:
                     raise CaptureError(f"Failed to get Window DC for handle {self.window_handle}")
                     
                dc_obj = win32ui.CreateDCFromHandle(window_dc)
                mem_dc = dc_obj.CreateCompatibleDC()
                resources.extend([dc_obj, mem_dc])

                bitmap = win32ui.CreateBitmap()
                bitmap.CreateCompatibleBitmap(dc_obj, capture_width, capture_height)
                mem_dc.SelectObject(bitmap)
                resources.append(bitmap.GetHandle())

                # --- Correct BitBlt Source Calculation ---
                # Get the window's top-left corner on the screen
                win_rect = win32gui.GetWindowRect(self.window_handle)
                window_left, window_top = win_rect[0], win_rect[1]

                # Calculate the client area's top-left offset relative to the window's top-left
                # capture_left/top are screen coordinates of the client area
                source_x = capture_left - window_left
                source_y = capture_top - window_top
                # --- End Correction ---
                
                # Copy client area content from window DC to memory DC
                # Source position is the client area's offset within the window DC
                mem_dc.BitBlt(
                    (0, 0), 
                    (capture_width, capture_height), 
                    dc_obj, 
                    (source_x, source_y), # Use calculated offset
                    win32con.SRCCOPY
                )

                # --- REMOVE CURSOR DRAWING TO PREVENT DETECTION INTERFERENCE ---
                # Draw cursor if visible
                # cursor_info = win32gui.GetCursorInfo()
                # if cursor_info[1]: # Check if cursor is visible (cursor handle)
                #     cursor_pos_screen = win32gui.GetCursorPos() # Cursor pos on screen
                #     # Calculate cursor position relative to the captured client area (top-left)
                #     cursor_x = cursor_pos_screen[0] - capture_left
                #     cursor_y = cursor_pos_screen[1] - capture_top

                #     if 0 <= cursor_x < capture_width and 0 <= cursor_y < capture_height:
                #         cursor_width = win32api.GetSystemMetrics(win32con.SM_CXCURSOR)
                #         cursor_height = win32api.GetSystemMetrics(win32con.SM_CYCURSOR)
                #         try:
                #             win32gui.DrawIconEx(
                #                 mem_dc.GetHandleOutput(),
                #                 cursor_x, cursor_y,
                #                 cursor_info[1],
                #                 cursor_width, cursor_height,
                #                 0, None, CURSOR_DRAW_MODE
                #             )
                #         except Exception as icon_e:
                #             # Ignore potential errors drawing specific cursors
                #             # print(f"Warning: Could not draw cursor icon: {icon_e}")
                #             pass 
                # --- END REMOVED CURSOR DRAWING ---

                # Convert bitmap to PIL Image
                bmp_info = bitmap.GetInfo()
                bmp_str = bitmap.GetBitmapBits(True)
                image = Image.frombuffer(
                    'RGB', 
                    (bmp_info['bmWidth'], bmp_info['bmHeight']), 
                    bmp_str, 'raw', 'BGRX', 0, 1
                )

                capture_time = (time.perf_counter() - capture_start) * 1000
                self.last_capture_time = capture_time
                return image, capture_time

        except Exception as e:
             # Check if window handle is still valid
            if not win32gui.IsWindow(self.window_handle):
                 raise WindowNotFoundError(f"Window '{self.window_title}' closed or invalid handle.")
            # Specific check for GetWindowDC failure common when window minimized/occluded
            if isinstance(e, win32ui.error) and "GetDC" in str(e):
                # Don't raise, just return None - window likely minimized or occluded
                # print(f"Skipping capture: Could not get Window DC (possibly minimized/occluded). Error: {e}")
                return None, 0.0 
            # Check for BitBlt errors (often parameter issues)
            if isinstance(e, win32ui.error) and "BitBlt" in str(e):
                 print(f"BitBlt Error: {e}. Capture dimensions: {capture_region}")
                 raise CaptureError(f"BitBlt failed: {e}")

            # General capture error
            # Avoid raising error repeatedly if window is just hidden/minimized
            # print(f"Capture error: {e}") # Log other errors if needed
            raise CaptureError(f"Failed to capture window content: {e}")
            
        finally:
            if window_dc:
                try:
                    # Release the Window DC, not the client DC
                    win32gui.ReleaseDC(self.window_handle, window_dc)
                except Exception:
                    pass

# --- Standalone Functions (Deprecated by ScreenCapturer, kept for potential compatibility) ---

def find_game_window(window_title: str) -> int:
    """Find game window by title and return its handle."""
    def window_callback(handle: int, results: List[Optional[int]]) -> None:
        # Check if window is visible and title matches
        if win32gui.IsWindowVisible(handle) and window_title in win32gui.GetWindowText(handle):
            results[0] = handle

    found_handle = [None]  # Use list for mutable reference in callback
    win32gui.EnumWindows(window_callback, found_handle)  # Enumerate all windows
    
    if found_handle[0] is None:
        raise WindowNotFoundError(f"Could not find window with title '{window_title}'")
    return found_handle[0]

def get_game_window_dimensions(window_handle: int) -> Tuple[int, int, int, int]:
    """Get game window dimensions accounting for borders."""
    try:
        # Get full window rectangle including borders
        window_rect = win32gui.GetWindowRect(window_handle)
        # Get client area rectangle (game content)
        client_rect = win32gui.GetClientRect(window_handle)
        
        # Convert client origin to screen coordinates
        client_origin = win32gui.ClientToScreen(window_handle, (client_rect[0], client_rect[1]))
        client_width = client_rect[2] - client_rect[0]
        client_height = client_rect[3] - client_rect[1]

        return (client_origin[0], client_origin[1], client_width, client_height)
        
    except Exception as e:
        raise CaptureError(f"Failed to get window dimensions: {e}")

def calculate_capture_region(
    window_dimensions: Tuple[int, int, int, int],
    region_type: str = "full"
) -> Tuple[int, int, int, int]:
    """Calculate capture region within game window."""
    window_left, window_top, window_width, window_height = window_dimensions
    
    if region_type == "full":
        return window_left, window_top, window_width, window_height
    elif region_type == "center":
        # Calculate dimensions for center region (50% of window size)
        capture_width = window_width // 2
        capture_height = window_height // 2
        # Center the capture region in the window
        return (
            window_left + (window_width - capture_width) // 2,
            window_top + (window_height - capture_height) // 2,
            capture_width,
            capture_height
        )
    # Default to full if region_type is unknown
    return window_left, window_top, window_width, window_height

def capture_window_content(
    window_handle: int,
    region: Tuple[int, int, int, int]
) -> Tuple[Optional[Image.Image], float]:
    """Capture window content with cursor."""
    capture_start = time.perf_counter()  # Start timing the capture
    window_dc = None
    
    with win32_dc_resources() as resources:
        try:
            # Create device contexts for screen capture
            window_dc = win32gui.GetWindowDC(window_handle)  # Get window's DC
            dc_obj = win32ui.CreateDCFromHandle(window_dc)   # Create DC object
            mem_dc = dc_obj.CreateCompatibleDC()            # Create memory DC
            resources.extend([dc_obj, mem_dc])  # Track for cleanup
            
            # Create bitmap for the capture
            capture_left, capture_top, capture_width, capture_height = region
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc_obj, capture_width, capture_height)
            mem_dc.SelectObject(bitmap)  # Select bitmap into memory DC
            resources.append(bitmap.GetHandle())  # Track bitmap handle
            
            # Find client area origin relative to window origin
            client_rect = win32gui.GetClientRect(window_handle)
            client_origin_in_window = win32gui.ClientToScreen(window_handle, (client_rect[0], client_rect[1]))
            window_origin = win32gui.GetWindowRect(window_handle)[:2] # Top-left corner of the window
            client_offset_x = client_origin_in_window[0] - window_origin[0]
            client_offset_y = client_origin_in_window[1] - window_origin[1]

            # Copy specified region content from window DC to memory DC
            # Source position needs to be relative to the window DC's origin (window top-left)
            source_x = (capture_left - window_origin[0])
            source_y = (capture_top - window_origin[1])
            
            mem_dc.BitBlt(
                (0, 0),  # Destination position (memory DC top-left)
                (capture_width, capture_height),  # Size
                dc_obj,  # Source DC (Window DC)
                (source_x, source_y), # Source position (capture region top-left relative to window DC)
                win32con.SRCCOPY  # Copy operation
            )
            
            # Draw cursor if visible
            cursor_info = win32gui.GetCursorInfo()
            if cursor_info[1]:  # Check if cursor is visible
                cursor_pos_screen = win32gui.GetCursorPos()
                # Calculate cursor position relative to capture region's top-left
                cursor_x = cursor_pos_screen[0] - capture_left
                cursor_y = cursor_pos_screen[1] - capture_top
                
                # Only draw cursor if it's within capture region
                if 0 <= cursor_x < capture_width and 0 <= cursor_y < capture_height:
                    cursor_width = win32api.GetSystemMetrics(win32con.SM_CXCURSOR)
                    cursor_height = win32api.GetSystemMetrics(win32con.SM_CYCURSOR)
                    try:
                        win32gui.DrawIconEx(
                            mem_dc.GetHandleOutput(),  # DC to draw on
                            cursor_x, cursor_y,        # Position
                            cursor_info[1],            # Cursor handle
                            cursor_width, cursor_height,# Size
                            0, None, CURSOR_DRAW_MODE  # Drawing flags
                        )
                    except Exception as icon_e:
                        # print(f"Warning: Could not draw cursor icon: {icon_e}")
                        pass
            
            # Convert bitmap to PIL Image
            bmp_info = bitmap.GetInfo()
            bmp_str = bitmap.GetBitmapBits(True)
            image = Image.frombuffer(
                'RGB',  # Color mode
                (bmp_info['bmWidth'], bmp_info['bmHeight']),  # Dimensions
                bmp_str, 'raw', 'BGRX', 0, 1  # Raw bitmap data
            )
            
            capture_time = (time.perf_counter() - capture_start) * 1000  # Calculate capture time
            return image, capture_time
            
        except Exception as e:
            if not win32gui.IsWindow(window_handle):
                raise WindowNotFoundError("Window not found during capture.")
            # Specific check for GetWindowDC failure
            if isinstance(e, win32ui.error) and "GetDC" in str(e):
                 # print(f"Skipping capture: Could not get Window DC. Error: {e}")
                 return None, 0.0
            if isinstance(e, win32ui.error) and "BitBlt" in str(e):
                 print(f"BitBlt Error: {e}. Capture region: {region}")
                 raise CaptureError(f"BitBlt failed: {e}")
                 
            raise CaptureError(f"Failed to capture window content: {e}")
        finally:
            if window_dc:
                try:
                    win32gui.ReleaseDC(window_handle, window_dc)  # Always release window DC
                except Exception:
                    pass  # Ignore cleanup errors

class GameCaptureWindow:
    """Window for displaying captured game content."""
    
    def __init__(self, window_title: str, region_type: str = "full"):
        self.display_window = tk.Tk()  # Create main window
        self.display_window.title("Game Capture")
        
        # Initialize performance tracking
        self.capture_times = deque(maxlen=PERFORMANCE_BUFFER_SIZE)
        self.frame_times = deque(maxlen=PERFORMANCE_BUFFER_SIZE)
        
        # Initialize fish detector
        self.fish_detector = FishDetector()
        
        try:
            # Use ScreenCapturer for finding and getting initial dimensions
            self.capturer = ScreenCapturer(window_title)
            # Get initial dimensions to set window size
            initial_dimensions = self.capturer.get_capture_dimensions()
            self.capture_region_width = initial_dimensions[2]
            self.capture_region_height = initial_dimensions[3]
            
            # Setup display window based on captured dimensions
            self._setup_display_window(self.capture_region_width, self.capture_region_height)
            self.is_running = True
            self.capture_and_display_loop()
            
        except (WindowNotFoundError, CaptureError) as e:
            print(f"Setup error: {e}")
            self.display_window.destroy()
    
    def _setup_display_window(self, width: int, height: int) -> None:
        """Initialize display window and canvas."""
        # Set window size to match capture region
        self.display_window.geometry(f"{width}x{height}")
        
        # Create canvas for displaying captured frames
        self.display_canvas = tk.Canvas(
            self.display_window,
            width=width,
            height=height,
            highlightthickness=0  # Remove canvas border
        )
        self.display_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize timing and display variables
        self.last_frame_time = time.perf_counter()
        self.current_photo = None  # Current frame being displayed
        self.display_window.protocol("WM_DELETE_WINDOW", self.close_window)
    
    def update_capture_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Update capture region based on current window position."""
        # This is handled by the capturer now, but we might need dimensions
        try:
            # We just need the dimensions, ScreenCapturer handles position
            return self.capturer.get_capture_dimensions()
        except (CaptureError, WindowNotFoundError):
            return None
    
    def capture_and_display_loop(self) -> None:
        """Main capture and display loop using ScreenCapturer."""
        if not self.is_running:
            return
        
        frame_start = time.perf_counter() # Start timing frame

        try:
            # Capture frame using ScreenCapturer
            # ScreenCapturer handles finding the window and region internally
            captured_frame_data = self.capturer.capture_frame()
            
            # Check if capture was successful (returns None if window minimized/etc)
            if captured_frame_data is None or captured_frame_data[0] is None:
                # print("Capture skipped (window likely minimized or inaccessible).")
                # Continue the loop after a delay, don't update display
                delay = max(1, int((1000/TARGET_FPS) - ((time.perf_counter() - frame_start) * 1000)))
                self.display_window.after(delay, self.capture_and_display_loop)
                return
                
            captured_frame, capture_time = captured_frame_data
            self.capture_times.append(capture_time)
            self._update_display(captured_frame, frame_start)
            
        except WindowNotFoundError as e:
            print(f"Error: {e}. Stopping capture.")
            self.close_window()
            return
        except CaptureError as e:
            print(f"Capture error: {e}")
            # Potentially retry or close depending on the error
            # For now, just log and continue the loop after delay
            delay = max(1, int((1000/TARGET_FPS) - ((time.perf_counter() - frame_start) * 1000)))
            self.display_window.after(delay, self.capture_and_display_loop)
            return
        except Exception as e:
            # Catch unexpected errors during the loop
            print(f"Unexpected error in capture loop: {e}")
            import traceback
            traceback.print_exc()
            self.close_window()
            return

        # Calculate delay for next frame
        frame_time = (time.perf_counter() - frame_start) * 1000
        self.frame_times.append(frame_time)
        delay = max(1, int((1000/TARGET_FPS) - frame_time))
        self.display_window.after(delay, self.capture_and_display_loop)

    def _update_display(self, frame: Image.Image, frame_start: float) -> None:
        """Update display with new frame and performance metrics."""
        # Convert PIL Image to numpy array for OpenCV processing
        frame_np = np.array(frame)
        # Note: PIL images are RGB, but OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Detect fish in the frame
        tracked_fish = self.fish_detector.detect_fish(frame_bgr, playground="dg")
        
        # Draw debug visualization
        debug_frame = self.fish_detector.draw_debug(frame_bgr)
        
        # Convert back to RGB for PIL
        frame_rgb = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        
        # Convert back to PIL Image
        frame = Image.fromarray(frame_rgb)
        
        frame_draw = ImageDraw.Draw(frame)
        # Calculate performance metrics
        avg_capture = statistics.mean(self.capture_times) if self.capture_times else 0
        avg_frame = statistics.mean(self.frame_times) if self.frame_times else 0
        fps = 1000 / avg_frame if avg_frame > 0 else 0
        
        # Draw performance metrics and fish count
        frame_draw.text(
            (10, 10),
            f"FPS: {fps:.1f}\n"
            f"Capture: {avg_capture:.1f}ms\n"
            f"Frame: {avg_frame:.1f}ms\n"
            f"Fish: {len(tracked_fish)}",
            fill="lime"
        )
        
        # Update display with new frame
        try:
            # Ensure canvas still exists before updating
            if self.display_canvas.winfo_exists():
                self.current_photo = ImageTk.PhotoImage(frame)
                self.display_canvas.create_image(0, 0, image=self.current_photo, anchor=tk.NW)
        except tk.TclError:
             # Handle race condition where window is closed during update
             print("Display update skipped: Window closed.")
             pass
    
    def close_window(self) -> None:
        """Clean up resources and close window."""
        self.is_running = False
        self.display_window.destroy()
    
    def start(self) -> None:
        """Start the capture window."""
        self.display_window.mainloop()

def main() -> None:
    """Main entry point."""
    capture_window = GameCaptureWindow("Corporate Clash", "full")
    capture_window.start()

if __name__ == "__main__":
    main() 