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
        
        # Calculate border sizes by comparing full window to client area
        border_width = (window_rect[2] - window_rect[0]) - client_rect[2]
        border_height = (window_rect[3] - window_rect[1]) - client_rect[3]
        
        # Return adjusted coordinates to exclude borders
        return (
            window_rect[0] + border_width // 2,  # Left edge + half border
            window_rect[1] + border_height - border_width // 2,  # Top edge + full border
            client_rect[2],  # Client width
            client_rect[3]   # Client height
        )
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
            capture_width, capture_height = region[2:4]
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc_obj, capture_width, capture_height)
            mem_dc.SelectObject(bitmap)  # Select bitmap into memory DC
            resources.append(bitmap.GetHandle())  # Track bitmap handle
            
            # Copy window content to memory DC
            window_rect = win32gui.GetWindowRect(window_handle)
            mem_dc.BitBlt(
                (0, 0),  # Destination position
                (capture_width, capture_height),  # Size
                dc_obj,  # Source DC
                (region[0] - window_rect[0], region[1] - window_rect[1]),  # Source position
                win32con.SRCCOPY  # Copy operation
            )
            
            # Draw cursor if visible
            cursor_info = win32gui.GetCursorInfo()
            if cursor_info[1]:  # Check if cursor is visible
                cursor_pos = win32gui.GetCursorPos()
                # Calculate cursor position relative to capture region
                cursor_x = cursor_pos[0] - region[0]
                cursor_y = cursor_pos[1] - region[1]
                
                # Only draw cursor if it's within capture region
                if 0 <= cursor_x < capture_width and 0 <= cursor_y < capture_height:
                    cursor_width = win32api.GetSystemMetrics(win32con.SM_CXCURSOR)
                    cursor_height = win32api.GetSystemMetrics(win32con.SM_CYCURSOR)
                    win32gui.DrawIconEx(
                        mem_dc.GetHandleOutput(),  # DC to draw on
                        cursor_x, cursor_y,        # Position
                        cursor_info[1],            # Cursor handle
                        cursor_width, cursor_height,# Size
                        0, None, CURSOR_DRAW_MODE  # Drawing flags
                    )
            
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
        
        try:
            # Find and setup game window
            self.game_window_handle = find_game_window(window_title)
            window_dimensions = get_game_window_dimensions(self.game_window_handle)
            
            self.region_type = region_type
            self.capture_region = calculate_capture_region(window_dimensions, region_type)
            
            self._setup_display_window()
            self.is_running = True
            self.capture_and_display_loop()
            
        except (WindowNotFoundError, CaptureError) as e:
            print(f"Setup error: {e}")
            self.display_window.destroy()
    
    def _setup_display_window(self) -> None:
        """Initialize display window and canvas."""
        # Set window size to match capture region
        self.display_window.geometry(f"{self.capture_region[2]}x{self.capture_region[3]}")
        
        # Create canvas for displaying captured frames
        self.display_canvas = tk.Canvas(
            self.display_window,
            width=self.capture_region[2],
            height=self.capture_region[3],
            highlightthickness=0  # Remove canvas border
        )
        self.display_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize timing and display variables
        self.last_frame_time = time.perf_counter()
        self.current_photo = None  # Current frame being displayed
        self.display_window.protocol("WM_DELETE_WINDOW", self.close_window)
    
    def update_capture_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Update capture region based on current window position."""
        try:
            window_dimensions = get_game_window_dimensions(self.game_window_handle)
            return calculate_capture_region(window_dimensions, self.region_type)
        except CaptureError:
            return None
    
    def capture_and_display_loop(self) -> None:
        """Main capture and display loop that continuously:
        1. Captures the game window content
        2. Updates the display with the captured frame
        3. Maintains target FPS through frame timing
        4. Handles window tracking and error recovery
        5. Updates performance metrics
        """
        if not self.is_running:
            return
        
        frame_start = time.perf_counter()  # Start timing frame
        
        try:
            # Update capture region to follow game window
            new_region = self.update_capture_region()
            if new_region is None:
                raise CaptureError("Lost track of game window")
            
            self.capture_region = new_region
            captured_frame, capture_time = capture_window_content(
                self.game_window_handle,
                self.capture_region
            )
            
            if captured_frame:
                self.capture_times.append(capture_time)
                self._update_display(captured_frame, frame_start)
            
        except CaptureError as e:
            print(f"Capture error: {e}")
            self.close_window()
            return
        
        # Calculate delay for next frame to maintain target FPS
        frame_time = (time.perf_counter() - frame_start) * 1000
        self.frame_times.append(frame_time)
        delay = max(1, int((1000/TARGET_FPS) - frame_time))
        self.display_window.after(delay, self.capture_and_display_loop)
    
    def _update_display(self, frame: Image.Image, frame_start: float) -> None:
        """Update display with new frame and performance metrics."""
        frame_draw = ImageDraw.Draw(frame)
        # Calculate performance metrics
        avg_capture = statistics.mean(self.capture_times) if self.capture_times else 0
        avg_frame = statistics.mean(self.frame_times) if self.frame_times else 0
        fps = 1000 / avg_frame if avg_frame > 0 else 0
        
        # Draw performance metrics on frame
        frame_draw.text(
            (10, 10),
            f"FPS: {fps:.1f}\n"
            f"Capture: {avg_capture:.1f}ms\n"
            f"Frame: {avg_frame:.1f}ms",
            fill="lime"
        )
        
        # Update display with new frame
        self.current_photo = ImageTk.PhotoImage(frame)
        self.display_canvas.create_image(0, 0, image=self.current_photo, anchor=tk.NW)
    
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