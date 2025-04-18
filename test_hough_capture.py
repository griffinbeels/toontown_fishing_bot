import time
from screen_capture import GameCaptureWindow
from fish_detection import FishDetector

def main():
    """
    Test the improved fish detection with Hough Circle Transform 
    in real-time using screen capture.
    """
    print("Starting Toontown Fishing Bot with Hough Circle Transform detection...")
    print("Press Ctrl+C to exit")
    
    # Initialize detection with Hough Circle Transform enabled
    detector = FishDetector()
    detector.use_hough_transform = True
    
    # Create game capture window
    capture_window = GameCaptureWindow("Corporate Clash", "full")
    
    # Update the fish detector in the capture window
    capture_window.fish_detector = detector
    
    # Run the capture window
    try:
        capture_window.start()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Cleanup
        if hasattr(capture_window, 'close_window'):
            capture_window.close_window()

if __name__ == "__main__":
    main() 