# toontown_fishing_bot
Automates fishing in Toontown. Tool assisted automation is against the 
rules of both Corporate Clash (https://corporateclash.net/help/
in-game-rules) and Toontown Rewritten (https://toontownrewritten.com/play/
terms-of-service); this project is for educational and learning purposes.

# Game Window Capture Tool

A Python-based screen capture tool designed to capture and display game window content in real-time. This project evolved from solving specific challenges in game automation, particularly the need for reliable window capture regardless of window focus.

## The Implementation Story

### The Initial Problem
Our journey began with a specific challenge: we needed a screen capture system that could:
1. Continuously monitor a game window
2. Work even when the window is not in focus
3. Capture the entire window area reliably
4. Handle window movements and state changes

### First Attempt: MSS Library
We started with the `mss` library, a popular Python screen capture tool. While it seemed promising initially, we quickly hit several limitations:
- The capture would only work when the game window was in focus
- Alt-tabbing away from the game would cause capture failures
- Window tracking became unreliable when the window moved
- Performance wasn't consistent enough for real-time capture

This approach taught us that simple screen region capture wasn't sufficient for our needs. We needed something more robust.

### The Breakthrough: Windows API
After researching alternatives, we discovered that Windows provides direct access to window content through window handles. This led us to the `win32gui` and `win32ui` libraries, which opened up new possibilities:
1. We could find any window by its title, even in the background
2. We could get exact window dimensions, including accounting for borders
3. Most importantly, we could capture window content directly, regardless of focus

### Solving Technical Challenges

#### 1. Window Detection and Tracking
- Implemented `win32gui.EnumWindows` to reliably find our target window
- Used `GetWindowRect` and `GetClientRect` to handle window borders correctly
- Added automatic position updating to follow window movements

#### 2. Efficient Capture System
- Created a device context (DC) management system
- Implemented proper resource cleanup using context managers
- Added cursor overlay support for complete capture

#### 3. Performance Optimization
- Implemented frame rate control
- Added performance monitoring
- Optimized resource usage and cleanup
- Added error recovery mechanisms

## Features

- Real-time window capture with cursor overlay
- Performance monitoring (FPS, capture time, frame time)
- Flexible capture regions (full window or center)
- Resource-efficient capture using Windows API
- Automatic window tracking and repositioning

## Technical Implementation

### Core Components

1. **Window Management**
   ```python
   def find_game_window(window_title: str) -> int:
       """Find game window by title and return its handle."""
       def window_callback(handle: int, results: List[Optional[int]]) -> None:
           if win32gui.IsWindowVisible(handle) and window_title in win32gui.GetWindowText(handle):
               results[0] = handle
   ```

2. **Capture System**
   ```python
   def capture_window_content(window_handle: int, region: Tuple[int, int, int, int]):
       """Capture window content with cursor overlay."""
       with win32_dc_resources() as resources:
           # Create device contexts
           window_dc = win32gui.GetWindowDC(window_handle)
           # ... capture implementation
   ```

3. **Resource Management**
   ```python
   @contextmanager
   def win32_dc_resources():
       """Context manager for handling win32 DC resources."""
       resources = []
       try:
           yield resources
       finally:
           # Clean up resources
   ```

## Requirements

- Windows OS (tested on Windows 10)
- Python 3.6 or higher
- Required Python packages:
  - `pywin32` - Windows API interface
  - `Pillow` - Image processing
  - `tkinter` - GUI (usually comes with Python)

## Installation

1. Clone this repository or download the source code
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage (capture entire window):
```python
python screen_capture.py
```

2. In your code:
```python
from screen_capture import GameCaptureWindow

# Capture full window
capture = GameCaptureWindow("Window Title", "full")
capture.start()

# Capture center region (50% of window)
capture = GameCaptureWindow("Window Title", "center")
capture.start()
```

## Configuration

The following constants can be adjusted in `screen_capture.py`:

- `PERFORMANCE_BUFFER_SIZE`: Number of frames to keep for performance metrics (default: 100)
- `TARGET_FPS`: Target frames per second for capture (default: 30)

## Performance Metrics

The tool displays real-time performance metrics:
- FPS: Current frames per second
- Capture: Time taken to capture each frame (ms)
- Frame: Total frame processing time (ms)

## Lessons Learned

Through this development process, we learned several valuable lessons:
1. Sometimes the obvious solution (screen capture library) isn't the best solution
2. Understanding the underlying system (Windows API) can lead to better solutions
3. Proper resource management is crucial for reliable operation
4. Error handling and recovery are as important as the core functionality

## Known Limitations

- Windows-only support
- May have reduced performance with very large windows
- Some games using hardware acceleration might not be captured correctly

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.