# Automated Fishing Bot for Toontown: Corporate Clash

## Objective

Create a fully automated, reliable fishing bot for **Toontown: Corporate Clash**, using Python and computer vision methods, along with any visualization methods to allow for the development process to be turned into an entertaining YouTube video.

---

## Technical Stack

- **Language:** Python
- **Libraries:**
  - **Screen Capture:** Win32 API
  - **Computer Vision:** OpenCV
  - **Mouse Simulation:** PyDirectInput
  - **Visualization:** OpenCV overlay, Win32gui

---

## Functional Requirements

### 1. Screen Capture

- Real-time capture of specific region of the screen at ~30 FPS using Win32 API.
- Easy-to-configure capture region.
- Context-managed resources and error handling.
- Continuous window tracking.
- Toontown Cursor rendering.
- Real-time performance metrics (FPS, capture time, frame time).

### 2. Fish Detection

- Real-time detection of dark circular fish shadows:
  - Grayscale conversion
  - Binary thresholding
  - Contour detection
- Configurable area threshold filters.
- Real-time visual validation with OpenCV.

### 3. Casting Mechanics

- Accurate mouse drag simulation away from target fish to cast toward it.
- Configurable coordinates for cast button.
- Visual validation of mouse movements.

### 4. Integrated Casting Automation

- Continuous fish detection and casting loop.
- Target selection (largest stationary or slowest-moving fish).
- Adjustable casting delays (~2 seconds).
- Automated scenario testing.

### 5. Transparent Visualization Overlay

- Semi-transparent overlay matching capture region.
- Visual indicators:
  - Green circles around detected fish.
  - Red line for casting direction.
- Always-on-top transparency with Win32gui.

### 6. Popup Automation

- Automatic detection and closure of catch popups:
  - Template matching or pixel color detection.
  - Simulated mouse clicks or keypresses (ESC, ENTER).

### 7. Statistics and Logging

- Timestamp logging for each successful catch.
- Metrics logging:
  - Total catches
  - Catches per minute
  - Average catch interval
- Validated via simulated catches.

### 8. Fish Bucket Detection

- Internal tracking to bucket limit (20 fish).
- Automatic pause and log notification upon reaching limit.

---

## Advanced Features

### 9. Automated Fish Selling

- Navigation automation to NPC vendor and back:
  - Keyboard/mouse interaction simulation.
  - Automated vendor dialogue and confirmation.
  - Automatic return and resume fishing.

### 10. Bingo Automation

- Glowing bingo square detection via brightness analysis.
- Automated clicking on valid squares.
- Simulated scenarios for validation.

---

## Non-Functional Requirements

### Performance

- Minimal resource consumption.
- No noticeable latency during normal PC usage.

### Usability

- Configurable settings (coordinates, delays, thresholds).
- Clear debugging and monitoring logs.

### Reliability

- Comprehensive error handling.
- Clear logging and user feedback.

### Security and Compliance

- No client injection/modification.
- Human-like interaction patterns to minimize detection.

### Visualization

- Real-time overlays suitable for demonstrations and YouTube content.
- Optional statistical graph plotting.

---

## Implementation Roadmap

### Milestone 1: MVP (Basic Automation)

- [Done] Advanced screen capture setup and validation [Done: implemented via `screen_capture.py`]
- [ ] Basic fish detection implementation and visual validation.
- [ ] Fundamental casting mechanics development and validation.
- [ ] Popup handling automation.
- [ ] Basic overlay visualization.

### Milestone 2: Robust Automation

- [ ] Integration of detection and casting automation.
- [ ] Bucket limit detection and automated handling.
- [ ] Logging and statistics integration.
- [ ] Comprehensive validation testing.

### Milestone 3: Advanced Capabilities

- [ ] Fish selling automation (navigation and interaction).
- [ ] Bingo detection and interaction automation.
- [ ] Performance optimization and enhanced visualization.
- [ ] Extensive scenario validation testing.

---

## Testing and Validation

- Incremental unit testing.
- Visual verification via overlays.
- Automated milestone validation scenarios.

---

## Deliverables

- Python script (`toontown_fishing_bot.py`).
- Comprehensive documentation for setup and troubleshooting.
- Demonstration video showcasing full functionality with visual overlays and logs.