# AirDesk

Webcam hand-gesture controller built with Python, OpenCV, MediaPipe (`HandLandmarker`), and PyAutoGUI.

## Gestures
- **Thumb Up (other 4 fingers folded):** Volume Down
- **Index Finger Only Up:** Volume Up
- **Open Palm (4 fingers up):** Mute / Unmute system audio
- **Index + Pinky Up:** Play / Pause video (`Space`)
- **Closed Fist:** Take screenshot (`screenshots/` folder)

> **Cooldown / Spare Time:** A `1.0s` cooldown (`ACTION_COOLDOWN`) is set between actions so you have enough time to reposition your hand between movements.

## Setup & Run
1. Make sure `hand_landmarker.task` is in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt