# AirDesk

Webcam hand-gesture controller built with Python, OpenCV, MediaPipe (`HandLandmarker`), and PyAutoGUI.

## Gestures
- **Closed Fist:** Take screenshot (`screenshots/` folder)
- **Open Palm (4 fingers up):** Mute / Unmute system audio
- **Index + Pinky Up:** Play / Pause video (`Space`)
- **Index Finger Only Up:** Volume Up (`y < 0.38`) / Volume Down (`y > 0.58`)

## Setup & Run
1. Make sure `hand_landmarker.task` is in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt