from pathlib import Path
import time
import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
SHOTS_DIR = BASE_DIR / "screenshots"
SHOTS_DIR.mkdir(exist_ok=True)

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

# Finger landmark indices: [index, middle, ring, pinky]
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def get_fingers_up(landmarks):
    return [
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
    ]


def draw_hand(frame, landmarks, w, h):
    for a, b in HAND_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(
            frame,
            (int(pa.x * w), int(pa.y * h)),
            (int(pb.x * w), int(pb.y * h)),
            (180, 180, 180),
            2,
        )
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 100), -1)


cap = cv2.VideoCapture(0)
last_ts_ms = 0
last_vol_time = 0.0
last_shot_time = 0.0
last_toggle_time = 0.0
last_mute_time = 0.0
banner_text = ""
banner_until = 0.0

try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam frame open nahi ho pa raha.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now_ms = int(time.time() * 1000)
            timestamp_ms = now_ms if now_ms > last_ts_ms else last_ts_ms + 1
            last_ts_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)

            mode = "IDLE"
            now = time.time()

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                draw_hand(frame, hand_lms, w, h)

                up = get_fingers_up(hand_lms)
                index_up, middle_up, ring_up, pinky_up = up
                index_y = hand_lms[8].y

                # 1. Closed fist -> Screenshot (2.5s cooldown)
                if not any(up):
                    mode = "SCREENSHOT"
                    if now - last_shot_time > 2.5:
                        last_shot_time = now
                        shot_path = SHOTS_DIR / f"airdesk_{int(now)}.png"
                        pyautogui.screenshot(str(shot_path))
                        banner_text = f"Saved: {shot_path.name}"
                        banner_until = now + 1.5

                # 2. Open palm (all 4 fingers up) -> Mute / Unmute (1.5s cooldown)
                elif all(up):
                    mode = "MUTE / UNMUTE"
                    if now - last_mute_time > 1.5:
                        last_mute_time = now
                        pyautogui.press("volumemute")
                        banner_text = "Toggled Mute / Unmute"
                        banner_until = now + 1.0

                # 3. Index + Pinky up -> Play / Pause video (1.2s cooldown)
                elif index_up and pinky_up and not middle_up and not ring_up:
                    mode = "PLAY / PAUSE"
                    if now - last_toggle_time > 1.2:
                        last_toggle_time = now
                        pyautogui.press("space")
                        banner_text = "Toggled Play/Pause (Space)"
                        banner_until = now + 1.0

                # 4. Index only up -> High = Volume Up, Low = Volume Down
                elif index_up and not middle_up and not ring_up and not pinky_up:
                    if index_y < 0.38:
                        mode = "VOLUME UP"
                        if now - last_vol_time > 0.12:
                            pyautogui.press("volumeup")
                            last_vol_time = now
                    elif index_y > 0.58:
                        mode = "VOLUME DOWN"
                        if now - last_vol_time > 0.12:
                            pyautogui.press("volumedown")
                            last_vol_time = now
                    else:
                        mode = "VOLUME (MID)"

                else:
                    mode = "READY"

            # Status HUD bar
            cv2.rectangle(frame, (0, 0), (w, 46), (25, 25, 30), -1)
            cv2.putText(
                frame,
                f"AirDesk | {mode}",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (90, 235, 135),
                2,
            )

            # Banner notification
            if now < banner_until and banner_text:
                cv2.rectangle(frame, (14, 58), (430, 96), (35, 130, 70), -1)
                cv2.putText(
                    frame,
                    banner_text,
                    (22, 83),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("AirDesk", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()