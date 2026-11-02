"""
Step 1: Collect hand-sign examples using MediaPipe Tasks API.
Press A-Z to save the current hand sign.
Press ESC to stop.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
CSV_FILE = "hand_data.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Download hand_landmarker.task and place it inside the 'models' folder."
    )

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"x{i}" for i in range(21)] +
                        [f"y{i}" for i in range(21)])

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)
frame_timestamp_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp_ms += 33  # approximately 30 FPS
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        hand_detected = False

        if result.hand_landmarks:
            hand_detected = True
            landmarks = result.hand_landmarks[0]

            # Draw simple hand connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            h, w, _ = frame.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for start, end in connections:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

            for point in points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)

            cv2.putText(
                frame,
                "Press A-Z to save this sign",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Collect Sign Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if 65 <= key <= 90 and hand_detected:
            label = chr(key)
            row = [label]
            for landmark in result.hand_landmarks[0]:
                row.extend([landmark.x, landmark.y])

            with open(CSV_FILE, "a", newline="") as file:
                csv.writer(file).writerow(row)

            print(f"Saved example for letter: {label}")

camera.release()
cv2.destroyAllWindows()
