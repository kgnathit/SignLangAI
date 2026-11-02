"""
Mode 1: Show a hand sign to the webcam and identify the character.

Controls:
  C - Clear detected word
  SPACE - Add a space
  ESC - Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
SIGN_MODEL = "sign_model.pkl"
LABEL_FILE = "label_map.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Download hand_landmarker.task and place it inside the 'models' folder."
    )

if not os.path.exists(SIGN_MODEL):
    raise FileNotFoundError(
        "No trained model found. First run collect_data.py and train_model.py."
    )

model = joblib.load(SIGN_MODEL)

with open(LABEL_FILE) as file:
    label_map = {int(k): v for k, v in json.load(file).items()}

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

predicted_word = ""
last_letter = ""
stable_count = 0

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

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

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        current_prediction = ""

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape

            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for start, end in connections:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

            for point in points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)

            hand_points = []
            for landmark in landmarks:
                hand_points.extend([landmark.x, landmark.y])

            hand_points = np.array(hand_points).reshape(1, -1)

            prediction_index = model.predict(hand_points)[0]
            confidence = max(model.predict_proba(hand_points)[0]) * 100
            current_prediction = label_map[prediction_index]

            if current_prediction == last_letter:
                stable_count += 1
            else:
                stable_count = 0
                last_letter = current_prediction

            if stable_count == 12:
                predicted_word += current_prediction
                stable_count = 0

            cv2.putText(
                frame,
                f"Detected: {current_prediction} ({confidence:.1f}%)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            last_letter = ""
            stable_count = 0

        cv2.putText(
            frame,
            f"Word: {predicted_word}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "C: Clear | SPACE: Add space | ESC: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Language Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord("c"):
            predicted_word = ""
        elif key == 32:
            predicted_word += " "

camera.release()
cv2.destroyAllWindows()
