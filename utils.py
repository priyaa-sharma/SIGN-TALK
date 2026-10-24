import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class HandTracker:
    """Finds one hand in a BGR webcam frame and returns its 21 landmarks."""

    def __init__(self):
        _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def find(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self.detector.detect_for_video(image, ts)
        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (203, 150, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 5, (147, 20, 255), -1)


def extract_features(landmarks):
    """Turn 21 hand landmarks into 63 position/size-independent features.

    1. Subtract the wrist point -> position independent.
    2. Divide by the largest absolute value -> hand-size/distance independent.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    pts = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()
