import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from utils import MODEL_PATH as HAND_MODEL_PATH, ensure_hand_model

POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 258
TEMPORAL_BINS = 10


def pool_sequences(sequences):
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3 or sequences.shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    ):
        raise ValueError(
            f"Expected sequences shaped (n, {SEQUENCE_LENGTH}, "
            f"{FEATURE_COUNT}), got {sequences.shape}"
        )
    pooled = sequences.reshape(
        len(sequences), TEMPORAL_BINS, SEQUENCE_LENGTH // TEMPORAL_BINS,
        FEATURE_COUNT
    ).mean(axis=2)
    return pooled.reshape(len(pooled), -1)


class SignMaticTracker:
    def __init__(self):
        ensure_hand_model()
        if not os.path.exists(POSE_MODEL_PATH):
            print("Downloading MediaPipe pose landmark model...")
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = 0

    def close(self):
        self.detector.close()
        self.hand_detector.close()

    def frame_features(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = timestamp
        hand_result = self.hand_detector.detect_for_video(image, timestamp)
        result = self.detector.detect_for_video(image, timestamp)

        features = np.zeros(FEATURE_COUNT, dtype=np.float32)
        if result.pose_landmarks:
            for index, landmark in enumerate(result.pose_landmarks[0]):
                start = index * 4
                features[start:start + 4] = (
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", 0.0),
                )

        for landmarks, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            if not handedness:
                continue
            side = handedness[0].category_name
            start = 132 if side == "Left" else 195 if side == "Right" else None
            if start is None:
                continue
            values = np.asarray(
                [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
                dtype=np.float32,
            ).flatten()
            features[start:start + 63] = values

        return features
