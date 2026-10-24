"""Record movement words.

Usage:  python record_words.py --labels HELLO THANKYOU SORRY PLEASE ILOVEYOU

Keys:  R = record one clip (2 sec get-ready, then 2 sec of signing)
       N = next word     P = previous word     Q = quit
Record about 30 clips for each word.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
from utils import HandTracker, draw_hand
from words_common import (COUNTDOWN, CLIP_SECONDS, clip_to_sequence,
                          frame_features)

parser = argparse.ArgumentParser()
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--out", default="data/words")
args = parser.parse_args()


def count_clips(label):
    folder = os.path.join(args.out, label)
    return len(os.listdir(folder)) if os.path.isdir(folder) else 0


counts = {l: count_clips(l) for l in args.labels}
tracker = HandTracker()
cap = cv2.VideoCapture(0)
idx, state, t0, frames, note = 0, "idle", 0.0, [], ""

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    lm = tracker.find(frame)
    if lm:
        draw_hand(frame, lm)
    now = time.time()
    label = args.labels[idx]

    if state == "countdown":
        left = COUNTDOWN - (now - t0)
        note = f"Get ready... {math.ceil(max(left, 0))}"
        if left <= 0:
            state, t0, frames = "recording", now, []
    elif state == "recording":
        frames.append(frame_features(lm) if lm else None)
        note = "GO! sign now"
        if now - t0 >= CLIP_SECONDS:
            seq = clip_to_sequence(frames)
            if seq is None:
                note = "Hand not seen enough - not saved, try again"
            else:
                folder = os.path.join(args.out, label)
                os.makedirs(folder, exist_ok=True)
                np.save(os.path.join(folder, f"{int(now * 1000)}.npy"), seq)
                counts[label] += 1
                note = "Saved!"
            state = "idle"

    color = (0, 0, 255) if state == "recording" else (0, 200, 0)
    cv2.putText(frame, f"Word: {label}  (clips: {counts[label]})", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, note or "Press R to record a clip", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Record words", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r") and state == "idle":
        state, t0, note = "countdown", now, ""
    elif key == ord("n") and state == "idle":
        idx, note = (idx + 1) % len(args.labels), ""
    elif key == ord("p") and state == "idle":
        idx, note = (idx - 1) % len(args.labels), ""

cap.release()
cv2.destroyAllWindows()
print("Clips per word:", counts)
