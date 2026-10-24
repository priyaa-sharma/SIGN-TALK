"""Step 1: record training samples from your webcam.

Usage:  python collect_data.py --labels A B C D E HELLO THANKS SPACE DEL

Keys:   S = start/stop recording the current label
        N = next label      P = previous label
        Q = quit
Tip: record 200-300 samples per sign; move your hand a little (angle,
distance, position) while recording so the model generalises.
"""
import argparse
import csv
import os

import cv2
from utils import HandTracker, extract_features, draw_hand

parser = argparse.ArgumentParser()
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--out", default="data/dataset.csv")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
new_file = not os.path.exists(args.out)
f = open(args.out, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow(["label"] + [f"f{i}" for i in range(63)])

tracker = HandTracker()
cap = cv2.VideoCapture(0)
idx, recording, counts = 0, False, {l: 0 for l in args.labels}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    landmarks = tracker.find(frame)
    label = args.labels[idx]

    if landmarks:
        draw_hand(frame, landmarks)
        if recording:
            writer.writerow([label] + list(extract_features(landmarks)))
            counts[label] += 1

    status = "RECORDING" if recording else "paused"
    cv2.putText(frame, f"Label: {label}  ({counts[label]})  [{status}]",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if recording else (0, 200, 0), 2)
    cv2.imshow("Collect data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        recording = not recording
    elif key == ord("n"):
        idx, recording = (idx + 1) % len(args.labels), False
    elif key == ord("p"):
        idx, recording = (idx - 1) % len(args.labels), False

f.close()
cap.release()
cv2.destroyAllWindows()
print("Samples collected:", counts)
