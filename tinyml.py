"""TinyML step: shrink the letter model until it fits on a microcontroller.

What this script does
  1. Trains a tiny neural network (63 inputs -> 32 hidden -> letters) on data/dataset.csv
  2. Quantizes it to int8 (the number format microcontrollers use)
  3. Checks the int8 model is still accurate, using integer-only maths
  4. Compares size and accuracy with the big Random Forest model
  5. Reports how much memory it needs on real boards
  6. Exports tiny_model.h: C code you can paste into an Arduino / ESP32 project

Run:  python tinyml.py            (needs numpy, pandas, scikit-learn only)
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/dataset.csv")
parser.add_argument("--hidden", type=int, default=32, help="hidden neurons (smaller = tinier)")
parser.add_argument("--out", default="tiny_model.h")
parser.add_argument("--report", default="tinyml_report.txt")
args = parser.parse_args()

lines = []


def say(text=""):
    print(text)
    lines.append(text)


# ---------------------------------------------------------------- data
df = pd.read_csv(args.data)
X = np.clip(df.drop(columns="label").values.astype(np.float32), -1.0, 1.0)
y = df["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------------- models
mlp = MLPClassifier(hidden_layer_sizes=(args.hidden,), activation="relu",
                    max_iter=600, random_state=42)
mlp.fit(X_tr, y_tr)
classes = list(mlp.classes_)
N_IN, N_HID, N_CLS = X.shape[1], args.hidden, len(classes)
y_te_idx = np.array([classes.index(v) for v in y_te])

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# ---------------------------------------------------------------- quantize to int8
S_X = 1.0 / 127.0                              # input scale (inputs are in -1..1)
W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
s_w1 = np.abs(W1).max() / 127.0
s_w2 = np.abs(W2).max() / 127.0
W1q = np.clip(np.round(W1 / s_w1), -127, 127).astype(np.int8)
W2q = np.clip(np.round(W2 / s_w2), -127, 127).astype(np.int8)
B1 = np.round(b1 / (S_X * s_w1)).astype(np.int32)


def quant_input(x):
    """float -1..1  ->  int8 -127..127 (round half away from zero, like the C code)"""
    v = x * 127.0
    return np.clip(np.trunc(v + np.where(v >= 0, 0.5, -0.5)), -127, 127).astype(np.int8)


# calibrate the hidden-layer scale on the training data
acc1_cal = quant_input(X_tr).astype(np.int32) @ W1q.astype(np.int32) + B1
s_h = max(float(np.maximum(acc1_cal, 0).max()) * S_X * s_w1, 1e-9) / 127.0
L1_MULT = np.float32((S_X * s_w1) / s_h)
B2 = np.round(b2 / (s_h * s_w2)).astype(np.int32)


def int8_predict(x):
    """Integer-only inference: exactly what the microcontroller does."""
    acc1 = quant_input(x).astype(np.int32) @ W1q.astype(np.int32) + B1
    acc1 = np.maximum(acc1, 0)
    h = np.minimum(np.floor(acc1.astype(np.float32) * L1_MULT + np.float32(0.5)), 127)
    acc2 = h.astype(np.int32) @ W2q.astype(np.int32) + B2
    return np.argmax(acc2, axis=1)


# ---------------------------------------------------------------- results
float_pred = np.array([classes.index(v) for v in mlp.predict(X_te)])
int8_pred = int8_predict(X_te)

acc_rf = rf.score(X_te, y_te) * 100
acc_float = (float_pred == y_te_idx).mean() * 100
acc_int8 = (int8_pred == y_te_idx).mean() * 100
agree = (float_pred == int8_pred).mean() * 100

weights_bytes = W1q.size + W2q.size
bias_bytes = 4 * (B1.size + B2.size)
flash_bytes = weights_bytes + bias_bytes
ram_bytes = N_IN + N_HID                       # int8 input + int8 hidden activations
macs = N_IN * N_HID + N_HID * N_CLS
rf_bytes = os.path.getsize("model.pkl") if os.path.exists("model.pkl") else len(pickle.dumps(rf))

say("TINYML REPORT")
say("=" * 60)
say(f"Samples: {len(df)}  |  Letters/classes: {N_CLS}  |  Test samples: {len(y_te)}")
say(f"Tiny network: {N_IN} inputs -> {N_HID} hidden (ReLU) -> {N_CLS} outputs")
say(f"Parameters: {W1.size + W2.size + b1.size + b2.size}")
say()
say("ACCURACY (same test split)")
say(f"  Random Forest (the laptop model) ........ {acc_rf:6.2f} %")
say(f"  Tiny network, float32 ................... {acc_float:6.2f} %")
say(f"  Tiny network, int8 (microcontroller) .... {acc_int8:6.2f} %")
say(f"  int8 agrees with float32 on ............. {agree:6.2f} % of samples")
say()
say("SIZE")
say(f"  Random Forest model.pkl ................. {rf_bytes / 1024:9.1f} KB")
say(f"  Tiny int8 model (flash) ................. {flash_bytes / 1024:9.2f} KB"
    f"   ({weights_bytes} B weights + {bias_bytes} B biases)")
say(f"  Working memory (RAM, activations) ....... {ram_bytes:9d} B")
say(f"  Smaller than the Random Forest by ....... {rf_bytes / flash_bytes:9.0f} x")
say()
say("WILL IT FIT? (model only, without your program code)")
boards = [("Arduino Uno R3", 32 * 1024, 2 * 1024),
          ("Arduino Nano 33 BLE", 1024 * 1024, 256 * 1024),
          ("ESP32 (typical)", 4 * 1024 * 1024, 520 * 1024)]
for name, flash, ram in boards:
    say(f"  {name:<22} flash {100 * flash_bytes / flash:6.2f} %   RAM {100 * ram_bytes / ram:6.2f} %")
say()
say(f"SPEED: {macs} multiply-adds per prediction")
say(f"  rough estimate: {macs * 5 / 64e6 * 1000:.2f} ms on a 64 MHz Cortex-M4")
say("  (assumes ~5 clock cycles per multiply-add; an estimate, not a measurement)")

# ---------------------------------------------------------------- export C header
def c_array(name, ctype, arr, per_line=12):
    flat = arr.flatten()
    rows = []
    for i in range(0, len(flat), per_line):
        rows.append("  " + ", ".join(str(int(v)) for v in flat[i:i + per_line]))
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


header = f"""// tiny_model.h - generated by tinyml.py. Do not edit by hand.
// Sign-language letter classifier: {N_IN} inputs -> {N_HID} hidden (ReLU) -> {N_CLS} classes, int8 weights.
//
// How to use on a board:
//   float features[{N_IN}];            // the same 63 numbers the Python code makes
//   int letter = tiny_predict(features);
//   const char *name = CLASS_NAMES[letter];
// On AVR boards (Arduino Uno) put the const arrays in PROGMEM to save RAM.
#pragma once
#include <stdint.h>

#define N_IN {N_IN}
#define N_HIDDEN {N_HID}
#define N_CLASSES {N_CLS}

static const float L1_MULT = {float(L1_MULT):.10e}f;

"""
header += "static const char *CLASS_NAMES[N_CLASSES] = {" + ", ".join(f'"{c}"' for c in classes) + "};\n\n"
header += c_array("W1", "int8_t", W1q) + "\n"
header += c_array("B1", "int32_t", B1) + "\n"
header += c_array("W2", "int8_t", W2q) + "\n"
header += c_array("B2", "int32_t", B2) + "\n"
header += """
// Integer-only prediction. Returns the index of the most likely class.
static inline int tiny_predict(const float *x) {
  int8_t xq[N_IN];
  for (int i = 0; i < N_IN; i++) {
    float v = x[i] * 127.0f;
    v += (v >= 0.0f) ? 0.5f : -0.5f;
    if (v > 127.0f) v = 127.0f;
    if (v < -127.0f) v = -127.0f;
    xq[i] = (int8_t)v;
  }
  int8_t h[N_HIDDEN];
  for (int j = 0; j < N_HIDDEN; j++) {
    int32_t acc = B1[j];
    for (int i = 0; i < N_IN; i++) acc += (int32_t)xq[i] * W1[i * N_HIDDEN + j];
    if (acc < 0) acc = 0;
    float r = (float)acc * L1_MULT;
    if (r > 127.0f) r = 127.0f;
    h[j] = (int8_t)(r + 0.5f);
  }
  int best = 0;
  int32_t best_acc = 0;
  for (int k = 0; k < N_CLASSES; k++) {
    int32_t acc = B2[k];
    for (int j = 0; j < N_HIDDEN; j++) acc += (int32_t)h[j] * W2[j * N_CLASSES + k];
    if (k == 0 || acc > best_acc) { best_acc = acc; best = k; }
  }
  return best;
}
"""
with open(args.out, "w") as f:
    f.write(header)
say()
say(f"Saved {args.out}  (C code for the microcontroller)")
with open(args.report, "w") as f:
    f.write("\n".join(lines) + "\n")
say(f"Saved {args.report}  (copy these numbers into your project report)")