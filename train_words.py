import glob
import os

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from words_common import sequence_to_vector

X, y = [], []
for folder in sorted(glob.glob("data/words/*")):
    label = os.path.basename(folder)
    for path in glob.glob(os.path.join(folder, "*.npy")):
        X.append(sequence_to_vector(np.load(path)))
        y.append(label)

X, y = np.array(X), np.array(y)
labels, counts = np.unique(y, return_counts=True)
print("Clips per word:", dict(zip(labels, counts)), "\n")
if len(labels) < 2 or counts.min() < 5:
    raise SystemExit("Record at least 2 words with 5+ clips each first "
                     "(about 30 clips per word is best).")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("Test accuracy:", round((pred == y_te).mean() * 100, 2), "%\n")
print(classification_report(y_te, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_te, pred, labels=model.classes_))

joblib.dump(model, "words_model.pkl")
print("\nSaved words_model.pkl")
