import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from signmatic import pool_sequences

DATASET_DIR = os.path.join(
    "data", "signmatic", "SignMatic_ASL_Keypoint_Dataset_Kaggle"
)


def load_dataset(dataset_dir):
    data_dir = os.path.join(dataset_dir, "data")
    summary = pd.read_csv(os.path.join(dataset_dir, "class_summary.csv"))
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    for row in summary.itertuples(index=False):
        label = row.label
        path = os.path.join(data_dir, f"{str(label).lower()}.npz")
        with np.load(path, allow_pickle=False) as shard:
            real = shard["real"]
            synthetic = shard["synthetic"]

        if real.ndim != 3 or real.shape[1:] != (30, 258):
            raise ValueError(f"{path} has unexpected real sequence shape {real.shape}")
        if synthetic.ndim != 3 or synthetic.shape[1:] != (30, 258):
            raise ValueError(
                f"{path} has unexpected synthetic sequence shape {synthetic.shape}"
            )

        real_train, real_test = train_test_split(
            real, test_size=0.2, random_state=42
        )
        train_features.extend((real_train, synthetic))
        train_labels.extend(
            (
                np.full(len(real_train), label),
                np.full(len(synthetic), label),
            )
        )
        test_features.append(real_test)
        test_labels.append(np.full(len(real_test), label))

    x_train = pool_sequences(np.concatenate(train_features))
    y_train = np.concatenate(train_labels)
    x_test = pool_sequences(np.concatenate(test_features))
    y_test = np.concatenate(test_labels)
    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Train a word classifier from the downloaded SignMatic dataset."
    )
    parser.add_argument("--data-dir", default=DATASET_DIR)
    parser.add_argument("--out", default="signmatic_model.pkl")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_dataset(args.data_dir)
    print(f"Training sequences: {len(y_train):,}")
    print(f"Test sequences (real only): {len(y_test):,}")
    print("Training Extra Trees classifier...")

    model = ExtraTreesClassifier(
        n_estimators=120,
        max_depth=24,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"Test accuracy (random sample split): {accuracy:.1%}")

    joblib.dump(model, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
