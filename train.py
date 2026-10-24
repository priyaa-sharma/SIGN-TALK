import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")
X, y = df.drop(columns="label").values, df["label"].values
print("Samples per class:\n", df["label"].value_counts(), "\n")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("Test accuracy:", round((pred == y_te).mean() * 100, 2), "%\n")
print(classification_report(y_te, pred))
print("Confusion matrix:\n", confusion_matrix(y_te, pred, labels=model.classes_))

joblib.dump(model, "model.pkl")
print("\nSaved model.pkl")
