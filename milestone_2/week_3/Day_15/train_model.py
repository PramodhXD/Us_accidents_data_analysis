
# train_model.py
# Trains a simple classifier on the Iris dataset and saves the model artifacts.
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, multi_class="auto", random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.3f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (ARTIFACT_DIR / "target_names.json").write_text(json.dumps(target_names, indent=2))

    print(f"Saved model and metadata under: {ARTIFACT_DIR.resolve()}")

if __name__ == "__main__":
    main()
