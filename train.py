import os
import json
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "data/resume_dataset.csv"
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
METRICS_PATH = "model/metrics.json"


def clean_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+.#\\s-]", " ", text)
    return text.strip()


def load_dataset():
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in ["Category", "Resume"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["Category", "Resume"])
    df["Resume"] = df["Resume"].astype(str).map(clean_text)
    df["Category"] = df["Category"].astype(str)
    return df


def main():
    df = load_dataset()
    X = df["Resume"]
    y_raw = df["Category"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy:  {acc:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    metrics = {
        "accuracy": acc,
        "labels": label_encoder.classes_.tolist(),
        "confusion_matrix": cm,
        "report": report,
        "vectorizer": {
            "max_features": 8000,
            "ngram_range": [1, 2],
            "min_df": 2,
        },
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 2000,
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model, vectorizer, label encoder, and metrics.")


if __name__ == "__main__":
    main()
