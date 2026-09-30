"""
Spam Email Classifier
=====================
Trains and compares two classic text classifiers for spam detection:

  * Multinomial Naive Bayes
  * Linear Support Vector Machine (SVM)

Pipeline: raw text -> TF-IDF features (unigrams + bigrams) -> classifier.

Author: Nithish.S
"""

import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Paths are resolved relative to this file so the project runs from any folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.joblib")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Light normalisation: lowercase, mask URLs / numbers, drop punctuation."""
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV (columns: label, text), clean it and add a 0/1 target."""
    with open(path, "r", encoding="utf-8") as f:
        df = pd.read_csv("C:\\CODEC_Spam Email Classifier\\spam.csv")
    df = df.dropna(subset=["label", "text"]).drop_duplicates().reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["target"] = (df["label"].str.lower() == "spam").astype(int)
    return df


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def build_pipeline(model_name: str) -> Pipeline:
    """Return a TF-IDF + classifier pipeline for 'naive_bayes' or 'svm'."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    if model_name == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif model_name == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced")
    else:
        raise ValueError("model_name must be 'naive_bayes' or 'svm'")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Compute the standard classification metrics for the spam class."""
    preds = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds, target_names=["ham", "spam"]),
    }


def train_and_compare(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Train both models on the same split and return (results, best_name, pipelines)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["target"],
        test_size=test_size, random_state=seed, stratify=df["target"],
    )

    results, pipelines = {}, {}
    for name in ("naive_bayes", "svm"):
        pipe = build_pipeline(name)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        results[name] = evaluate(pipe, X_test, y_test)

    best = max(results, key=lambda n: results[n]["f1"])
    return results, best, pipelines


# ----------------------------------------------------------------------
# Save / load / predict
# ----------------------------------------------------------------------
def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model found. Run `python train.py` first.")
    return joblib.load(path)


def predict(pipeline: Pipeline, message: str) -> str:
    """Classify one message as 'spam' or 'ham' (not spam)."""
    return "spam" if pipeline.predict([clean_text(message)])[0] == 1 else "ham"
