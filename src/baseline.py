"""Reusable baseline training for the group's standard train/dev experiment.

Models read text_clean and predict class_label. This module never loads test data.
The fitted pipeline includes TF-IDF so evaluation can reuse training vocabulary.
"""
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


def load_split(path):
    """Load a prepared split and fail early if text or labels are unusable."""
    frame = pd.read_csv(path, sep="\t", dtype={"id": "string"})
    for column in ("id", "text_clean", "class_label"):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"Missing values or column: {column}")
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Blank values in {column}")
    return frame


def build_pipeline(ngram_range=(1, 1), C=1.0, class_weight=None):
    """C controls regularization; balanced weights emphasize smaller classes."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range)),
        ("classifier", LogisticRegression(
            C=C, class_weight=class_weight, max_iter=1000, random_state=42,
        )),
    ])


def dev_metrics(truth, predictions):
    """Macro-F1 is the selection metric; other metrics describe tradeoffs."""
    return {
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(truth, predictions),
    }


def run_experiments(train, dev):
    """Compare 12 fixed settings using dev only; return a fitted best pipeline.

    TF-IDF is fitted once per n-gram setting on TRAIN ONLY and reused across
    classifiers. Dev is transformed, never fitted. Ties keep the first setting.
    Thread limits avoid oversubscribing laptop CPU cores.
    """
    for key in ("id", "text_clean"):
        if set(train[key]) & set(dev[key]):
            raise ValueError(f"Train/dev overlap in {key}")
    if set(dev["class_label"]) - set(train["class_label"]):
        raise ValueError("Dev contains labels not present in train")
    rows, best_model, best_score = [], None, -1.0
    with threadpool_limits(limits=1):
        for ngrams in [(1, 1), (1, 2)]:
            vectorizer = TfidfVectorizer(ngram_range=ngrams)
            X_train = vectorizer.fit_transform(train["text_clean"])
            X_dev = vectorizer.transform(dev["text_clean"])
            for C in [0.1, 1.0, 10.0]:
                for weight in [None, "balanced"]:
                    model = build_pipeline(ngrams, C, weight)
                    classifier = model.named_steps["classifier"]
                    # A failed convergence check must not silently select a model.
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        classifier.fit(X_train, train["class_label"])
                    predictions = classifier.predict(X_dev)
                    scores = dev_metrics(dev["class_label"], predictions)
                    row = {
                        "ngram_range": str(ngrams), "C": C,
                        "class_weight": weight or "none",
                        **{f"dev_{key}": value for key, value in scores.items()},
                        "iterations": int(classifier.n_iter_.max()),
                    }
                    rows.append(row)
                    print(f"{len(rows):2}/12 ngrams={ngrams}, C={C}, "
                          f"weights={weight}: dev macro-F1={scores['macro_f1']:.4f}", flush=True)
                    if scores["macro_f1"] > best_score:
                        best_score = scores["macro_f1"]
                        # Both components are already fitted. This pipeline can
                        # predict raw text_clean directly without fitting again.
                        model.steps[0] = ("tfidf", vectorizer)
                        best_model = model
    results = pd.DataFrame(rows).sort_values(
        "dev_macro_f1", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return results, best_model


def per_class_report(model, frame):
    """Return a report suitable for CSV export and notebook inspection."""
    predictions = model.predict(frame["text_clean"])
    report = classification_report(
        frame["class_label"], predictions,
        labels=model.classes_, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T


def save_pipeline(model, path):
    """Save the complete fitted pipeline, including training vocabulary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
