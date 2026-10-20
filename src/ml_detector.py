"""Base text detectors used by the hybrid phishing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


SEED = 20260715


def make_vectorizer(kind: str = "word"):
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if kind == "word":
        return word
    if kind == "word_char":
        character = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30_000,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        return FeatureUnion((("word", word), ("char", character)))
    raise ValueError(f"Unsupported vectorizer kind: {kind}")


def fit_ml_detector(texts, labels, kind: str = "word"):
    vectorizer = make_vectorizer(kind)
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=1_200,
        solver="liblinear",
        random_state=SEED,
    )
    model.fit(matrix, labels)
    return vectorizer, model


def predict_ml_probability(vectorizer, model, texts) -> np.ndarray:
    return np.asarray(model.predict_proba(vectorizer.transform(texts))[:, 1], dtype=float)
