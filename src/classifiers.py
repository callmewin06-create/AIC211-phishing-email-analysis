"""Persistable local classifiers used by the phishing training pipeline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class HybridPhishingClassifier:
    """Probability ensemble for discriminative phrases and sparse messages.

    The class lives in an importable module (rather than ``train.py``) so
    joblib artifacts remain loadable whether training is started with
    ``python src/train.py`` or imported by the Streamlit application.
    """

    def __init__(self, logistic_weight: float = 0.90, nb_alpha: float = 0.35):
        self.logistic_weight = float(logistic_weight)
        self.nb_alpha = float(nb_alpha)

    def fit(self, matrix, labels, sample_weight=None):
        self.logistic_model = LogisticRegression(
            max_iter=1400, random_state=42, class_weight="balanced", C=8.0, solver="liblinear"
        )
        self.nb_model = MultinomialNB(alpha=self.nb_alpha)
        self.logistic_model.fit(matrix, labels, sample_weight=sample_weight)
        self.nb_model.fit(matrix, labels, sample_weight=sample_weight)
        self.classes_ = self.logistic_model.classes_
        # Compatibility contract for the existing word-level explanation UI.
        self.coef_ = self.logistic_model.coef_
        return self

    def predict_proba(self, matrix):
        logistic = self.logistic_model.predict_proba(matrix)
        naive_bayes = self.nb_model.predict_proba(matrix)
        return self.logistic_weight * logistic + (1.0 - self.logistic_weight) * naive_bayes
