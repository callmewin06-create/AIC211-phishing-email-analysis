"""Faithful sparse linear feature contributions for the analyst UI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def linear_feature_contributions(vectorizer, model, document: str, top_k: int = 15) -> pd.DataFrame:
    """Return non-zero TF-IDF x coefficient contributions.

    Both unigrams and bigrams are retained.  FeatureUnion prefixes are also
    preserved so the UI does not pretend character n-grams are raw words.
    """

    estimator = getattr(model, "logistic_model", model)
    if not hasattr(estimator, "coef_"):
        raise TypeError("Feature contributions require a linear model with coef_.")
    matrix = vectorizer.transform([str(document or "")]).tocsr()
    feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    weights = np.asarray(estimator.coef_[0], dtype=float)
    indices = matrix.indices
    values = matrix.data.astype(float)
    contributions = values * weights[indices]
    frame = pd.DataFrame(
        {
            "feature": feature_names[indices],
            "tfidf": values,
            "coefficient": weights[indices],
            "contribution": contributions,
        }
    )
    if frame.empty:
        return frame.assign(direction=pd.Series(dtype=str))
    frame["direction"] = np.where(frame["contribution"] >= 0, "phishing", "safe")
    return (
        frame.assign(abs_contribution=frame["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(max(1, int(top_k)))
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
