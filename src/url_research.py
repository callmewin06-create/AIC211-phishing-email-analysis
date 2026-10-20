"""Compatibility facade for the verified PhishVN URL-IOC benchmark.

Older notebooks imported this module before the source-integrity, pinned-PSL,
and artifact-trust work was added.  Keep the import path, but route all public
operations to the v2 implementation so it cannot accidentally emit a weaker
research artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .phishvn_url_benchmark import (
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PHISHVN_ROOT,
        load_phishvn_core,
        main,
        train_phishvn_url_ioc,
    )
    from .url_ioc_model import URL_FEATURE_COLUMNS as _URL_FEATURE_COLUMNS
    from .url_ioc_model import extract_url_features as _extract_url_features
except ImportError:  # pragma: no cover - direct script compatibility
    from phishvn_url_benchmark import (  # type: ignore[import-not-found]
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PHISHVN_ROOT,
        load_phishvn_core,
        main,
        train_phishvn_url_ioc,
    )
    from url_ioc_model import URL_FEATURE_COLUMNS as _URL_FEATURE_COLUMNS  # type: ignore[import-not-found]
    from url_ioc_model import extract_url_features as _extract_url_features  # type: ignore[import-not-found]


URL_FEATURE_COLUMNS = _URL_FEATURE_COLUMNS
extract_url_features = _extract_url_features


def load_phishvn_splits(dataset_root: str | Path = DEFAULT_PHISHVN_ROOT) -> dict[str, pd.DataFrame]:
    """Return the source-verified official gold/silver URL partitions."""

    return load_phishvn_core(dataset_root)[0]


def train_phishvn_url_detector(
    dataset_root: str | Path = DEFAULT_PHISHVN_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
):
    """Compatibility name for the non-promoted URL-IOC benchmark runner."""

    return train_phishvn_url_ioc(dataset_root, output_root)


if __name__ == "__main__":
    main()
