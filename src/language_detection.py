"""Offline, checksum-verified general language identification support.

The bundled fastText LID-176 model assists English/Vietnamese routing only. It
is deliberately not a phishing classifier and performs no network activity at
runtime.  Rule-based Vietnamese evidence remains in ``preprocessing`` because
the general model is weak on unaccented Vietnamese and very short messages.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASTTEXT_MODEL = ROOT / "data" / "external" / "fasttext_lid" / "lid.176.ftz"
FASTTEXT_LID_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
FASTTEXT_LID_SOURCE = "fastText LID-176 (CC BY-SA 3.0)"
MAX_LANGUAGE_MODEL_CHARS = 12_000
MIN_LANGUAGE_MODEL_LETTERS = 24


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ascii_safe_model_path(path: Path, expected_sha256: str) -> Path:
    """Mirror the verified model when fastText cannot open a Unicode path.

    The Windows fastText 0.9.2 binding passes file names through a narrow C++
    string and fails on otherwise valid project paths containing Vietnamese
    characters.  The source is verified before copying, and the ASCII-path
    mirror is verified again before it is loaded.
    """

    if str(path).isascii():
        return path
    mirror_dir = Path(tempfile.gettempdir()) / "aic211-fasttext-models"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror = mirror_dir / f"lid.176-{expected_sha256[:16]}.ftz"
    if not mirror.is_file() or _sha256_file(mirror).casefold() != expected_sha256.casefold():
        shutil.copyfile(path, mirror)
    if _sha256_file(mirror).casefold() != expected_sha256.casefold():
        raise ValueError("ASCII-path language model mirror failed checksum verification")
    return mirror


@lru_cache(maxsize=2)
def _load_fasttext_model(path_value: str, expected_sha256: str) -> tuple[Any | None, str | None]:
    path = Path(path_value)
    if not path.is_file():
        return None, "language model file is absent"
    if _sha256_file(path).casefold() != expected_sha256.casefold():
        return None, "language model checksum does not match pinned provenance"
    try:
        import fasttext

        load_path = _ascii_safe_model_path(path, expected_sha256)
        return fasttext.load_model(str(load_path)), None
    except Exception as error:  # pragma: no cover - depends on local binary loader.
        return None, f"language model could not be loaded: {error}"


def fasttext_language_evidence(
    text: object,
    *,
    model_path: str | Path = DEFAULT_FASTTEXT_MODEL,
    expected_sha256: str = FASTTEXT_LID_SHA256,
) -> dict[str, Any]:
    """Return top general-language predictions without contacting any service."""

    value = re.sub(r"\s+", " ", str(text or "")).strip()[:MAX_LANGUAGE_MODEL_CHARS]
    letter_count = sum(character.isalpha() for character in value)
    if letter_count < MIN_LANGUAGE_MODEL_LETTERS:
        return {
            "state": "NOT_APPLICABLE",
            "source": FASTTEXT_LID_SOURCE,
            "top_language": None,
            "top_probability": None,
            "predictions": [],
            "detail": "too little natural-language text for model routing",
        }
    model, error = _load_fasttext_model(str(Path(model_path).resolve()), expected_sha256)
    if model is None:
        return {
            "state": "UNAVAILABLE",
            "source": FASTTEXT_LID_SOURCE,
            "top_language": None,
            "top_probability": None,
            "predictions": [],
            "detail": error,
        }
    try:
        # fasttext-wheel 0.9.2 is otherwise compatible with CPython 3.12 but
        # its public ``predict`` wrapper calls ``np.array(..., copy=False)``,
        # which NumPy 2 rejects.  The compiled API returns the same immutable
        # score/label tuples and avoids that wrapper-only compatibility bug.
        raw_predictions = model.f.predict(value + "\n", 3, 0.0, "strict")
    except Exception as prediction_error:
        return {
            "state": "UNAVAILABLE",
            "source": FASTTEXT_LID_SOURCE,
            "top_language": None,
            "top_probability": None,
            "predictions": [],
            "detail": f"language model prediction failed: {prediction_error}",
        }
    predictions = [
        {"language": str(label).removeprefix("__label__"), "probability": round(float(probability), 6)}
        for probability, label in raw_predictions
    ]
    top = predictions[0] if predictions else {"language": None, "probability": None}
    return {
        "state": "AVAILABLE",
        "source": FASTTEXT_LID_SOURCE,
        "top_language": top["language"],
        "top_probability": top["probability"],
        "predictions": predictions,
        "detail": "offline general-language routing evidence; not a phishing decision",
    }
