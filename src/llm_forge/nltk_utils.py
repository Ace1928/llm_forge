"""Utility functions related to NLTK resources."""

from __future__ import annotations

import nltk  # type: ignore


def ensure_punkt() -> None:
    """Ensure that the NLTK 'punkt' tokenizer models are downloaded.

    This function checks whether the 'punkt' tokenizer data is available. If the
    resource is missing, it triggers a download using :func:`nltk.download`.
    The check is idempotent to avoid repeated downloads during subsequent
    imports.
    """
    try:
        nltk.data.find("tokenizers/punkt")  # type: ignore[attr-defined]
    except LookupError:
        nltk.download("punkt")  # type: ignore[attr-defined]


__all__ = ["ensure_punkt"]
