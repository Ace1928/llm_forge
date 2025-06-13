"""Utility helpers for Natural Language Toolkit (NLTK) assets."""

from typing import Final

import nltk  # type: ignore

from llm_forge.logging_config import configure_logging

logger: Final = configure_logging()


def ensure_punkt() -> None:
    """Ensure that the NLTK ``punkt`` tokenizer is available.

    The function attempts to locate the required tokenizer data and will
    download it using ``nltk.download`` if not already present. It is
    idempotent and safe to invoke multiple times.
    """
    try:
        nltk.data.find("tokenizers/punkt")  # type: ignore
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer data...")
        nltk.download("punkt")  # type: ignore
