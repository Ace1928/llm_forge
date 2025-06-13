"""NLTK utilities for Eidosian Forge.

This module provides helper functions to manage NLTK resources used
throughout the project. It currently ensures that the 'punkt'
tokenizer required for sentence splitting is available.
"""

from typing import Final, List

import nltk  # type: ignore

from llm_forge.logging_config import configure_logging

# Module-level logger
logger: Final = configure_logging()


def ensure_punkt() -> None:
    """Ensure that the NLTK 'punkt' tokenizer is installed."""
    try:
        nltk.data.find("tokenizers/punkt")  # type: ignore
        logger.debug("NLTK 'punkt' tokenizer already available")
    except LookupError:
        logger.info("Downloading NLTK 'punkt' tokenizer")
        nltk.download("punkt")  # type: ignore


__all__: Final[List[str]] = ["ensure_punkt"]
