"""
Type definitions for the LLM Forge system.

This module contains TypedDict and other type definitions used throughout
the LLM Forge system to ensure type safety and consistency.
"""

from typing import Dict, List, TypedDict


class StructuredInput(TypedDict):
    """
    Structured representation of a parsed user prompt.

    Attributes:
        models: List of model names to generate responses for
        sections: List of section names to include in responses
        raw_prompt: The original unmodified user prompt
        topic: The main subject extracted from the prompt
    """

    models: List[str]
    sections: List[str]
    raw_prompt: str
    topic: str


class ModelResponse(TypedDict):
    """
    Structured representation of generated model responses.

    Attributes:
        topic: The main subject of the response
        models: Dictionary mapping model names to their section responses
    """

    topic: str
    models: Dict[str, Dict[str, str]]
