#!/usr/bin/env python
"""
Recursive LLM Reasoning Framework with Self-Correction.

This module implements a multi-component system for generating structured
AI responses with recursive refinement. It parses user prompts into structured
requests, generates content for each section, and applies recursive correction
to enhance output quality.
"""

import json

# Ensure NLTK 'punkt' tokenizer is available
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from llm_forge_proto.content_generator import (
    ensure_complete_response,
    generate_response,
)
from llm_forge_proto.input_parser import parse_input
from llm_forge_proto.logging_config import configure_logging
from llm_forge_proto.type_definitions import ModelResponse, StructuredInput

# Configure logging
logger = configure_logging()


def process_user_prompt(user_prompt: str) -> ModelResponse:
    """Process a user prompt into a structured response.

    This is the main orchestration function that coordinates the entire process
    from parsing the prompt to generating and finalizing the response.

    Args:
        user_prompt: The raw user input text to process

    Returns:
        A structured response with content for each model and section

    Raises:
        ValueError: If the prompt cannot be properly parsed
    """
    # Step 1: Parse the prompt
    logger.info("Parsing user prompt...")
    structured_input: StructuredInput = parse_input(user_prompt)
    logger.debug("Structured Input:\n" + json.dumps(structured_input, indent=2))

    # Step 2: Generate an initial response (with refinement & template formatting)
    logger.info("Generating initial AI response...")
    initial_response: ModelResponse = generate_response(structured_input)
    logger.debug("Initial Response:\n" + json.dumps(initial_response, indent=2))

    # Step 3: Recursively check and fill in any missing sections
    logger.info("Validating and completing AI response...")
    final_response: ModelResponse = ensure_complete_response(
        initial_response, structured_input
    )
    logger.debug("Final Completed Response:\n" + json.dumps(final_response, indent=2))

    return final_response


def main() -> None:
    """Execute the main program workflow.

    Orchestrates the entire process from parsing an example prompt to
    generating structured content, refining it recursively, and outputting
    the final comprehensive response with multiple models and sections.

    This function demonstrates the full pipeline in action with an example
    prompt comparing different LLM architectures.

    Returns:
        None: Results are printed to standard output
    """
    logger.info("Program started.")

    # Example user prompt
    user_prompt: str = (
        "Compare three different LLM architectures: GPT, Claude, and Mistral. Provide details on:\n"
        "1. Model architecture\n"
        "2. Training data and methodology\n"
        "3. Strengths & Weaknesses\n"
        "4. Real-world use cases"
    )

    # Process the prompt
    final_response: ModelResponse = process_user_prompt(user_prompt)

    # Output the result
    logger.info("Program finished. Final response:")
    print(json.dumps(final_response, indent=2))


if __name__ == "__main__":
    main()
