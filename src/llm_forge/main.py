"""
Main entry point for the LLM Forge prototype.

This module provides a demonstration of the recursive LLM reasoning framework
by processing a sample prompt and generating a structured response. It serves
as the primary interface for running the LLM Forge system from the command line.
"""

import json
from typing import Final

from llm_forge.logging_config import configure_logging
from llm_forge.response_loop import process_user_prompt
from llm_forge.type_definitions import ModelResponse

# Configure logging with module context
logger: Final = configure_logging()


def main() -> None:
    """
    Execute the main program workflow for the LLM Forge system.

    Orchestrates the entire process from parsing an example prompt to
    generating structured content, refining it recursively, and outputting
    the final comprehensive response with multiple models and sections.

    This function demonstrates the full pipeline in action using an example
    prompt comparing different LLM architectures.

    Returns:
        None: Results are printed to standard output

    Examples:
        >>> main()
        # Outputs a structured JSON response to stdout comparing LLM architectures
    """
    logger.info("Program started.")

    # Example prompt comparing LLM architectures
    user_prompt: Final[str] = (
        "Compare three different LLM architectures: GPT, Claude, and Mistral. Provide details on:\n"
        "1. Model architecture\n"
        "2. Training data and methodology\n"
        "3. Strengths & Weaknesses\n"
        "4. Real-world use cases"
    )

    # Process the prompt through the full pipeline
    final_response: ModelResponse = process_user_prompt(user_prompt)

    # Output the fully processed result
    logger.info("Program finished. Final response:")
    print(json.dumps(final_response, indent=2))


if __name__ == "__main__":
    main()
