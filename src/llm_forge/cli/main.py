#!/usr/bin/env python
"""
Command-line interface for the LLM Forge system.

This module provides a command-line entry point for using the LLM Forge
system to generate structured responses from natural language prompts.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from llm_forge import __version__
from llm_forge.logging_config import configure_logging
from llm_forge.response_loop import process_user_prompt
from llm_forge.type_definitions import ModelResponse

# Configure logging
logger = configure_logging()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """
    LLM Forge: A recursive framework for language model orchestration.

    Generate structured, multi-model responses to natural language prompts
    with automatic refinement and validation.
    """
    pass


@cli.command()
@click.argument("prompt", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save response to specified file path (JSON format)",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Format output JSON with indentation (pretty) or compact",
)
def generate(prompt: str, output: Optional[str] = None, pretty: bool = True) -> None:
    """
    Generate a structured response to a natural language prompt.

    The system will analyze the prompt, extract relevant models and sections,
    and generate content for each combination. The response is output as JSON.

    PROMPT is the natural language query to process.
    """
    try:
        logger.info(f"Processing prompt: {prompt}")

        # Process the prompt to generate a response
        response: ModelResponse = process_user_prompt(prompt)

        # Format the response as JSON
        indent = 2 if pretty else None
        result = json.dumps(response, indent=indent)

        # Output the result
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result, encoding="utf-8")
            logger.info(f"Response saved to: {output_path}")
        else:
            print(result)

        return 0

    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.option(
    "--example", "-e", is_flag=True, help="Show an example prompt and response"
)
def demo(example: bool = False) -> None:
    """
    Run a demonstration of the LLM Forge system.

    Processes a pre-defined example prompt and displays the generated response.
    Use this to see the system's capabilities without writing your own prompt.
    """
    # Example prompt comparing LLM architectures
    example_prompt = (
        "Compare three different LLM architectures: GPT, Claude, and Mistral. Provide details on:\n"
        "1. Model architecture\n"
        "2. Training data and methodology\n"
        "3. Strengths & Weaknesses\n"
        "4. Real-world use cases"
    )

    if example:
        click.echo("Example prompt:")
        click.echo(f"\n{example_prompt}\n")
        return 0

    click.echo("Running LLM Forge demonstration...")
    click.echo(f"Prompt: {example_prompt}\n")

    # Process the prompt
    response = process_user_prompt(example_prompt)

    # Output the response
    click.echo("Generated Response:")
    click.echo(json.dumps(response, indent=2))

    return 0


def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        Integer exit code (0 for success, non-zero for errors)
    """
    try:
        return cli() or 0
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
