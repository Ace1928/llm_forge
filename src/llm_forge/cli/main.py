#!/usr/bin/env python
"""
Command Line Interface for LLM Forge.

This module provides the main entry point for the LLM Forge CLI,
with commands for generating model comparisons and other utilities.
"""

import sys
from typing import Dict, Literal, Optional, Union, cast

import click
from click import Context

from llm_forge.comparison_generator import generate_comparison
from llm_forge.formatters.renderer import FormatType, render_output
from llm_forge.input_parser import parse_input
from llm_forge.logging_config import configure_logging

# Configure logging for CLI context
logger = configure_logging()

# Exit status code type for CLI operations
ExitCode = Union[Literal[0], Literal[1]]

__version__ = "0.1.0"
@click.group()
@click.version_option(str(__version__), prog_name="LLM Forge")
@click.pass_context
def cli(ctx: Context) -> None:
    """
    LLM Forge: Compare and analyze different language models.

    A tool for generating standardized comparisons between different
    LLM models based on natural language prompts.
    """
    # Initialize context for command
    ctx.ensure_object(dict)


@cli.command()
@click.argument("prompt", required=True)
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "html", "text"]),
    default="markdown",
    help="Output format for the comparison"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path (defaults to stdout)"
)
@click.option(
    "--models", "-m",
    multiple=True,
    help="Models to compare (defaults to automatic detection)"
)
def compare(
    prompt: str,
    format: str,
    output: Optional[str] = None,
    models: Optional[tuple[str, ...]] = None
) -> ExitCode:
    """
    Generate a comparison between different LLMs for a given prompt.

    The comparison analyzes how different models respond to the same prompt,
    highlighting differences in their approaches, capabilities, and limitations.

    Args:
        prompt: The natural language prompt to analyze
        format: Output format (markdown, html, or text)
        output: Path to save output (default: print to console)
        models: Specific models to compare

    Returns:
        0 for success, 1 for error
    """
    try:
        # Parse the user input
        structured_input = parse_input(prompt)

        # Override models if specified in the command
        if models:
            structured_input["models"] = list(models)

        # Generate the comparison
        comparison_data = generate_comparison(structured_input)

        # Render the output in the requested format
        # Cast to FormatType to ensure type safety
        rendered_output = render_output(comparison_data, cast(FormatType, format))

        # Write to file or stdout
        if output:
            with open(output, "w") as f:
                f.write(rendered_output)
        else:
            print(rendered_output)

        return 0
    except Exception as e:
        logger.error(f"Error generating comparison: {e}")
        return 1


@cli.command()
@click.option(
    "--list", "list_models",
    is_flag=True,
    help="List all available models"
)
def models(list_models: bool) -> ExitCode:
    """
    Manage and view available LLM models.

    View information about the models supported by LLM Forge,
    including their capabilities, strengths, and limitations.

    Args:
        list_models: Flag to list all available models

    Returns:
        0 for success
    """
    if list_models:
        # This would normally fetch from a registry or configuration
        available_models: Dict[str, str] = {
            "gpt-4": "OpenAI's most advanced model",
            "claude-2": "Anthropic's constitutional AI",
            "llama-2": "Meta's open model",
            "mistral": "Mistral AI's efficient model",
        }

        print("Available models:")
        for model, description in available_models.items():
            print(f"- {model}: {description}")
    else:
        print("Use --list to see available models")

    return 0


def main() -> ExitCode:
    """
    Entry point for the CLI application.

    Executes the CLI command group and handles any unexpected exceptions
    that weren't caught by individual commands.

    Returns:
        0 for success or handled errors, representing a clean exit
    """
    try:
        cli()  # pylint: disable=no-value-for-parameter
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
