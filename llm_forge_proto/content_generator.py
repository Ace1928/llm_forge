"""
Content generation system for LLM Forge.

This module handles the generation of structured content based on parsed
user prompts. It orchestrates calls to language models and ensures the
generation of complete, well-formatted responses.
"""

from typing import Dict, List

from llm_forge_proto.logging_config import configure_logging
from llm_forge_proto.type_definitions import ModelResponse, StructuredInput

# Configure logging
logger = configure_logging()


def generate_response(structured_input: StructuredInput) -> ModelResponse:
    """
    Generate a comprehensive response based on the structured input.

    Iterates through the models and sections specified in the input,
    generating appropriate content for each combination.

    Args:
        structured_input: Parsed and structured user prompt information

    Returns:
        A ModelResponse containing generated content for each model and section
    """
    logger.info(f"Generating content for topic: {structured_input['topic']}")

    # Initialize response structure
    response: ModelResponse = {"topic": structured_input["topic"], "models": {}}

    # Generate content for each model and section
    for model_name in structured_input["models"]:
        logger.debug(f"Generating content for model: {model_name}")
        response["models"][model_name] = {}

        for section_name in structured_input["sections"]:
            logger.debug(f"Generating {section_name} section for {model_name}")

            # In a real implementation, this would call an actual LLM API
            # Here we're just creating placeholder content
            content = _simulate_content_generation(
                model_name, section_name, structured_input["topic"]
            )

            response["models"][model_name][section_name] = content

    return response


def ensure_complete_response(
    response: ModelResponse, structured_input: StructuredInput
) -> ModelResponse:
    """
    Ensure the response contains content for all required models and sections.

    Identifies and fills in any missing sections in the response by
    generating additional content as needed.

    Args:
        response: The initial model response to check
        structured_input: The original structured input for reference

    Returns:
        A complete ModelResponse with all required sections
    """
    logger.info("Checking response completeness")

    # Identify missing sections for each model
    missing_sections = _identify_missing_sections(response, structured_input)

    # If everything is complete, return as is
    if not any(sections for sections in missing_sections.values()):
        logger.info("Response is already complete")
        return response

    # Generate content for missing sections
    updated_response = _fill_missing_sections(response, missing_sections)

    logger.info("Response completeness check finished")
    return updated_response


def _simulate_content_generation(model_name: str, section_name: str, topic: str) -> str:
    """
    Simulate content generation for development/testing purposes.

    In a production system, this would call an actual LLM API service.

    Args:
        model_name: The name of the model to simulate
        section_name: The section to generate content for
        topic: The main topic of the content

    Returns:
        Simulated content as a string
    """
    # This is a placeholder. In a real implementation, this would call an LLM API
    return f"Simulated {section_name} content for {model_name} about {topic}."


def _identify_missing_sections(
    response: ModelResponse, structured_input: StructuredInput
) -> Dict[str, List[str]]:
    """
    Identify which sections are missing from each model in the response.

    Args:
        response: The model response to check
        structured_input: The structured input containing expected sections

    Returns:
        Dictionary mapping model names to lists of missing section names
    """
    missing_sections: Dict[str, List[str]] = {}

    for model_name in structured_input["models"]:
        # Check if model exists in response
        if model_name not in response["models"]:
            missing_sections[model_name] = structured_input["sections"].copy()
            continue

        # Check for missing sections in this model
        model_missing_sections = []
        for section_name in structured_input["sections"]:
            if section_name not in response["models"][model_name]:
                model_missing_sections.append(section_name)

        if model_missing_sections:
            missing_sections[model_name] = model_missing_sections

    return missing_sections


def _fill_missing_sections(
    response: ModelResponse, missing_sections: Dict[str, List[str]]
) -> ModelResponse:
    """
    Generate content to fill in missing sections in the response.

    Args:
        response: The incomplete model response
        missing_sections: Dictionary mapping models to their missing sections

    Returns:
        Updated ModelResponse with previously missing sections filled in
    """
    updated_response = response.copy()
    topic = response["topic"]

    for model_name, sections in missing_sections.items():
        # Ensure model exists in response
        if model_name not in updated_response["models"]:
            updated_response["models"][model_name] = {}

        # Generate content for each missing section
        for section_name in sections:
            logger.debug(f"Filling missing {section_name} for {model_name}")

            # Generate content for the missing section
            content = _simulate_content_generation(model_name, section_name, topic)
            updated_response["models"][model_name][section_name] = content

    return updated_response
