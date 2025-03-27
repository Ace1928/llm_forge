"""
Content generation system for LLM Forge.

This module handles the generation of structured content based on parsed
user prompts. It orchestrates calls to language models and ensures the
generation of complete, well-formatted responses.
"""

import random
from typing import Dict, Final, List

from llm_forge.logging_config import configure_logging
from llm_forge.templates.content_templates import get_section_template
from llm_forge.type_definitions import ContentGenerator, ModelResponse, StructuredInput

# Configure logging
logger: Final = configure_logging()


class SimulatedContentGenerator:
    """
    Simulated content generator for development and testing.

    Generates realistic-looking content for different models and sections
    without requiring actual LLM API calls.
    """

    def __init__(self, randomize: bool = True) -> None:
        """
        Initialize the simulated content generator.

        Args:
            randomize: Whether to add random variations to generated content
        """
        self.randomize: bool = randomize

    def generate(self, model: str, section: str, topic: str) -> str:
        """
        Generate simulated content for a specific model and section.

        Args:
            model: The model to generate content for
            section: The section to generate
            topic: The topic to generate content about

        Returns:
            Generated content as a string
        """
        # Get template for this section
        template: str = get_section_template(section)

        # Fill in template with model-specific information
        content: str = template.format(
            model=model.upper(),
            topic=topic,
            # Add some model-specific characteristics
            **self._get_model_characteristics(model),
        )

        # Add some randomness if enabled
        if self.randomize:
            content = self._add_variations(content)

        return content

    def _get_model_characteristics(self, model: str) -> Dict[str, str]:
        """
        Get characteristic details for a specific model.

        Args:
            model: The model name

        Returns:
            Dictionary of model characteristics
        """
        characteristics: Final[Dict[str, Dict[str, str]]] = {
            "gpt": {
                "architecture": "transformer-based autoregressive language model",
                "training": "trained on diverse internet text with reinforcement learning from human feedback",
                "strength": "broad general knowledge and instruction following",
                "weakness": "potential for hallucinations and verbosity",
                "company": "OpenAI",
                "year": "2020-2023",
            },
            "claude": {
                "architecture": "constitutional AI framework with advanced reasoning",
                "training": "trained with constitutional AI and RLHF techniques",
                "strength": "thoughtful reasoning and balanced responses",
                "weakness": "sometimes overexplains simple concepts",
                "company": "Anthropic",
                "year": "2022-2023",
            },
            "llama": {
                "architecture": "open-source transformer architecture",
                "training": "trained on publicly available datasets with custom tokenization",
                "strength": "open ecosystem and customizability",
                "weakness": "less refined than commercial counterparts in some tasks",
                "company": "Meta",
                "year": "2023",
            },
            "mistral": {
                "architecture": "mixture-of-experts transformer architecture",
                "training": "trained with sparse mixture of experts approach",
                "strength": "efficient performance and specialized knowledge routing",
                "weakness": "newer model with less ecosystem integration",
                "company": "Mistral AI",
                "year": "2023",
            },
        }

        # Return default characteristics if model not found
        return characteristics.get(
            model.lower(),
            {
                "architecture": "neural network-based language model",
                "training": "trained on text data",
                "strength": "natural language processing",
                "weakness": "limited by training data",
                "company": "Various",
                "year": "recent",
            },
        )

    def _add_variations(self, content: str) -> str:
        """
        Add random variations to make simulated content more realistic.

        Args:
            content: Base content string

        Returns:
            Content with added variations
        """
        # Add some filler phrases randomly
        fillers: Final[List[str]] = [
            "It's worth noting that ",
            "Interestingly, ",
            "According to recent research, ",
            "Many experts believe that ",
            "Based on available information, ",
        ]

        sentences: List[str] = content.split(". ")

        # Add a filler to about 30% of sentences
        for i in range(len(sentences)):
            if random.random() < 0.3:
                sentences[i] = random.choice(fillers) + sentences[i].lower()

        return ". ".join(sentences)


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

    # Create content generator
    generator: SimulatedContentGenerator = SimulatedContentGenerator()

    # Generate content for each model and section
    for model_name in structured_input["models"]:
        logger.debug(f"Generating content for model: {model_name}")
        response["models"][model_name] = {}

        for section_name in structured_input["sections"]:
            logger.debug(f"Generating {section_name} section for {model_name}")

            # Generate content for this model and section
            content: str = generator.generate(
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
    missing_sections: Dict[str, List[str]] = _identify_missing_sections(response, structured_input)

    # If everything is complete, return as is
    if not any(sections for sections in missing_sections.values()):
        logger.info("Response is already complete")
        return response

    # Create content generator for filling missing sections
    generator: SimulatedContentGenerator = SimulatedContentGenerator()

    # Generate content for missing sections
    updated_response: ModelResponse = _fill_missing_sections(
        response, missing_sections, generator, structured_input["topic"]
    )

    logger.info("Response completeness check finished")
    return updated_response


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
        model_missing_sections: List[str] = []
        for section_name in structured_input["sections"]:
            if section_name not in response["models"][model_name]:
                model_missing_sections.append(section_name)

        if model_missing_sections:
            missing_sections[model_name] = model_missing_sections

    return missing_sections


def _fill_missing_sections(
    response: ModelResponse,
    missing_sections: Dict[str, List[str]],
    generator: ContentGenerator,
    topic: str,
) -> ModelResponse:
    """
    Generate content to fill in missing sections in the response.

    Args:
        response: The incomplete model response
        missing_sections: Dictionary mapping models to their missing sections
        generator: Content generator instance
        topic: The main topic for content generation

    Returns:
        Updated ModelResponse with previously missing sections filled in
    """
    updated_response: ModelResponse = {
        "topic": response["topic"],
        "models": {**response["models"]}
    }

    for model_name, sections in missing_sections.items():
        # Ensure model exists in response
        if model_name not in updated_response["models"]:
            updated_response["models"][model_name] = {}

        # Generate content for each missing section
        for section_name in sections:
            logger.debug(f"Filling missing {section_name} for {model_name}")

            # Generate content for the missing section
            content: str = generator.generate(model_name, section_name, topic)
            updated_response["models"][model_name][section_name] = content

    return updated_response
