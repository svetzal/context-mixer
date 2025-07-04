"""
Slice command implementation for Context Mixer.

This module provides the implementation of the slice command, which analyzes
the context.md file and splits it into pieces based on content categories.
"""

from pathlib import Path
from typing import List, Optional

from rich.panel import Panel
from mojentic.llm import LLMMessage, MessageRole

from context_mixer.config import Config, DEFAULT_ROOT_CONTEXT_FILENAME
from context_mixer.gateways.llm import LLMGateway
from context_mixer.domain.llm_instructions import system_message, clean_prompt


def do_slice(console, config: Config, llm_gateway: LLMGateway, output_path: Optional[Path] = None):
    """
    Slice the context.md file into pieces based on content categories.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating content
        output_path: Optional path to write the output files to
    """
    try:
        # Get the path to the context.md file
        context_file = config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME

        # Check if the file exists
        if not context_file.exists():
            console.print(f"[red]Error: {DEFAULT_ROOT_CONTEXT_FILENAME} does not exist in the library[/red]")
            return

        # Read the content of the file
        context_content = context_file.read_text()

        # Define the categories to extract
        categories = ["why", "who", "what", "how"]

        # Create the output directory if it doesn't exist
        output_dir = output_path or config.library_path / "slices"
        output_dir.mkdir(exist_ok=True, parents=True)

        console.print(Panel(f"Slicing {DEFAULT_ROOT_CONTEXT_FILENAME} into categories", title="Context Mixer"))

        # Process each category
        for category in categories:
            console.print(f"Extracting [bold]{category}[/bold] content...")

            # Create the extraction prompt for this category
            extraction_result = extract_category_content(category, context_content, llm_gateway)

            # If content was found, write it to a file
            if extraction_result.lower() != "not applicable":
                output_file = output_dir / f"{category}.md"
                output_file.write_text(extraction_result)
                console.print(f"[green]Created {output_file}[/green]")
            else:
                console.print(f"[yellow]No content found for category: {category}[/yellow]")

        console.print("[green]Slicing completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]Error during slicing: {str(e)}[/red]")


def extract_category_content(category: str, context_content: str, llm_gateway: LLMGateway) -> str:
    """
    Extract content for a specific category from the context content.

    Args:
        category: The category to extract (why, who, what, how)
        context_content: The content of the context.md file
        llm_gateway: The LLM gateway to use for extraction

    Returns:
        The extracted content or "not applicable" if no content was found
    """
    # Create the system message for extraction
    system_prompt = create_extraction_system_prompt(category)

    # Create the user message with the context content
    user_message = LLMMessage(
        role=MessageRole.User,
        content=f"Here is the content to analyze:\n\n{context_content}"
    )

    # Generate the extraction using the LLM
    messages = [system_prompt, user_message]
    return llm_gateway.generate(messages)


def create_extraction_system_prompt(category: str) -> LLMMessage:
    """
    Create a system prompt for extracting a specific category of content.

    Args:
        category: The category to extract (why, who, what, how)

    Returns:
        A system message for the extraction
    """
    prompts = {
        "why": """
            You are an expert content analyzer. Extract ONLY content about the purpose of the project.
            Focus on why the project exists, its goals, mission, and vision.

            If there is no content about the purpose of the project, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# Why" at the beginning of your response.
        """,
        "who": """
            You are an expert content analyzer. Extract ONLY content about the people and organizations
            creating and sponsoring the project.
            Focus on who is involved, team members, stakeholders, and organizational affiliations.

            If there is no content about the people or organizations involved, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# Who" at the beginning of your response.
        """,
        "what": """
            You are an expert content analyzer. Extract ONLY content about the rules around components of the project,
            the rules about the technologies being leveraged in the project, and rules about engineering approaches.
            Focus on what the project is, its components, technologies, and engineering guidelines.

            If there is no content about the project components, technologies, or engineering rules, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# What" at the beginning of your response.
        """,
        "how": """
            You are an expert content analyzer. Extract ONLY content about processes or workflows.
            Focus on how things are done, methodologies, procedures, and operational guidelines.

            If there is no content about processes or workflows, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# How" at the beginning of your response.
        """
    }

    return system_message(prompts[category])