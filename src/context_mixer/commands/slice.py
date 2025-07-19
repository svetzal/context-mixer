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
from .base import Command, CommandContext, CommandResult


class SliceCommand(Command):
    """
    Command for slicing context files into categories.

    Implements the Command pattern as specified in the architectural improvements backlog.
    """

    async def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the slice command with the given context.

        Args:
            context: CommandContext containing console, config, llm_gateway, and parameters

        Returns:
            CommandResult indicating success/failure and any relevant data
        """
        try:
            # Extract parameters from context
            output_path = context.parameters.get('output_path')
            granularity = context.parameters.get('granularity', 'basic')
            domains = context.parameters.get('domains')
            project_ids = context.parameters.get('project_ids')
            exclude_projects = context.parameters.get('exclude_projects')
            authority_level = context.parameters.get('authority_level')

            # Call the existing implementation for backward compatibility
            do_slice(
                console=context.console,
                config=context.config,
                llm_gateway=context.llm_gateway,
                output_path=output_path,
                granularity=granularity,
                domains=domains,
                project_ids=project_ids,
                exclude_projects=exclude_projects,
                authority_level=authority_level
            )

            return CommandResult(
                success=True,
                message="Context sliced successfully",
                data={
                    'output_path': str(output_path) if output_path else None,
                    'granularity': granularity,
                    'domains': domains,
                    'project_ids': project_ids,
                    'exclude_projects': exclude_projects,
                    'authority_level': authority_level
                }
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to slice context: {str(e)}",
                error=e
            )


def do_slice(
    console, 
    config: Config, 
    llm_gateway: LLMGateway, 
    output_path: Optional[Path] = None,
    granularity: str = "basic",
    domains: Optional[List[str]] = None,
    project_ids: Optional[List[str]] = None,
    exclude_projects: Optional[List[str]] = None,
    authority_level: Optional[str] = None
):
    """
    Slice the context.md file into pieces based on content categories with CRAFT-aware filtering.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating content
        output_path: Optional path to write the output files to
        granularity: Level of detail (basic, detailed, comprehensive)
        domains: List of domains to focus on (technical, business, operational, etc.)
        project_ids: List of project IDs to include
        exclude_projects: List of project IDs to exclude
        authority_level: Minimum authority level filter
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

        # Build enhanced parameters display
        params_info = []
        if granularity != "basic":
            params_info.append(f"Granularity: {granularity}")
        if domains:
            params_info.append(f"Domains: {', '.join(domains)}")
        if project_ids:
            params_info.append(f"Projects: {', '.join(project_ids)}")
        if exclude_projects:
            params_info.append(f"Excluding: {', '.join(exclude_projects)}")
        if authority_level:
            params_info.append(f"Authority: {authority_level}+")

        title_suffix = f" ({', '.join(params_info)})" if params_info else ""
        console.print(Panel(f"Slicing {DEFAULT_ROOT_CONTEXT_FILENAME} into categories{title_suffix}", title="Context Mixer"))

        # Process each category
        for category in categories:
            console.print(f"Extracting [bold]{category}[/bold] content...")

            # Create the extraction prompt for this category with enhanced parameters
            extraction_result = extract_category_content(
                category, 
                context_content, 
                llm_gateway,
                granularity=granularity,
                domains=domains,
                project_ids=project_ids,
                exclude_projects=exclude_projects,
                authority_level=authority_level
            )

            # If content was found, write it to a file
            if extraction_result.lower() != "not applicable":
                # Add granularity suffix to filename if not basic
                filename_suffix = f"_{granularity}" if granularity != "basic" else ""
                output_file = output_dir / f"{category}{filename_suffix}.md"
                output_file.write_text(extraction_result)
                console.print(f"[green]Created {output_file}[/green]")
            else:
                console.print(f"[yellow]No content found for category: {category}[/yellow]")

        console.print("[green]Slicing completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]Error during slicing: {str(e)}[/red]")


def extract_category_content(
    category: str, 
    context_content: str, 
    llm_gateway: LLMGateway,
    granularity: str = "basic",
    domains: Optional[List[str]] = None,
    project_ids: Optional[List[str]] = None,
    exclude_projects: Optional[List[str]] = None,
    authority_level: Optional[str] = None
) -> str:
    """
    Extract content for a specific category from the context content with CRAFT-aware filtering.

    Args:
        category: The category to extract (why, who, what, how)
        context_content: The content of the context.md file
        llm_gateway: The LLM gateway to use for extraction
        granularity: Level of detail (basic, detailed, comprehensive)
        domains: List of domains to focus on
        project_ids: List of project IDs to include
        exclude_projects: List of project IDs to exclude
        authority_level: Minimum authority level filter

    Returns:
        The extracted content or "not applicable" if no content was found
    """
    # Create the system message for extraction with enhanced parameters
    system_prompt = create_extraction_system_prompt(
        category,
        granularity=granularity,
        domains=domains,
        project_ids=project_ids,
        exclude_projects=exclude_projects,
        authority_level=authority_level
    )

    # Create the user message with the context content
    user_message = LLMMessage(
        role=MessageRole.User,
        content=f"Here is the content to analyze:\n\n{context_content}"
    )

    # Generate the extraction using the LLM
    messages = [system_prompt, user_message]
    return llm_gateway.generate(messages)


def create_extraction_system_prompt(
    category: str,
    granularity: str = "basic",
    domains: Optional[List[str]] = None,
    project_ids: Optional[List[str]] = None,
    exclude_projects: Optional[List[str]] = None,
    authority_level: Optional[str] = None
) -> LLMMessage:
    """
    Create a system prompt for extracting a specific category of content with CRAFT-aware filtering.

    Args:
        category: The category to extract (why, who, what, how)
        granularity: Level of detail (basic, detailed, comprehensive)
        domains: List of domains to focus on
        project_ids: List of project IDs to include
        exclude_projects: List of project IDs to exclude
        authority_level: Minimum authority level filter

    Returns:
        A system message for the extraction
    """
    # Build granularity instructions
    granularity_instructions = {
        "basic": "Extract only the most essential information.",
        "detailed": "Extract comprehensive information including examples, rationale, and context.",
        "comprehensive": "Extract all relevant information including detailed explanations, examples, edge cases, and background context."
    }

    # Build domain filtering instructions
    domain_filter = ""
    if domains:
        domain_filter = f"\n\nFOCUS DOMAINS: Pay special attention to content related to: {', '.join(domains)}. Prioritize information from these domains while still extracting relevant content from other areas."

    # Build project filtering instructions
    project_filter = ""
    if project_ids:
        project_filter = f"\n\nPROJECT SCOPE: Focus on content related to projects: {', '.join(project_ids)}."
    if exclude_projects:
        project_filter += f"\n\nEXCLUDE PROJECTS: Ignore content specifically related to: {', '.join(exclude_projects)}."

    # Build authority level instructions
    authority_filter = ""
    if authority_level:
        authority_filter = f"\n\nAUTHORITY LEVEL: Prioritize content marked as '{authority_level}' or higher authority levels (official > verified > community > experimental)."

    prompts = {
        "why": f"""
            You are an expert content analyzer. Extract ONLY content about the purpose of the project.
            Focus on why the project exists, its goals, mission, and vision.

            GRANULARITY: {granularity_instructions[granularity]}{domain_filter}{project_filter}{authority_filter}

            If there is no content about the purpose of the project, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# Why" at the beginning of your response.
        """,
        "who": f"""
            You are an expert content analyzer. Extract ONLY content about the people and organizations
            creating and sponsoring the project.
            Focus on who is involved, team members, stakeholders, and organizational affiliations.

            GRANULARITY: {granularity_instructions[granularity]}{domain_filter}{project_filter}{authority_filter}

            If there is no content about the people or organizations involved, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# Who" at the beginning of your response.
        """,
        "what": f"""
            You are an expert content analyzer. Extract ONLY content about the rules around components of the project,
            the rules about the technologies being leveraged in the project, and rules about engineering approaches.
            Focus on what the project is, its components, technologies, and engineering guidelines.

            GRANULARITY: {granularity_instructions[granularity]}{domain_filter}{project_filter}{authority_filter}

            If there is no content about the project components, technologies, or engineering rules, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# What" at the beginning of your response.
        """,
        "how": f"""
            You are an expert content analyzer. Extract ONLY content about processes or workflows.
            Focus on how things are done, methodologies, procedures, and operational guidelines.

            GRANULARITY: {granularity_instructions[granularity]}{domain_filter}{project_filter}{authority_filter}

            If there is no content about processes or workflows, respond with ONLY the text "not applicable".

            Do NOT extrapolate or infer additional details. ONLY use content explicitly stated in the provided text.
            Format the extracted content in Markdown, preserving the original structure where possible.
            Include a top-level heading "# How" at the beginning of your response.
        """
    }

    return system_message(prompts[category])
