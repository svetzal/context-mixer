from pathlib import Path
from textwrap import dedent

from rich.panel import Panel

from prompt_mixer.config import Config
from prompt_mixer.gateways.llm import LLMGateway, Message


def merge_content(existing_content: str, new_content: str, llm_gateway: LLMGateway) -> str:
    """
    Merge existing content with new content using an LLM to create a coherent document.

    This function uses the LLM gateway to generate a prompt that will leverage the LLM
    in merging the concepts and building a new coherent document from the content
    of the two source documents.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for generating the merged content

    Returns:
        The merged content
    """
    # Create a prompt for the LLM to merge the content
    prompt = dedent(f"""
        Your task is to merge two documents into a single coherent document, accurately representing
         the content of both documents, and without adding anything extra.

        The first document contains existing content, and the second document contains new content
        to be merged.

        Please combine these documents, ensuring that:
        1. All unique information from both documents is preserved
        2. Duplicate information appears only once
        3. The resulting document is well-structured and coherent
        4. Related information is grouped together logically

        Document 1 (Existing Content):
        ```
        {existing_content}
        ```

        Document 2 (New Content):
        ```
        {new_content}
        ```

        Please provide only the merged document as your response, without any additional commentary
        or wrappers.
    """).strip()

    # Create a Message with the prompt
    message = Message(content=prompt)

    # Generate the merged content using the LLM gateway
    response = llm_gateway.generate(messages=[message])

    # Return the generated content
    return response


def do_ingest(console, config: Config, llm_gateway: LLMGateway, filename: Path=None):
    """
    Ingest existing prompt artifacts into the library.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating merged content
        filename: Path to the file to ingest
    """
    console.print(
        Panel(f"Ingesting prompts from: [bold]{filename}[/bold]", title="Prompt Mixer"))

    try:
        monolith_path = Path(filename)
        if not monolith_path.exists():
            console.print(f"[red]Error: Path {filename} does not exist[/red]")
            return

        ingest_content = monolith_path.read_text()

        # Check if context.md already exists
        output_file = config.library_path / "context.md"
        if output_file.exists():
            # Merge with existing content
            existing_content = output_file.read_text()

            # Use LLM to merge content intelligently
            merged_content = merge_content(existing_content, ingest_content, llm_gateway)
            output_file.write_text(merged_content)

            console.print("[green]Successfully merged prompt with existing context.md[/green]")
        else:
            # Simply save the monolithic prompt as context.md
            output_file.write_text(ingest_content)
            console.print("[green]Successfully imported prompt as context.md[/green]")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {str(e)}[/red]")
