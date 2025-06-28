from pathlib import Path

from rich.panel import Panel

from prompt_mixer.commands.operations.merge import merge_content
from prompt_mixer.config import Config, DEFAULT_ROOT_CONTEXT_FILENAME

from prompt_mixer.gateways.llm import LLMGateway


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
        output_file = config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        if output_file.exists():
            # Merge with existing content
            existing_content = output_file.read_text()

            # Use LLM to merge content intelligently, passing the console for conflict resolution
            merged_content = merge_content(existing_content, ingest_content, llm_gateway, console)
            output_file.write_text(merged_content)

            console.print("[green]Successfully merged prompt with existing context.md[/green]")
        else:
            # Simply save the monolithic prompt as context.md
            output_file.write_text(ingest_content)
            console.print("[green]Successfully imported prompt as context.md[/green]")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {str(e)}[/red]")
