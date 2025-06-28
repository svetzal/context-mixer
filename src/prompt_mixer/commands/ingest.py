from pathlib import Path

from mojentic.llm import LLMBroker
from rich.panel import Panel

from prompt_mixer.config import Config


def do_ingest(console, config: Config, llm_broker: LLMBroker, filename: Path=None):
    """
    Ingest existing prompt artifacts into the library.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_broker: Not used in the simplified version
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

        # Simply save the monolithic prompt as instructions.md
        output_file = config.library_path / "instructions.md"
        output_file.write_text(ingest_content)

        console.print("[green]Successfully imported prompt as instructions.md[/green]")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {str(e)}[/red]")
