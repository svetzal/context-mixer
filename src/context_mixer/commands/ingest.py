from pathlib import Path

from rich.panel import Panel

from context_mixer.commands.operations.merge import merge_content
from context_mixer.commands.operations.commit import CommitOperation
from context_mixer.config import Config, DEFAULT_ROOT_CONTEXT_FILENAME

from context_mixer.gateways.llm import LLMGateway
from context_mixer.gateways.git import GitGateway


def do_ingest(console, config: Config, llm_gateway: LLMGateway, filename: Path=None, commit: bool=True):
    """
    Ingest existing prompt artifacts into the library.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating merged content
        filename: Path to the file to ingest
        commit: Whether to commit changes after ingestion
    """
    console.print(
        Panel(f"Ingesting prompts from: [bold]{filename}[/bold]", title="Context Mixer"))

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

        # Commit changes if requested
        if commit:
            try:
                git_gateway = GitGateway()
                commit_operation = CommitOperation(git_gateway, llm_gateway)

                # Commit the changes
                success, message, commit_msg = commit_operation.commit_changes(config.library_path)

                if success:
                    console.print(f"[green]Successfully committed changes: {commit_msg.short}[/green]")
                else:
                    console.print(f"[yellow]Failed to commit changes: {message}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error during commit: {str(e)}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {str(e)}[/red]")
