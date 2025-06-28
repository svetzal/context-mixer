"""
Command-line interface for Prompt Mixer.

This module provides the main entry point for the command-line interface.
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

IS_NOT_YET_IMPLEMENTED = "This functionality is not yet implemented."

APP_NAME = "Prompt Mixer"

logging.basicConfig(level=logging.WARN)

import typer
from mojentic.llm import LLMBroker
from mojentic.llm.gateways import OpenAIGateway
from rich.console import Console
from rich.panel import Panel

from prompt_mixer.commands.init import do_init
from prompt_mixer.commands.ingest import do_ingest
from prompt_mixer.commands.open import do_open
from prompt_mixer.config import Config
from prompt_mixer.gateways.git import GitGateway

# Create Typer app
app = typer.Typer(
    name="pmx",
    help="A tool to create, organize, merge and deploy reusable prompt instructions.",
    add_completion=False,
)

console = Console()

git_gateway = GitGateway()

llm_gateway = OpenAIGateway(api_key=os.environ.get("OPENAI_API_KEY"))
llm_broker = LLMBroker(model="gpt-4.1", gateway=llm_gateway)

@app.command()
def init(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to initialize the prompt library (default: $HOME/.prompt-mixer)"
        ),
        remote: Optional[str] = typer.Option(
            None,
            help="URL of remote Git repository to link"
        ),
        provider: Optional[str] = typer.Option(
            None,
            help="LLM provider (e.g., 'ollama', 'openai')"
        ),
        model: Optional[str] = typer.Option(
            None,
            help="LLM model to use (e.g., 'phi3', 'gpt-4o')"
        ),
):
    """
    Initialize a new prompt library.

    Creates a new prompt library at the specified path (or default location),
    initializes it as a Git repository, and sets up the default taxonomy structure.
    """
    # Create a new config with the specified path if provided
    do_init(console, Config(library_path), remote, provider, model, git_gateway)


@app.command()
def assemble(
        target: str = typer.Argument(..., help="Target AI assistant (e.g., 'copilot', 'claude')"),
        output: Optional[Path] = typer.Option(
            None,
            help="Output path for the assembled prompt"
        ),
        profile: Optional[str] = typer.Option(
            None,
            help="LLM profile to use (e.g., 'ollama://phi3', 'openai://gpt-4o')"
        ),
        filter: Optional[str] = typer.Option(
            None,
            help="Filter fragments by tags (e.g., 'lang:python,layer:testing')"
        ),
):
    """
    Assemble prompt fragments for a specific target.

    Collects relevant fragments, orders them, and renders them into the format
    required by the specified target AI assistant.
    """
    console.print(
        Panel(f"Assembling prompts for target: [bold]{target}[/bold]", title=APP_NAME))
    console.print(IS_NOT_YET_IMPLEMENTED)


@app.command()
def slice(
        filters: List[str] = typer.Argument(None,
                                            help="Filters to apply (e.g., 'lang:python', "
                                                 "'layer:testing')"),
        output: Optional[Path] = typer.Option(
            None,
            help="Output path for the sliced fragments"
        ),
):
    """
    Output chosen fragments based on filters.

    Filters fragments by tags and outputs them to stdout or a file.
    """
    filters_str = ", ".join(filters) if filters else "none"
    console.print(
        Panel(f"Slicing fragments with filters: [bold]{filters_str}[/bold]", title=APP_NAME))
    console.print(IS_NOT_YET_IMPLEMENTED)


@app.command()
def ingest(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to initialize the prompt library (default: $HOME/.prompt-mixer)"
        ),
        filename: Path = typer.Argument(..., help="Path to instructions to ingest"),
):
    """
    Ingest existing prompt artifacts into the library.

    Analyzes the specified project, imports prompt files, lint configs,
    and style guides into the prompt library.
    """
    # Create a new config with the specified output path if provided
    do_ingest(console, Config(), llm_broker, filename)


@app.command()
def sync(
        pull: bool = typer.Option(False, "--pull", help="Only pull changes from remote"),
        push: bool = typer.Option(False, "--push", help="Only push changes to remote"),
        rebase: bool = typer.Option(False, "--rebase", help="Use rebase strategy when pulling"),
):
    """
    Synchronize the prompt library with a remote Git repository.

    By default, performs a pull followed by a push. Use flags to limit to
    pull or push only, or to specify the pull strategy.
    """
    if pull and push:
        console.print(
            "[yellow]Warning: Both --pull and --push specified. This is the default behavior.["
            "/yellow]")

    strategy = "rebase" if rebase else "merge"
    if not (pull or push):
        action = "Pulling and pushing"
    else:
        action = ("Pulling" if pull else "Pushing")

    console.print(Panel(f"{action} changes using {strategy} strategy", title=APP_NAME))
    console.print(IS_NOT_YET_IMPLEMENTED)


@app.command()
def open():
    """
    Open the prompt library in the default editor.

    Uses the editor specified by the $EDITOR environment variable,
    falling back to VS Code if not set.
    """
    do_open(console, Config())


if __name__ == "__main__":
    app()
