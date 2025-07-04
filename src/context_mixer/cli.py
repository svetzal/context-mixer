"""
Command-line interface for Context Mixer.

This module provides the main entry point for the command-line interface.
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

IS_NOT_YET_IMPLEMENTED = "This functionality is not yet implemented."

APP_NAME = "Context Mixer"

logging.basicConfig(level=logging.WARN)

import typer
from mojentic.llm.gateways import OpenAIGateway
from rich.console import Console
from rich.panel import Panel

from context_mixer.commands.init import do_init
from context_mixer.commands.ingest import do_ingest
from context_mixer.commands.open import do_open
from context_mixer.commands.slice import do_slice
from context_mixer.config import Config
from context_mixer.gateways.git import GitGateway
from context_mixer.gateways.llm import LLMGateway

# Create Typer app
app = typer.Typer(
    name="pmx",
    help="A tool to create, organize, merge and deploy reusable prompt instructions.",
    add_completion=False,
)

console = Console()

git_gateway = GitGateway()

# Initialize the OpenAI gateway and LLM gateway
openai_gateway = OpenAIGateway(api_key=os.environ.get("OPENAI_API_KEY"))
# llm_gateway = LLMGateway(model="gpt-4.1", gateway=openai_gateway)
llm_gateway = LLMGateway(model="o4-mini", gateway=openai_gateway)

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
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the prompt library (default: $HOME/.prompt-mixer)"
        ),
        output: Optional[Path] = typer.Option(
            None,
            help="Output path for the sliced fragments"
        ),
):
    """
    Analyze context.md and split it into pieces based on content categories.

    Extracts content about:
    - why (purpose of the project)
    - who (people and organizations involved)
    - what (components, technologies, and engineering approaches)
    - how (processes and workflows)

    Only creates output files for categories with applicable content.
    """
    # Create a new config with the specified library path if provided
    do_slice(console, Config(library_path), llm_gateway, output)


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
    # Create a new config with the specified library path if provided
    do_ingest(console, Config(library_path), llm_gateway, filename)


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
