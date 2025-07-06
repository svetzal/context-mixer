"""
Command-line interface for Context Mixer.

This module provides the main entry point for the command-line interface.
"""

import asyncio
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
from context_mixer.commands.assemble import do_assemble
from context_mixer.commands.quarantine import (
    do_quarantine_list,
    do_quarantine_review,
    do_quarantine_resolve,
    do_quarantine_stats,
    do_quarantine_clear
)
from context_mixer.config import Config
from context_mixer.gateways.git import GitGateway
from context_mixer.gateways.llm import LLMGateway

# Create Typer app
app = typer.Typer(
    name="cmx",
    help="A tool to create, organize, merge and deploy reusable context instructions.",
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
            help="Path to initialize the context library (default: $HOME/.context-mixer)"
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
    Initialize a new context library.

    Creates a new context library at the specified path (or default location),
    initializes it as a Git repository, and sets up the default taxonomy structure.
    """
    # Create a new config with the specified path if provided
    do_init(console, Config(library_path), remote, provider, model, git_gateway)


@app.command()
def assemble(
        target: str = typer.Argument(..., help="Target AI assistant (e.g., 'copilot', 'claude')"),
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
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
        project_ids: Optional[str] = typer.Option(
            None,
            help="Comma-separated list of project IDs to include (e.g., 'react-frontend,python-api')"
        ),
        exclude_projects: Optional[str] = typer.Option(
            None,
            help="Comma-separated list of project IDs to exclude (e.g., 'legacy-system,deprecated-app')"
        ),
        token_budget: int = typer.Option(
            8192,
            help="Maximum token budget for assembled context"
        ),
        quality_threshold: float = typer.Option(
            0.8,
            help="Minimum quality threshold for included chunks"
        ),
):
    """
    Assemble context fragments for a specific target.

    Collects relevant fragments, orders them, and renders them into the format
    required by the specified target AI assistant. Use project_ids and 
    exclude_projects to control which project contexts are included to prevent
    cross-project contamination.
    """
    # Parse project filtering parameters
    project_id_list = project_ids.split(',') if project_ids else None
    exclude_project_list = exclude_projects.split(',') if exclude_projects else None

    # Create a new config with the specified library path if provided
    asyncio.run(do_assemble(
        console, 
        Config(library_path), 
        target, 
        output, 
        profile, 
        filter, 
        project_id_list,
        exclude_project_list,
        token_budget, 
        quality_threshold
    ))


@app.command()
def slice(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
        output: Optional[Path] = typer.Option(
            None,
            help="Output path for the sliced fragments"
        ),
        granularity: str = typer.Option(
            "basic",
            help="Level of detail: basic, detailed, comprehensive"
        ),
        domains: Optional[str] = typer.Option(
            None,
            help="Comma-separated list of domains to focus on (technical, business, operational, security, etc.)"
        ),
        project_ids: Optional[str] = typer.Option(
            None,
            help="Comma-separated list of project IDs to include"
        ),
        exclude_projects: Optional[str] = typer.Option(
            None,
            help="Comma-separated list of project IDs to exclude"
        ),
        authority_level: Optional[str] = typer.Option(
            None,
            help="Minimum authority level: official, verified, community, experimental"
        ),
):
    """
    Analyze context.md and split it into pieces based on content categories with CRAFT-aware filtering.

    Extracts content about:
    - why (purpose of the project)
    - who (people and organizations involved)
    - what (components, technologies, and engineering approaches)
    - how (processes and workflows)

    Enhanced with CRAFT parameters:
    - Granularity levels for different detail depths
    - Domain filtering for focused extraction
    - Project-aware filtering to prevent cross-contamination
    - Authority-level filtering for quality control

    Only creates output files for categories with applicable content.
    """
    # Parse comma-separated lists
    domain_list = domains.split(',') if domains else None
    project_id_list = project_ids.split(',') if project_ids else None
    exclude_project_list = exclude_projects.split(',') if exclude_projects else None

    # Create a new config with the specified library path if provided
    do_slice(
        console, 
        Config(library_path), 
        llm_gateway, 
        output,
        granularity=granularity,
        domains=domain_list,
        project_ids=project_id_list,
        exclude_projects=exclude_project_list,
        authority_level=authority_level
    )


@app.command()
def ingest(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to initialize the context library (default: $HOME/.context-mixer)"
        ),
        path: Path = typer.Argument(..., help="Path to file or directory to ingest"),
        project_id: Optional[str] = typer.Option(
            None,
            help="Project identifier for organizing knowledge by project"
        ),
        project_name: Optional[str] = typer.Option(
            None,
            help="Human-readable project name"
        ),
):
    """
    Ingest existing context artifacts into the library.

    Analyzes the specified file or project directory, imports context files, 
    lint configs, and style guides into the context library. Use project_id
    and project_name to organize knowledge by project and prevent cross-project
    contamination.
    """
    # Create a new config with the specified library path if provided
    asyncio.run(do_ingest(console, Config(library_path), llm_gateway, path, project_id, project_name))


@app.command()
def sync(
        pull: bool = typer.Option(False, "--pull", help="Only pull changes from remote"),
        push: bool = typer.Option(False, "--push", help="Only push changes to remote"),
        rebase: bool = typer.Option(False, "--rebase", help="Use rebase strategy when pulling"),
):
    """
    Synchronize the context library with a remote Git repository.

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
    Open the context library in the default editor.

    Uses the editor specified by the $EDITOR environment variable,
    falling back to VS Code if not set.
    """
    do_open(console, Config())


# Create quarantine subcommand group
quarantine_app = typer.Typer(help="Manage quarantined knowledge chunks")
app.add_typer(quarantine_app, name="quarantine")


@quarantine_app.command("list")
def quarantine_list(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
        reason: Optional[str] = typer.Option(
            None,
            help="Filter by quarantine reason (semantic_conflict, authority_conflict, etc.)"
        ),
        resolved: Optional[bool] = typer.Option(
            None,
            help="Filter by resolution status (true for resolved, false for unresolved)"
        ),
        priority: Optional[int] = typer.Option(
            None,
            help="Filter by priority level (1=high, 5=low)"
        ),
        project: Optional[str] = typer.Option(
            None,
            help="Filter by project ID"
        ),
):
    """
    List quarantined knowledge chunks with optional filtering.

    Shows all quarantined chunks that match the specified filters.
    Use filters to narrow down results by reason, resolution status, priority, or project.
    """
    do_quarantine_list(console, Config(library_path), reason, resolved, priority, project)


@quarantine_app.command("review")
def quarantine_review(
        chunk_id: str = typer.Argument(..., help="ID of the quarantined chunk to review"),
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
):
    """
    Review a specific quarantined chunk in detail.

    Shows comprehensive information about a quarantined chunk including
    the reason for quarantine, conflicting chunks, and resolution status.
    """
    do_quarantine_review(console, Config(library_path), chunk_id)


@quarantine_app.command("resolve")
def quarantine_resolve(
        chunk_id: str = typer.Argument(..., help="ID of the quarantined chunk to resolve"),
        action: str = typer.Argument(..., help="Resolution action (accept, reject, merge, modify, defer, escalate)"),
        reason: str = typer.Argument(..., help="Reason for the resolution"),
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
        resolved_by: Optional[str] = typer.Option(
            None,
            help="Who is resolving the quarantine"
        ),
        notes: Optional[str] = typer.Option(
            None,
            help="Additional notes about the resolution"
        ),
):
    """
    Resolve a quarantined chunk with the specified action.

    Available actions:
    - accept: Accept the chunk, overriding conflicts
    - reject: Permanently reject the chunk
    - merge: Merge with existing conflicting knowledge
    - modify: Modify the chunk to resolve conflicts
    - defer: Defer resolution to later time
    - escalate: Escalate to higher authority
    """
    do_quarantine_resolve(console, Config(library_path), chunk_id, action, reason, resolved_by, notes)


@quarantine_app.command("stats")
def quarantine_stats(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
):
    """
    Display quarantine system statistics.

    Shows comprehensive statistics about quarantined chunks including
    counts by reason, priority breakdown, and age information.
    """
    do_quarantine_stats(console, Config(library_path))


@quarantine_app.command("clear")
def quarantine_clear(
        library_path: Optional[Path] = typer.Option(
            None,
            help="Path to the context library (default: $HOME/.context-mixer)"
        ),
):
    """
    Clear all resolved quarantined chunks from the system.

    Removes all quarantined chunks that have been resolved to clean up
    the quarantine system. Requires confirmation before proceeding.
    """
    do_quarantine_clear(console, Config(library_path))


if __name__ == "__main__":
    app()
