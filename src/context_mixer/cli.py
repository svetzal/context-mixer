"""
Command-line interface for Context Mixer.

This module provides the main entry point for the command-line interface.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

IS_NOT_YET_IMPLEMENTED = "This functionality is not yet implemented."

APP_NAME = "Context Mixer"

logging.basicConfig(level=logging.WARN)

import typer
from rich.console import Console
from rich.panel import Panel

from context_mixer.commands.init import InitCommand
from context_mixer.commands.ingest import IngestCommand
from context_mixer.commands.open import OpenCommand
from context_mixer.commands.slice import SliceCommand
from context_mixer.commands.assemble import AssembleCommand
from context_mixer.utils.event_driven_progress import create_cli_progress_tracker
from context_mixer.commands.quarantine import (
    QuarantineListCommand,
    QuarantineReviewCommand,
    QuarantineResolveCommand,
    QuarantineStatsCommand,
    QuarantineClearCommand
)
from context_mixer.commands.base import CommandContext
from context_mixer.config import Config
from context_mixer.cli_params import CLIParameterHandler
from context_mixer.gateways.git import GitGateway
from context_mixer.gateways.llm_factory import create_default_llm_gateway
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory

# Create Typer app
app = typer.Typer(
    name="cmx",
    help="A tool to create, organize, merge and deploy reusable context instructions.",
    add_completion=False,
)

console = Console()

git_gateway = GitGateway()

# Global LLM gateway - will be created on demand
llm_gateway = None
llm_gateway_error = None


def get_llm_gateway():
    """
    Get the LLM gateway, ensuring it's properly configured.
    
    Returns:
        LLMGateway: The configured LLM gateway.
        
    Raises:
        typer.Exit: If the gateway is not configured properly.
    """
    global llm_gateway, llm_gateway_error
    
    if llm_gateway is None:
        try:
            llm_gateway = create_default_llm_gateway()
        except ValueError as e:
            llm_gateway_error = str(e)
        except Exception as e:
            llm_gateway_error = f"Network error during gateway initialization: {str(e)[:100]}..."
        
        if llm_gateway is None:
            if llm_gateway_error:
                console.print(f"[red]Error: {llm_gateway_error}[/red]")
            else:
                console.print("[red]Error: LLM gateway not configured properly.[/red]")
            
            console.print("\nTo configure the LLM gateway, run:")
            console.print("  [cyan]cmx config --provider openai --model o4-mini --api-key YOUR_API_KEY[/cyan]")
            console.print("  [cyan]cmx config --provider ollama --model phi3[/cyan]")
            console.print("\nTo see current configuration:")
            console.print("  [cyan]cmx config --show[/cyan]")
            raise typer.Exit(1)
    
    return llm_gateway

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
    config = Config(library_path)

    # Create and execute the init command
    init_command = InitCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        git_gateway=git_gateway,
        parameters={
            'remote': remote,
            'provider': provider,
            'model': model
        }
    )

    asyncio.run(init_command.execute(context))


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
        verbose: bool = typer.Option(
            False,
            help="Enable verbose mode to show all metadata and chunk provenance information"
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
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the assemble command
    assemble_command = AssembleCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        llm_gateway=llm_gateway_instance,
        knowledge_store=knowledge_store,
        parameters={
            'target': target,
            'output': output,
            'profile': profile,
            'filter': filter,
            'project_ids': project_id_list,
            'exclude_projects': exclude_project_list,
            'token_budget': token_budget,
            'quality_threshold': quality_threshold,
            'verbose': verbose
        }
    )

    asyncio.run(assemble_command.execute(context))


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
    config = Config(library_path)

    # Create and execute the slice command
    slice_command = SliceCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        llm_gateway=get_llm_gateway(),
        parameters={
            'output_path': output,
            'granularity': granularity,
            'domains': domain_list,
            'project_ids': project_id_list,
            'exclude_projects': exclude_project_list,
            'authority_level': authority_level
        }
    )

    asyncio.run(slice_command.execute(context))


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
        clustering: bool = typer.Option(
            True,
            help="Enable HDBSCAN clustering for optimized conflict detection"
        ),
        min_cluster_size: int = typer.Option(
            3,
            help="Minimum number of chunks required to form a cluster"
        ),
        clustering_fallback: bool = typer.Option(
            True,
            help="Fall back to traditional O(n²) conflict detection if clustering fails"
        ),
        batch_size: int = typer.Option(
            5,
            help="Number of conflict detections to process concurrently"
        ),
):
    """
    Ingest existing context artifacts into the library.

    Analyzes the specified file or project directory, imports context files, 
    lint configs, and style guides into the context library. Use project_id
    and project_name to organize knowledge by project and prevent cross-project
    contamination.
    
    HDBSCAN clustering is enabled by default to optimize conflict detection
    from O(n²) to O(k*log(k)) by grouping semantically similar chunks and 
    only checking conflicts within/between related clusters.
    """
    # Create a new config with the specified library path and clustering settings
    config = CLIParameterHandler.create_config_from_ingest_params(
        library_path=library_path,
        clustering=clustering,
        min_cluster_size=min_cluster_size,
        clustering_fallback=clustering_fallback,
        batch_size=batch_size
    )

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance, config=config)

    # Create and execute the ingest command with injected dependencies
    ingest_command = IngestCommand(knowledge_store)

    # Create event-driven progress tracking for CLI
    # This will automatically subscribe to progress events and display progress bars
    event_driven_progress_tracker = create_cli_progress_tracker(console)

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        llm_gateway=llm_gateway_instance,
        knowledge_store=knowledge_store,
        parameters={
            'path': path,
            'project_id': project_id,
            'project_name': project_name
        }
    )

    asyncio.run(ingest_command.execute(context))


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
    # Create and execute the open command
    open_command = OpenCommand()

    # Create command context with all necessary dependencies
    context = CommandContext(
        console=console,
        config=Config()
    )

    asyncio.run(open_command.execute(context))


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
    # Create a new config with the specified library path if provided
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the quarantine list command
    quarantine_list_command = QuarantineListCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        knowledge_store=knowledge_store,
        parameters={
            'reason': reason,
            'resolved': resolved,
            'priority': priority,
            'project': project
        }
    )

    asyncio.run(quarantine_list_command.execute(context))


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
    # Create a new config with the specified library path if provided
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the quarantine review command
    quarantine_review_command = QuarantineReviewCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        knowledge_store=knowledge_store,
        parameters={
            'chunk_id': chunk_id
        }
    )

    asyncio.run(quarantine_review_command.execute(context))


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
    # Create a new config with the specified library path if provided
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the quarantine resolve command
    quarantine_resolve_command = QuarantineResolveCommand()

    # Create command context with all necessary dependencies and parameters
    context = CommandContext(
        console=console,
        config=config,
        knowledge_store=knowledge_store,
        parameters={
            'chunk_id': chunk_id,
            'action': action,
            'reason': reason,
            'resolved_by': resolved_by,
            'notes': notes
        }
    )

    asyncio.run(quarantine_resolve_command.execute(context))


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
    # Create a new config with the specified library path if provided
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the quarantine stats command
    quarantine_stats_command = QuarantineStatsCommand()

    # Create command context with all necessary dependencies
    context = CommandContext(
        console=console,
        config=config,
        knowledge_store=knowledge_store
    )

    asyncio.run(quarantine_stats_command.execute(context))


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
    # Create a new config with the specified library path if provided
    config = Config(library_path)

    # Create knowledge store with dependency injection
    vector_store_path = config.library_path / "vector_store"
    llm_gateway_instance = get_llm_gateway()
    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway_instance)

    # Create and execute the quarantine clear command
    quarantine_clear_command = QuarantineClearCommand()

    # Create command context with all necessary dependencies
    context = CommandContext(
        console=console,
        config=config,
        knowledge_store=knowledge_store
    )

    asyncio.run(quarantine_clear_command.execute(context))


@app.command(name="config")
def configure(
    provider: Optional[str] = typer.Option(
        None,
        help="LLM provider to use (openai, ollama)"
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Model name to use with the provider"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key for providers that require it (e.g., OpenAI)"
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
):
    """
    Configure LLM gateway settings.
    """
    config = Config.load()
    
    if show:
        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"  Library Path: {config.library_path}")
        console.print(f"  LLM Provider: {config.llm_provider}")
        console.print(f"  LLM Model: {config.llm_model}")
        console.print(f"  API Key: {'***' if config.llm_api_key else 'Not set'}")
        console.print(f"  Clustering Enabled: {config.clustering_enabled}")
        console.print(f"  Batch Size: {config.conflict_detection_batch_size}")
        return
    
    # Update configuration if any parameters are provided
    updated = False
    
    if provider is not None:
        if provider.lower() not in ["openai", "ollama"]:
            console.print(f"[red]Error: Unsupported provider '{provider}'. Use 'openai' or 'ollama'.[/red]")
            raise typer.Exit(1)
        config._llm_provider = provider.lower()
        updated = True
        console.print(f"✓ Set provider to: {provider.lower()}")
    
    if model is not None:
        config._llm_model = model
        updated = True
        console.print(f"✓ Set model to: {model}")
    
    if api_key is not None:
        config._llm_api_key = api_key
        updated = True
        console.print("✓ Set API key")
    
    if updated:
        try:
            config.save()
            console.print(f"✓ Configuration saved to: {config.config_path}")
            
            # Try to validate the configuration by creating a gateway
            from context_mixer.gateways.llm_factory import create_llm_gateway
            try:
                create_llm_gateway(config)
                console.print("✓ Configuration validated successfully")
            except ValueError as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")
            except Exception as e:
                # Handle network errors during validation gracefully
                console.print(f"[yellow]Warning: Could not validate configuration due to network error: {str(e)[:100]}...[/yellow]")
                console.print("[yellow]Configuration saved, but validation skipped due to network limitations.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error saving configuration: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("No configuration changes specified. Use --help to see available options.")


if __name__ == "__main__":
    app()
