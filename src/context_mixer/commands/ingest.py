import asyncio
import os
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from context_mixer.commands.operations.merge import merge_content
from context_mixer.commands.operations.commit import CommitOperation
from context_mixer.config import Config, DEFAULT_ROOT_CONTEXT_FILENAME

from context_mixer.gateways.llm import LLMGateway
from context_mixer.gateways.git import GitGateway
from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.domain.knowledge_store import KnowledgeStore, KnowledgeStoreFactory


def _find_ingestible_files(directory_path: Path) -> list[Path]:
    """
    Find files in a directory that are suitable for ingestion.

    Args:
        directory_path: Path to the directory to scan

    Returns:
        List of Path objects for files that should be ingested
    """
    ingestible_files = []

    # Common file patterns for context artifacts
    patterns = [
        "**/*.md",
        "**/*.txt",
    ]

    # Search for files matching the patterns
    for pattern in patterns:
        try:
            if "**" in pattern:
                # Handle recursive patterns
                matches = directory_path.rglob(pattern.replace("**/", ""))
            elif "/" in pattern and "*" not in pattern:
                # Handle specific file paths
                specific_path = directory_path / pattern
                if specific_path.exists() and specific_path.is_file():
                    ingestible_files.append(specific_path)
                continue
            else:
                # Handle simple glob patterns
                matches = directory_path.glob(pattern)

            for match in matches:
                if match.is_file() and match not in ingestible_files:
                    # Skip files in common ignore directories
                    if any(part.startswith('.') and part not in ['.github', '.junie'] 
                           for part in match.parts):
                        continue
                    if any(ignore_dir in match.parts 
                           for ignore_dir in ['node_modules', '__pycache__', '.git', 'venv', 'env']):
                        continue
                    ingestible_files.append(match)
        except Exception:
            # Skip patterns that cause errors
            continue

    return sorted(list(set(ingestible_files)))


async def do_ingest(console, config: Config, llm_gateway: LLMGateway, path: Path=None, project_id: str=None, project_name: str=None, commit: bool=True, detect_boundaries: bool=True):
    """
    Ingest existing prompt artifacts into the library using intelligent chunking.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating merged content and chunking
        path: Path to the file or directory to ingest
        project_id: Project identifier for organizing knowledge by project
        project_name: Human-readable project name
        commit: Whether to commit changes after ingestion
        detect_boundaries: Whether to use semantic boundary detection for chunking
    """
    # Disable tokenizer parallelism to avoid warnings when using async operations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        input_path = Path(path)
        if not input_path.exists():
            console.print(f"[red]Error: Path {path} does not exist[/red]")
            return

        # Determine if we're processing a single file or a directory
        if input_path.is_file():
            files_to_process = [input_path]
            console.print(
                Panel(f"Ingesting prompts from file: [bold]{path}[/bold]", title="Context Mixer"))
        elif input_path.is_dir():
            files_to_process = _find_ingestible_files(input_path)
            if not files_to_process:
                console.print(f"[yellow]No ingestible files found in directory: {path}[/yellow]")
                return
            console.print(
                Panel(f"Ingesting prompts from directory: [bold]{path}[/bold]\nFound {len(files_to_process)} files to process", title="Context Mixer"))
        else:
            console.print(f"[red]Error: Path {path} is neither a file nor a directory[/red]")
            return

        # Process all files
        all_chunks = []
        all_content = []

        for file_path in files_to_process:
            console.print(f"[blue]Processing: {file_path.relative_to(input_path) if input_path.is_dir() else file_path.name}[/blue]")

            try:
                file_content = file_path.read_text()
                # For single files, preserve original behavior (no header)
                # For multiple files, add headers to distinguish content
                if len(files_to_process) == 1:
                    all_content.append(file_content)
                else:
                    all_content.append(f"# Content from {file_path}\n\n{file_content}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read {file_path}: {str(e)}[/yellow]")
                continue

        if not all_content:
            console.print("[red]Error: No content could be read from any files[/red]")
            return

        # Combine all content for processing
        ingest_content = "\n\n---\n\n".join(all_content)

        # Ensure the library directory exists
        config.library_path.mkdir(parents=True, exist_ok=True)

        if detect_boundaries:
            # Use ChunkingEngine for intelligent semantic boundary detection with structured output
            console.print("[blue]Analyzing content using structured LLM output for complete chunks...[/blue]")

            # Determine project path - use the directory if ingesting a directory, or parent if ingesting a file
            determined_project_path = str(input_path if input_path.is_dir() else input_path.parent)

            # Display project context information if provided
            if project_id or project_name:
                project_info = f"Project: {project_name or project_id}"
                if project_id and project_name:
                    project_info = f"Project: {project_name} (ID: {project_id})"
                console.print(f"[cyan]{project_info}[/cyan]")

            chunking_engine = ChunkingEngine(llm_gateway)
            chunks = chunking_engine.chunk_by_structured_output(
                ingest_content, 
                source=str(path),
                project_id=project_id,
                project_name=project_name,
                project_path=determined_project_path
            )

            # Display chunking results with validation status
            if chunks:
                # Validate all chunks first
                chunk_validations = []
                valid_chunks = []

                for chunk in chunks:
                    validation_result = chunking_engine.validate_chunk_completeness(chunk)
                    chunk_validations.append((chunk, validation_result))
                    if validation_result.is_complete:
                        valid_chunks.append(chunk)

                # Create comprehensive report table
                table = Table(title="Chunk Ingestion Report")
                table.add_column("Status", style="bold", width=8)
                table.add_column("Chunk ID", style="cyan", width=15)
                table.add_column("Concept", style="green", width=20)
                table.add_column("Domain", style="yellow", width=15)
                table.add_column("Authority", style="magenta", width=12)
                table.add_column("Size", style="blue", width=8)
                table.add_column("Chunk Content", style="white", width=50)
                table.add_column("Validation Details", style="dim", width=40)

                for chunk, validation in chunk_validations:
                    domains_str = ", ".join(chunk.metadata.domains)
                    concept = chunk.metadata.tags[0] if chunk.metadata.tags else "Unknown"

                    if validation.is_complete:
                        status = "[green]✓ PASS[/green]"
                        validation_details = f"Complete (confidence: {validation.confidence:.2f})"
                    else:
                        status = "[red]✗ REJECT[/red]"
                        issues_str = f" | Issues: {', '.join(validation.issues)}" if validation.issues else ""
                        validation_details = f"{validation.reason} (confidence: {validation.confidence:.2f}){issues_str}"

                    # Truncate and format chunk content for display
                    chunk_content = chunk.content.strip()
                    # Replace newlines with spaces and truncate if too long
                    chunk_content_display = chunk_content.replace('\n', ' ').replace('\r', ' ')
                    if len(chunk_content_display) > 200:
                        chunk_content_display = chunk_content_display[:197] + "..."

                    table.add_row(
                        status,
                        chunk.id[:12] + "...",
                        concept,
                        domains_str,
                        chunk.metadata.authority.value,
                        f"{len(chunk.content)}",
                        chunk_content_display,
                        validation_details
                    )

                console.print(table)
                console.print(f"[green]Successfully processed {len(chunks)} chunks: {len(valid_chunks)} accepted, {len(chunks) - len(valid_chunks)} rejected[/green]")

                # Store chunks in vector knowledge store
                vector_store_path = config.library_path / "vector_store"
                knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path)

                try:
                    await knowledge_store.store_chunks(valid_chunks)
                    console.print(f"[green]Successfully stored {len(valid_chunks)} chunks in vector knowledge store[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to store chunks in vector store: {str(e)}[/yellow]")
                    console.print("[yellow]Continuing with context.md storage as fallback[/yellow]")

                # Create a summary for context.md (for compatibility)
                summary_content = f"# Ingested Content from {path}\n\n"
                summary_content += f"Processed on: {chunking_engine._generate_chunk_id('', '')[:8]}\n"
                summary_content += f"Total chunks: {len(valid_chunks)}\n\n"

                for i, chunk in enumerate(valid_chunks, 1):
                    summary_content += f"## Chunk {i}: {chunk.metadata.tags[0] if chunk.metadata.tags else 'Concept'}\n"
                    summary_content += f"**Domain:** {', '.join(chunk.metadata.domains)}\n"
                    summary_content += f"**Authority:** {chunk.metadata.authority.value}\n"
                    summary_content += f"**ID:** {chunk.id}\n\n"
                    summary_content += chunk.content + "\n\n---\n\n"

                output_file = config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
                output_file.write_text(summary_content)

            else:
                console.print("[yellow]No chunks detected, falling back to monolithic storage[/yellow]")
                output_file = config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
                output_file.write_text(ingest_content)
        else:
            # Legacy mode: save as monolithic file
            console.print("[blue]Using legacy monolithic storage (boundary detection disabled)[/blue]")

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
