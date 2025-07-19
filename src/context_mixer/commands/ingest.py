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
from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from context_mixer.commands.interactions.resolve_conflicts import resolve_conflicts, ConflictResolver
from context_mixer.domain.knowledge import KnowledgeChunk
from context_mixer.utils.timing import TimingCollector, time_operation, format_duration
from context_mixer.utils.progress import ProgressTracker, NoOpProgressObserver
from .base import Command, CommandContext, CommandResult


class IngestCommand(Command):
    """
    Command for ingesting content into the knowledge store.

    This class implements dependency injection to improve testability
    and modularity as outlined in the architectural improvements backlog.
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        """
        Initialize the IngestCommand with injected dependencies.

        Args:
            knowledge_store: The knowledge store to use for storing chunks
        """
        self.knowledge_store = knowledge_store

    async def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the ingest command with the given context.

        Args:
            context: CommandContext containing all necessary dependencies and parameters

        Returns:
            CommandResult indicating success/failure and any relevant data
        """
        try:
            # Extract parameters from context
            path = context.parameters.get('path')
            project_id = context.parameters.get('project_id')
            project_name = context.parameters.get('project_name')
            commit = context.parameters.get('commit', True)
            detect_boundaries = context.parameters.get('detect_boundaries', True)
            resolver = context.parameters.get('resolver')

            # Call the existing implementation for backward compatibility
            await do_ingest(
                console=context.console,
                config=context.config,
                llm_gateway=context.llm_gateway,
                path=path,
                project_id=project_id,
                project_name=project_name,
                commit=commit,
                detect_boundaries=detect_boundaries,
                resolver=resolver,
                knowledge_store=self.knowledge_store,
                progress_tracker=context.parameters.get('progress_tracker')
            )

            return CommandResult(
                success=True,
                message="Content ingested successfully",
                data={
                    'path': str(path) if path else None,
                    'project_id': project_id,
                    'project_name': project_name
                }
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to ingest content: {str(e)}",
                error=e
            )


async def _read_files_parallel(file_paths: list[Path], input_path: Path, console, progress_tracker: ProgressTracker) -> list[tuple[Path, str]]:
    """
    Read multiple files concurrently for better performance.

    Args:
        file_paths: List of file paths to read
        input_path: Base input path for relative path calculation
        console: Rich console for progress reporting

    Returns:
        List of tuples containing (file_path, file_content) for successfully read files
    """
    completed_files = 0

    async def read_file(file_path: Path) -> tuple[Path, str | None]:
        """Read a single file asynchronously with error handling."""
        nonlocal completed_files
        try:
            # Display progress for each file
            relative_name = file_path.relative_to(input_path) if input_path.is_dir() else file_path.name

            # Use asyncio.to_thread to run the blocking read_text() in a thread pool
            content = await asyncio.to_thread(file_path.read_text)

            # Update progress
            completed_files += 1
            progress_tracker.update_progress("file_reading", completed_files, f"Read {relative_name}")

            return file_path, content
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {str(e)}[/yellow]")
            completed_files += 1
            progress_tracker.update_progress("file_reading", completed_files, f"Failed to read {file_path.name}")
            return file_path, None

    # Create tasks for all files and run them concurrently
    tasks = [read_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks)

    # Filter out failed reads (where content is None)
    successful_reads = [(path, content) for path, content in results if content is not None]

    return successful_reads


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


def apply_conflict_resolutions(
    resolved_conflicts, 
    valid_chunks, 
    existing_chunks_to_store
):
    """
    Pure function to apply conflict resolutions by creating new chunks with resolved content
    and excluding conflicting chunks from storage.

    Args:
        resolved_conflicts: List of Conflict objects with resolutions
        valid_chunks: List of all valid chunks from ingestion
        existing_chunks_to_store: List of chunks already marked for storage (no conflicts)

    Returns:
        Tuple of (filtered_existing_chunks, additional_chunks_to_store, resolution_messages)
        where:
        - filtered_existing_chunks: existing chunks with conflicting ones removed
        - additional_chunks_to_store: new chunks to store (resolved chunks + non-conflicting ones)
        - resolution_messages: list of messages about created resolved chunks
    """
    chunks_to_add = []
    conflicting_chunk_contents = set()
    resolution_messages = []

    # Collect all conflicting chunk contents to exclude them
    for conflict in resolved_conflicts:
        for guidance in conflict.conflicting_guidance:
            conflicting_chunk_contents.add(guidance.content.strip())

    # Filter out conflicting chunks from existing chunks to store
    filtered_existing_chunks = [chunk for chunk in existing_chunks_to_store 
                               if chunk.content.strip() not in conflicting_chunk_contents]

    # For each resolved conflict, create a new chunk with the resolved content
    for i, conflict in enumerate(resolved_conflicts):
        if hasattr(conflict, 'resolution') and conflict.resolution:
            # Find one of the original conflicting chunks to use as a template
            template_chunk = None
            for chunk in valid_chunks:
                if chunk.content.strip() in conflicting_chunk_contents:
                    template_chunk = chunk
                    break

            if template_chunk:
                # Create a new chunk with the resolved content
                # Use the template chunk's metadata but update the content
                # Make the ID unique by including the conflict index
                resolved_chunk = KnowledgeChunk(
                    id=f"{template_chunk.id}_resolved_{i}",
                    content=conflict.resolution,
                    metadata=template_chunk.metadata,
                    embedding=None  # Will be generated when stored
                )
                chunks_to_add.append(resolved_chunk)
                resolution_messages.append(f"Created resolved chunk for conflict: {conflict.description[:50]}...")

    # Add all non-conflicting chunks that aren't already in existing_chunks_to_store
    for chunk in valid_chunks:
        if (chunk.content.strip() not in conflicting_chunk_contents and 
            chunk not in existing_chunks_to_store):
            chunks_to_add.append(chunk)

    return filtered_existing_chunks, chunks_to_add, resolution_messages


def _apply_conflict_resolutions(resolved_conflicts, valid_chunks, chunks_to_store, console):
    """
    Apply conflict resolutions by creating new chunks with resolved content
    and excluding conflicting chunks from storage.

    Args:
        resolved_conflicts: List of Conflict objects with resolutions
        valid_chunks: List of all valid chunks from ingestion
        chunks_to_store: List of chunks already marked for storage (no conflicts)
        console: Rich console for output

    Returns:
        List of additional chunks to store (resolved chunks, excluding conflicting ones)
    """
    # Use the pure function to do the actual work
    filtered_existing_chunks, chunks_to_add, resolution_messages = apply_conflict_resolutions(
        resolved_conflicts, valid_chunks, chunks_to_store
    )

    # Update the chunks_to_store list in place (maintaining original behavior)
    chunks_to_store[:] = filtered_existing_chunks

    # Print resolution messages
    for message in resolution_messages:
        console.print(f"[green]{message}[/green]")

    return chunks_to_add


async def do_ingest(console, config: Config, llm_gateway: LLMGateway, path: Path=None, project_id: str=None, project_name: str=None, commit: bool=True, detect_boundaries: bool=True, resolver: ConflictResolver=None, knowledge_store: KnowledgeStore=None, progress_tracker: ProgressTracker=None):
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
        resolver: Optional automated conflict resolver. If provided, conflicts will be
                 resolved automatically without user input.
        knowledge_store: Optional injected knowledge store. If not provided, will create
                        a vector store using the factory pattern.
        progress_tracker: Optional progress tracker for showing progress indicators.
                         If not provided, will use a no-op tracker.
    """
    # Disable tokenizer parallelism to avoid warnings when using async operations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize progress tracker if not provided
    if progress_tracker is None:
        progress_tracker = ProgressTracker(NoOpProgressObserver())

    # Initialize timing collector for performance monitoring
    timing_collector = TimingCollector()

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

        # Read files in parallel for better performance
        with time_operation("file_reading", timing_collector) as file_timing:
            progress_tracker.start_operation("file_reading", "Reading files", len(files_to_process))
            file_contents = await _read_files_parallel(files_to_process, input_path, console, progress_tracker)
            progress_tracker.complete_operation("file_reading")

        console.print(f"[dim]📁 File reading completed in {format_duration(file_timing.duration_seconds)}[/dim]")

        # Process the successfully read files
        for file_path, file_content in file_contents:
            # For single files, preserve original behavior (no header)
            # For multiple files, add headers to distinguish content
            if len(files_to_process) == 1:
                all_content.append(file_content)
            else:
                all_content.append(f"# Content from {file_path}\n\n{file_content}")

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
            with time_operation("chunk_creation", timing_collector) as chunk_timing:
                progress_tracker.start_operation("chunking", "Creating chunks", 1)
                chunks = chunking_engine.chunk_by_structured_output(
                    ingest_content, 
                    source=str(path),
                    project_id=project_id,
                    project_name=project_name,
                    project_path=determined_project_path
                )
                progress_tracker.complete_operation("chunking")

            console.print(f"[dim]🧩 Chunk creation completed in {format_duration(chunk_timing.duration_seconds)}[/dim]")

            # Display chunking results with validation status
            if chunks:
                # Validate all chunks first
                with time_operation("chunk_validation", timing_collector) as validation_timing:
                    chunk_validations = []
                    valid_chunks = []

                    progress_tracker.start_operation("validation", "Validating chunks", len(chunks))
                    for i, chunk in enumerate(chunks):
                        validation_result = chunking_engine.validate_chunk_completeness(chunk)
                        chunk_validations.append((chunk, validation_result))
                        if validation_result.is_complete:
                            valid_chunks.append(chunk)
                        progress_tracker.update_progress("validation", i + 1, f"Validated chunk {i + 1}/{len(chunks)}")
                    progress_tracker.complete_operation("validation")

                console.print(f"[dim]✅ Chunk validation completed in {format_duration(validation_timing.duration_seconds)}[/dim]")

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
                # Use injected knowledge store if provided, otherwise create one (for backward compatibility)
                if knowledge_store is None:
                    vector_store_path = config.library_path / "vector_store"
                    knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path, llm_gateway)

                # Check for conflicts before storing chunks
                all_conflicts = []
                chunks_to_store = []

                # First, check for conflicts between chunks within the same ingestion batch
                console.print("[blue]Checking for conflicts between new chunks...[/blue]")
                with time_operation("conflict_detection_internal", timing_collector) as internal_conflict_timing:
                    # Create pairs of chunks to check for conflicts
                    chunk_pairs = []
                    for i, chunk1 in enumerate(valid_chunks):
                        for j, chunk2 in enumerate(valid_chunks[i+1:], i+1):
                            chunk_pairs.append((chunk1, chunk2))

                    if chunk_pairs:
                        progress_tracker.start_operation("internal_conflicts", "Checking internal conflicts", len(chunk_pairs))
                        # Use batch conflict detection for improved performance
                        from context_mixer.commands.operations.merge import detect_conflicts_batch

                        # Get batch size from config or use default of 5
                        batch_size = getattr(config, 'conflict_detection_batch_size', 5)
                        console.print(f"[dim]Processing {len(chunk_pairs)} chunk pairs in batches of {batch_size}[/dim]")

                        try:
                            batch_results = await detect_conflicts_batch(chunk_pairs, llm_gateway, batch_size)

                            for chunk1, chunk2, conflicts in batch_results:
                                if conflicts.list:
                                    console.print(f"[yellow]Detected conflict between chunks {chunk1.id[:12]}... and {chunk2.id[:12]}...[/yellow]")
                                    for conflict in conflicts.list:
                                        # Update the conflict to indicate it's between new chunks
                                        updated_conflict = Conflict(
                                            description=f"Conflicting guidance detected between new chunks: {conflict.description}",
                                            conflicting_guidance=[
                                                ConflictingGuidance(content=chunk1.content, source=f"new chunk {chunk1.id[:12]}..."),
                                                ConflictingGuidance(content=chunk2.content, source=f"new chunk {chunk2.id[:12]}...")
                                            ]
                                        )
                                        all_conflicts.append(updated_conflict)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Batch conflict detection failed, falling back to sequential processing: {str(e)}[/yellow]")
                            # Fallback to sequential processing if batch processing fails
                            for chunk1, chunk2 in chunk_pairs:
                                try:
                                    from context_mixer.commands.operations.merge import detect_conflicts
                                    conflicts = detect_conflicts(chunk1.content, chunk2.content, llm_gateway)
                                    if conflicts.list:
                                        console.print(f"[yellow]Detected conflict between chunks {chunk1.id[:12]}... and {chunk2.id[:12]}...[/yellow]")
                                        for conflict in conflicts.list:
                                            updated_conflict = Conflict(
                                                description=f"Conflicting guidance detected between new chunks: {conflict.description}",
                                                conflicting_guidance=[
                                                    ConflictingGuidance(content=chunk1.content, source=f"new chunk {chunk1.id[:12]}..."),
                                                    ConflictingGuidance(content=chunk2.content, source=f"new chunk {chunk2.id[:12]}...")
                                                ]
                                            )
                                            all_conflicts.append(updated_conflict)
                                except Exception as inner_e:
                                    console.print(f"[yellow]Warning: Failed to check conflicts between chunks {chunk1.id[:12]}... and {chunk2.id[:12]}...: {str(inner_e)}[/yellow]")

                    if chunk_pairs:
                        progress_tracker.complete_operation("internal_conflicts")

                console.print(f"[dim]🔍 Internal conflict detection completed in {format_duration(internal_conflict_timing.duration_seconds)}[/dim]")

                # Then, check for conflicts with existing chunks in the knowledge store
                console.print("[blue]Checking for conflicts with existing content...[/blue]")
                with time_operation("conflict_detection_external", timing_collector) as external_conflict_timing:
                    progress_tracker.start_operation("external_conflicts", "Checking external conflicts", len(valid_chunks))
                    for i, chunk in enumerate(valid_chunks):
                        try:
                            conflicting_chunks = await knowledge_store.detect_conflicts(chunk)
                            if conflicting_chunks:
                                # Create conflict objects for user resolution
                                for conflicting_chunk in conflicting_chunks:
                                    conflict = Conflict(
                                        description=f"Conflicting guidance detected between new content and existing content",
                                        conflicting_guidance=[
                                            ConflictingGuidance(content=chunk.content, source="new"),
                                            ConflictingGuidance(content=conflicting_chunk.content, source="existing")
                                        ]
                                    )
                                    all_conflicts.append(conflict)
                            else:
                                # No conflicts, safe to store
                                chunks_to_store.append(chunk)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Failed to check conflicts for chunk {chunk.id[:12]}...: {str(e)}[/yellow]")
                            # If conflict detection fails, store the chunk anyway
                            chunks_to_store.append(chunk)

                        progress_tracker.update_progress("external_conflicts", i + 1, f"Checked chunk {i + 1}/{len(valid_chunks)}")

                    progress_tracker.complete_operation("external_conflicts")

                console.print(f"[dim]🔍 External conflict detection completed in {format_duration(external_conflict_timing.duration_seconds)}[/dim]")

                # Handle conflicts if any were detected
                if all_conflicts:
                    console.print(f"\n[yellow]Detected {len(all_conflicts)} conflict(s) that require resolution[/yellow]")
                    with time_operation("conflict_resolution", timing_collector) as resolution_timing:
                        resolved_conflicts = resolve_conflicts(all_conflicts, console, resolver)

                        # Apply conflict resolutions to modify chunks appropriately
                        chunks_to_store.extend(_apply_conflict_resolutions(resolved_conflicts, valid_chunks, chunks_to_store, console))

                    console.print(f"[dim]⚖️  Conflict resolution completed in {format_duration(resolution_timing.duration_seconds)}[/dim]")
                    console.print(f"[green]Conflicts resolved and applied. Proceeding with storage.[/green]")
                else:
                    # No conflicts detected, store all remaining chunks
                    chunks_to_store.extend([chunk for chunk in valid_chunks if chunk not in chunks_to_store])

                with time_operation("knowledge_store_operations", timing_collector) as store_timing:
                    try:
                        await knowledge_store.store_chunks(chunks_to_store)
                        console.print(f"[green]Successfully stored {len(chunks_to_store)} chunks in vector knowledge store[/green]")
                    except Exception as e:
                        console.print(f"[red]Error: Failed to store chunks in vector store: {str(e)}[/red]")
                        console.print("[red]This is a critical error - user input data could not be stored properly in the vector store![/red]")
                        console.print("[yellow]Attempting fallback to context.md storage to preserve your data[/yellow]")

                console.print(f"[dim]💾 Knowledge store operations completed in {format_duration(store_timing.duration_seconds)}[/dim]")

                # Create a summary for context.md (for compatibility) using ALL chunks in the store
                try:
                    all_chunks = await knowledge_store.get_all_chunks()
                    summary_content = f"# Knowledge Store Contents\n\n"
                    summary_content += f"Last updated: {chunking_engine._generate_chunk_id('', '')[:8]}\n"
                    summary_content += f"Total chunks: {len(all_chunks)}\n\n"

                    for i, chunk in enumerate(all_chunks, 1):
                        summary_content += f"## Chunk {i}: {chunk.metadata.tags[0] if chunk.metadata.tags else 'Concept'}\n"
                        summary_content += f"**Domain:** {', '.join(chunk.metadata.domains)}\n"
                        summary_content += f"**Authority:** {chunk.metadata.authority.value}\n"
                        summary_content += f"**ID:** {chunk.id}\n\n"
                        summary_content += chunk.content + "\n\n---\n\n"
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not retrieve all chunks for context.md: {str(e)}[/yellow]")
                    # Fallback to current ingestion chunks
                    summary_content = f"# Ingested Content from {path}\n\n"
                    summary_content += f"Processed on: {chunking_engine._generate_chunk_id('', '')[:8]}\n"
                    summary_content += f"Total chunks: {len(chunks_to_store)}\n\n"

                    for i, chunk in enumerate(chunks_to_store, 1):
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
            with time_operation("git_commit", timing_collector) as commit_timing:
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

            console.print(f"[dim]📝 Git commit completed in {format_duration(commit_timing.duration_seconds)}[/dim]")

        # Display timing summary
        total_time = timing_collector.get_total_duration()
        console.print(f"\n[bold dim]⏱️  Performance Summary (Total: {format_duration(total_time)})[/bold dim]")

        # Create timing summary table
        timing_table = Table(title="Operation Timing Details", show_header=True, header_style="bold blue")
        timing_table.add_column("Operation", style="cyan", width=25)
        timing_table.add_column("Duration", style="yellow", justify="right", width=12)
        timing_table.add_column("% of Total", style="dim", justify="right", width=10)

        for result in timing_collector.results:
            percentage = (result.duration_seconds / total_time * 100) if total_time > 0 else 0
            timing_table.add_row(
                result.operation_name.replace("_", " ").title(),
                format_duration(result.duration_seconds),
                f"{percentage:.1f}%"
            )

        console.print(timing_table)

    except Exception as e:
        console.print(f"[red]Error during ingestion: {str(e)}[/red]")
