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


async def do_ingest(console, config: Config, llm_gateway: LLMGateway, filename: Path=None, commit: bool=True, detect_boundaries: bool=True):
    """
    Ingest existing prompt artifacts into the library using intelligent chunking.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        llm_gateway: The LLM gateway to use for generating merged content and chunking
        filename: Path to the file to ingest
        commit: Whether to commit changes after ingestion
        detect_boundaries: Whether to use semantic boundary detection for chunking
    """
    # Disable tokenizer parallelism to avoid warnings when using async operations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    console.print(
        Panel(f"Ingesting prompts from: [bold]{filename}[/bold]", title="Context Mixer"))

    try:
        monolith_path = Path(filename)
        if not monolith_path.exists():
            console.print(f"[red]Error: Path {filename} does not exist[/red]")
            return

        ingest_content = monolith_path.read_text()

        if detect_boundaries:
            # Use ChunkingEngine for intelligent semantic boundary detection with structured output
            console.print("[blue]Analyzing content using structured LLM output for complete chunks...[/blue]")

            chunking_engine = ChunkingEngine(llm_gateway)
            chunks = chunking_engine.chunk_by_structured_output(ingest_content, source=str(filename))

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
                summary_content = f"# Ingested Content from {filename}\n\n"
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
