from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from context_mixer.commands.operations.merge import merge_content
from context_mixer.commands.operations.commit import CommitOperation
from context_mixer.config import Config, DEFAULT_ROOT_CONTEXT_FILENAME

from context_mixer.gateways.llm import LLMGateway
from context_mixer.gateways.git import GitGateway
from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.domain.knowledge_store import KnowledgeStore


def do_ingest(console, config: Config, llm_gateway: LLMGateway, filename: Path=None, commit: bool=True, detect_boundaries: bool=True):
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
    console.print(
        Panel(f"Ingesting prompts from: [bold]{filename}[/bold]", title="Context Mixer"))

    try:
        monolith_path = Path(filename)
        if not monolith_path.exists():
            console.print(f"[red]Error: Path {filename} does not exist[/red]")
            return

        ingest_content = monolith_path.read_text()

        if detect_boundaries:
            # Use ChunkingEngine for intelligent semantic boundary detection
            console.print("[blue]Analyzing content for semantic boundaries...[/blue]")

            chunking_engine = ChunkingEngine(llm_gateway)
            chunks = chunking_engine.chunk_by_concepts(ingest_content, source=str(filename))

            # Display chunking results
            if chunks:
                table = Table(title="Detected Knowledge Chunks")
                table.add_column("Chunk ID", style="cyan")
                table.add_column("Concept", style="green")
                table.add_column("Domain", style="yellow")
                table.add_column("Authority", style="magenta")
                table.add_column("Size", style="blue")

                for chunk in chunks:
                    domains_str = ", ".join(chunk.metadata.domains)
                    table.add_row(
                        chunk.id[:12] + "...",
                        chunk.metadata.tags[0] if chunk.metadata.tags else "Unknown",
                        domains_str,
                        chunk.metadata.authority.value,
                        f"{len(chunk.content)} chars"
                    )

                console.print(table)
                console.print(f"[green]Successfully chunked content into {len(chunks)} semantic chunks[/green]")

                # Validate chunks
                valid_chunks = []
                for chunk in chunks:
                    validation_result = chunking_engine.validate_chunk_completeness(chunk)
                    if validation_result.is_complete:
                        valid_chunks.append(chunk)
                    else:
                        console.print(f"[yellow]Warning: Chunk {chunk.id[:12]}... appears incomplete[/yellow]")
                        console.print(f"[dim]  Reason: {validation_result.reason}[/dim]")
                        if validation_result.issues:
                            console.print(f"[dim]  Issues: {', '.join(validation_result.issues)}[/dim]")
                        console.print(f"[dim]  Confidence: {validation_result.confidence:.2f}[/dim]")

                console.print(f"[blue]{len(valid_chunks)} of {len(chunks)} chunks validated as complete[/blue]")

                # Store chunks in knowledge store (for now, also save to context.md for compatibility)
                # TODO: Integrate with actual knowledge store when available

                # Create a summary for context.md
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
