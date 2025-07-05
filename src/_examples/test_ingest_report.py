#!/usr/bin/env python3
"""
Test script to verify the new ingest reporting functionality.

This script tests the enhanced ingest command that shows chunks and rejection reasons
side by side in a comprehensive table.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.gateways.llm import LLMGateway
from rich.console import Console


def test_ingest_reporting():
    """Test the ingest reporting functionality with various content types."""

    console = Console()

    # Initialize LLM gateway (using a local model for testing)
    llm_gateway = LLMGateway(model="qwen3:32b")
    chunking_engine = ChunkingEngine(llm_gateway)

    # Test content that should produce both valid and invalid chunks
    test_content = """# Complete Section
This is a complete section with proper content that should validate successfully.
It contains enough information to be meaningful and has a clear beginning and end.

## Another Complete Section
This section also contains complete information about a specific topic.
It provides context and detailed explanation that makes it self-contained.

# Incomplete Section
This section starts but then gets cut off mid-sentence and doesn't provide...

# Very Short
Too short.

# Empty Section


# Truncated List
Here are the important steps:
1. First step
2. Second step
3. Third step and

# Section with Ellipsis
This section contains some information but then trails off...
"""

    console.print("[blue]Testing ingest reporting with mixed content...[/blue]")

    try:
        # Chunk the content
        chunks = chunking_engine.chunk_by_structured_output(test_content, source="test")

        if chunks:
            # Validate all chunks first (mimicking the ingest command logic)
            chunk_validations = []
            valid_chunks = []

            for chunk in chunks:
                validation_result = chunking_engine.validate_chunk_completeness(chunk)
                chunk_validations.append((chunk, validation_result))
                if validation_result.is_complete:
                    valid_chunks.append(chunk)

            # Create the same comprehensive report table as in ingest command
            from rich.table import Table

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

            # Show some details about rejected chunks
            rejected_chunks = [chunk for chunk, validation in chunk_validations if not validation.is_complete]
            if rejected_chunks:
                console.print(f"\n[yellow]Rejected chunks details:[/yellow]")
                for chunk, validation in chunk_validations:
                    if not validation.is_complete:
                        console.print(f"[red]• {chunk.metadata.tags[0] if chunk.metadata.tags else 'Unknown'}[/red]: {validation.reason}")

        else:
            console.print("[yellow]No chunks were generated from the test content[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during testing: {str(e)}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ingest_reporting()
