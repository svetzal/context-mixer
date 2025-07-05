"""
Implementation of the assemble command for Context Mixer.

This module provides the functionality to assemble context fragments for specific AI assistants
using the CRAFT (Context-Aware Retrieval and Fusion Technology) system.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from context_mixer.config import Config
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory
from context_mixer.domain.knowledge import (
    SearchQuery, 
    AuthorityLevel, 
    GranularityLevel,
    TemporalScope
)


async def do_assemble(
    console: Console,
    config: Config,
    target: str,
    output: Optional[Path] = None,
    profile: Optional[str] = None,
    filter_tags: Optional[str] = None,
    token_budget: int = 8192,
    quality_threshold: float = 0.8
):
    """
    Assemble context fragments for a specific target AI assistant.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        target: Target AI assistant (e.g., 'copilot', 'claude', 'cursor')
        output: Optional output path for the assembled context
        profile: Optional LLM profile specification
        filter_tags: Optional filter string for tags (e.g., 'lang:python,layer:testing')
        token_budget: Maximum token budget for the assembled context
        quality_threshold: Minimum quality threshold for included chunks
    """
    # Disable tokenizer parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    console.print(
        Panel(f"Assembling context for target: [bold]{target}[/bold]", title="Context Mixer")
    )

    try:
        # Initialize vector knowledge store
        vector_store_path = config.library_path / "vector_store"
        if not vector_store_path.exists():
            console.print("[red]Error: No vector store found. Please run 'cmx ingest' first to populate the knowledge base.[/red]")
            return

        knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path)

        # Parse filter tags if provided
        domain_filters = []
        tag_filters = []
        if filter_tags:
            filters = filter_tags.split(',')
            for f in filters:
                if ':' in f:
                    key, value = f.split(':', 1)
                    if key.strip() == 'domain':
                        domain_filters.append(value.strip())
                    else:
                        tag_filters.append(f.strip())
                else:
                    tag_filters.append(f.strip())

        # Get all chunks with authority filtering (prefer higher authority)
        authority_levels = [AuthorityLevel.FOUNDATIONAL, AuthorityLevel.OFFICIAL, AuthorityLevel.CONVENTIONAL]
        chunks = await knowledge_store.get_chunks_by_authority(authority_levels)

        # Apply domain filtering if specified
        if domain_filters:
            domain_chunks = await knowledge_store.get_chunks_by_domain(domain_filters)
            # Intersect with authority-filtered chunks
            chunk_ids = {chunk.id for chunk in chunks}
            chunks = [chunk for chunk in domain_chunks if chunk.id in chunk_ids]

        # Apply tag filtering if specified
        if tag_filters:
            chunks = [
                chunk for chunk in chunks 
                if any(tag in chunk.metadata.tags for tag in tag_filters)
            ]

        if not chunks:
            console.print("[yellow]No chunks found matching the specified criteria.[/yellow]")
            return

        # Sort chunks by authority level and relevance
        chunks.sort(key=lambda c: (c.metadata.authority.value, -len(c.content)))

        # Assemble context based on target format
        if target.lower() == 'copilot':
            assembled_content = _assemble_for_copilot(chunks, token_budget, quality_threshold)
        elif target.lower() == 'claude':
            assembled_content = _assemble_for_claude(chunks, token_budget, quality_threshold)
        elif target.lower() == 'cursor':
            assembled_content = _assemble_for_cursor(chunks, token_budget, quality_threshold)
        else:
            # Generic format
            assembled_content = _assemble_generic(chunks, token_budget, quality_threshold)

        # Display assembly results
        _display_assembly_results(console, chunks, assembled_content, target, token_budget)

        # Output to file if specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(assembled_content)
            console.print(f"[green]Context assembled and saved to: {output_path}[/green]")
        else:
            # Output to target-specific location
            target_dir = config.library_path / "assembled" / target.lower()
            target_dir.mkdir(parents=True, exist_ok=True)

            if target.lower() == 'copilot':
                output_path = target_dir / "copilot-instructions.md"
            else:
                output_path = target_dir / f"{target.lower()}-context.md"

            output_path.write_text(assembled_content)
            console.print(f"[green]Context assembled and saved to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error during assembly: {str(e)}[/red]")


def _assemble_for_copilot(chunks: List, token_budget: int, quality_threshold: float) -> str:
    """
    Assemble chunks in GitHub Copilot instruction format.

    Args:
        chunks: List of KnowledgeChunk objects
        token_budget: Maximum token budget
        quality_threshold: Minimum quality threshold

    Returns:
        Assembled content formatted for GitHub Copilot
    """
    content = "# GitHub Copilot Instructions\n\n"
    content += "## Project Context\n\n"

    current_tokens = len(content.split())

    # Group chunks by domain for better organization
    domain_groups = {}
    for chunk in chunks:
        for domain in chunk.metadata.domains:
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(chunk)

    # Add chunks by domain
    for domain, domain_chunks in domain_groups.items():
        if current_tokens >= token_budget * 0.9:  # Leave 10% buffer
            break

        content += f"### {domain.title()} Guidelines\n\n"

        for chunk in domain_chunks:
            chunk_tokens = len(chunk.content.split())
            if current_tokens + chunk_tokens > token_budget * 0.9:
                break

            # Add authority indicator
            authority_indicator = {
                'foundational': '🏛️ FOUNDATIONAL',
                'official': '🔒 OFFICIAL',
                'conventional': '📋 CONVENTIONAL',
                'experimental': '🧪 EXPERIMENTAL',
                'deprecated': '⚠️ DEPRECATED'
            }.get(chunk.metadata.authority.value, '📝 STANDARD')

            content += f"#### {authority_indicator}: {chunk.metadata.tags[0] if chunk.metadata.tags else 'Guideline'}\n\n"
            content += chunk.content + "\n\n"
            current_tokens += chunk_tokens

    content += "\n---\n"
    content += f"*Generated by Context Mixer CRAFT system - {len(chunks)} chunks processed*\n"

    return content


def _assemble_for_claude(chunks: List, token_budget: int, quality_threshold: float) -> str:
    """
    Assemble chunks in Claude-optimized format.

    Args:
        chunks: List of KnowledgeChunk objects
        token_budget: Maximum token budget
        quality_threshold: Minimum quality threshold

    Returns:
        Assembled content formatted for Claude
    """
    content = "# Claude Context Instructions\n\n"
    content += "You are an AI assistant with the following project context:\n\n"

    current_tokens = len(content.split())

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = len(chunk.content.split())
        if current_tokens + chunk_tokens > token_budget * 0.9:
            break

        content += f"## Context {i}: {chunk.metadata.tags[0] if chunk.metadata.tags else 'Information'}\n\n"
        content += f"**Authority Level:** {chunk.metadata.authority.value.title()}\n"
        content += f"**Domains:** {', '.join(chunk.metadata.domains)}\n\n"
        content += chunk.content + "\n\n"
        current_tokens += chunk_tokens

    return content


def _assemble_for_cursor(chunks: List, token_budget: int, quality_threshold: float) -> str:
    """
    Assemble chunks in Cursor-optimized format.

    Args:
        chunks: List of KnowledgeChunk objects
        token_budget: Maximum token budget
        quality_threshold: Minimum quality threshold

    Returns:
        Assembled content formatted for Cursor
    """
    content = "# Cursor AI Context\n\n"
    content += "## Development Guidelines\n\n"

    current_tokens = len(content.split())

    for chunk in chunks:
        chunk_tokens = len(chunk.content.split())
        if current_tokens + chunk_tokens > token_budget * 0.9:
            break

        content += f"### {chunk.metadata.tags[0] if chunk.metadata.tags else 'Guideline'}\n\n"
        content += chunk.content + "\n\n"
        current_tokens += chunk_tokens

    return content


def _assemble_generic(chunks: List, token_budget: int, quality_threshold: float) -> str:
    """
    Assemble chunks in generic format.

    Args:
        chunks: List of KnowledgeChunk objects
        token_budget: Maximum token budget
        quality_threshold: Minimum quality threshold

    Returns:
        Assembled content in generic format
    """
    content = "# Assembled Context\n\n"

    current_tokens = len(content.split())

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = len(chunk.content.split())
        if current_tokens + chunk_tokens > token_budget * 0.9:
            break

        content += f"## Section {i}\n\n"
        content += f"**ID:** {chunk.id}\n"
        content += f"**Authority:** {chunk.metadata.authority.value}\n"
        content += f"**Domains:** {', '.join(chunk.metadata.domains)}\n"
        content += f"**Tags:** {', '.join(chunk.metadata.tags)}\n\n"
        content += chunk.content + "\n\n"
        current_tokens += chunk_tokens

    return content


def _display_assembly_results(console: Console, chunks: List, content: str, target: str, token_budget: int):
    """
    Display assembly results in a formatted table.

    Args:
        console: Rich console for output
        chunks: List of processed chunks
        content: Assembled content
        target: Target AI assistant
        token_budget: Token budget used
    """
    # Create results table
    table = Table(title=f"Assembly Results for {target.title()}")
    table.add_column("Metric", style="bold cyan", width=20)
    table.add_column("Value", style="green", width=30)

    content_tokens = len(content.split())
    utilization = (content_tokens / token_budget) * 100

    table.add_row("Chunks Processed", str(len(chunks)))
    table.add_row("Content Tokens", f"{content_tokens:,}")
    table.add_row("Token Budget", f"{token_budget:,}")
    table.add_row("Budget Utilization", f"{utilization:.1f}%")
    table.add_row("Target Format", target.title())

    console.print(table)

    # Show chunk breakdown
    if chunks:
        chunk_table = Table(title="Included Chunks")
        chunk_table.add_column("Authority", style="magenta", width=12)
        chunk_table.add_column("Domain", style="yellow", width=15)
        chunk_table.add_column("Tags", style="cyan", width=20)
        chunk_table.add_column("Size", style="blue", width=8)

        for chunk in chunks[:10]:  # Show first 10 chunks
            domains_str = ", ".join(chunk.metadata.domains[:2])  # Show first 2 domains
            tags_str = ", ".join(chunk.metadata.tags[:3])  # Show first 3 tags

            chunk_table.add_row(
                chunk.metadata.authority.value,
                domains_str,
                tags_str,
                f"{len(chunk.content)}"
            )

        if len(chunks) > 10:
            chunk_table.add_row("...", "...", "...", "...")

        console.print(chunk_table)
