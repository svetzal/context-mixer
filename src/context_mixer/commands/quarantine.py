"""
Quarantine command for managing quarantined knowledge chunks.

This module provides CLI functionality for reviewing, resolving, and managing
knowledge chunks that have been quarantined due to conflicts or validation issues.
"""

import asyncio
from pathlib import Path
from typing import Optional, List

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.text import Text

from context_mixer.config import Config
from context_mixer.domain.knowledge_quarantine import (
    KnowledgeQuarantine,
    QuarantineReason,
    ResolutionAction,
    Resolution
)
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory


def do_quarantine_list(console: Console, 
                      config: Config,
                      reason_filter: Optional[str] = None,
                      resolved_filter: Optional[bool] = None,
                      priority_filter: Optional[int] = None,
                      project_filter: Optional[str] = None) -> None:
    """
    List quarantined chunks with optional filtering.
    
    Args:
        console: Rich console for output
        config: Application configuration
        reason_filter: Filter by quarantine reason
        resolved_filter: Filter by resolution status
        priority_filter: Filter by priority level
        project_filter: Filter by project ID
    """
    try:
        # Initialize quarantine system
        quarantine = KnowledgeQuarantine()
        
        # Convert string reason filter to enum if provided
        reason_enum = None
        if reason_filter:
            try:
                reason_enum = QuarantineReason(reason_filter.lower())
            except ValueError:
                console.print(f"[red]Invalid reason filter: {reason_filter}[/red]")
                console.print(f"Valid reasons: {', '.join([r.value for r in QuarantineReason])}")
                return
        
        # Get filtered quarantined chunks
        quarantined_chunks = quarantine.review_quarantined_chunks(
            reason_filter=reason_enum,
            resolved_filter=resolved_filter,
            priority_filter=priority_filter,
            project_filter=project_filter
        )
        
        if not quarantined_chunks:
            console.print("[yellow]No quarantined chunks found matching the specified filters.[/yellow]")
            return
        
        # Create table for displaying results
        table = Table(title="Quarantined Knowledge Chunks")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Reason", style="magenta")
        table.add_column("Priority", style="yellow", justify="center")
        table.add_column("Project", style="green")
        table.add_column("Age (days)", style="blue", justify="right")
        table.add_column("Status", style="red")
        table.add_column("Description", style="white")
        
        for chunk in quarantined_chunks:
            status = "✓ Resolved" if chunk.is_resolved() else "⚠ Unresolved"
            status_style = "green" if chunk.is_resolved() else "red"
            
            project_name = chunk.chunk.get_project_name() or chunk.chunk.get_project_id() or "N/A"
            
            table.add_row(
                chunk.id[:8] + "...",  # Truncated ID
                chunk.reason.value,
                str(chunk.priority),
                project_name,
                str(chunk.get_age_days()),
                Text(status, style=status_style),
                chunk.description[:50] + "..." if len(chunk.description) > 50 else chunk.description
            )
        
        console.print(table)
        
        # Show summary statistics
        stats = quarantine.get_quarantine_stats()
        console.print(f"\n[bold]Summary:[/bold] {stats['total_quarantined']} total, "
                     f"{stats['unresolved']} unresolved, {stats['resolved']} resolved")
        
    except Exception as e:
        console.print(f"[red]Error listing quarantined chunks: {str(e)}[/red]")


def do_quarantine_review(console: Console, 
                        config: Config,
                        chunk_id: str) -> None:
    """
    Review a specific quarantined chunk in detail.
    
    Args:
        console: Rich console for output
        config: Application configuration
        chunk_id: ID of the quarantined chunk to review
    """
    try:
        quarantine = KnowledgeQuarantine()
        quarantined_chunk = quarantine.get_quarantined_chunk(chunk_id)
        
        if not quarantined_chunk:
            console.print(f"[red]Quarantined chunk with ID '{chunk_id}' not found.[/red]")
            return
        
        # Display detailed information
        panel_content = []
        panel_content.append(f"[bold]Chunk ID:[/bold] {quarantined_chunk.id}")
        panel_content.append(f"[bold]Reason:[/bold] {quarantined_chunk.reason.value}")
        panel_content.append(f"[bold]Priority:[/bold] {quarantined_chunk.priority}")
        panel_content.append(f"[bold]Quarantined At:[/bold] {quarantined_chunk.quarantined_at}")
        panel_content.append(f"[bold]Age:[/bold] {quarantined_chunk.get_age_days()} days")
        
        if quarantined_chunk.quarantined_by:
            panel_content.append(f"[bold]Quarantined By:[/bold] {quarantined_chunk.quarantined_by}")
        
        if quarantined_chunk.conflicting_chunks:
            panel_content.append(f"[bold]Conflicting Chunks:[/bold] {', '.join(quarantined_chunk.conflicting_chunks)}")
        
        panel_content.append(f"\n[bold]Description:[/bold]\n{quarantined_chunk.description}")
        
        # Show chunk content
        panel_content.append(f"\n[bold]Chunk Content:[/bold]\n{quarantined_chunk.chunk.content}")
        
        # Show project information
        project_info = []
        if quarantined_chunk.chunk.get_project_id():
            project_info.append(f"ID: {quarantined_chunk.chunk.get_project_id()}")
        if quarantined_chunk.chunk.get_project_name():
            project_info.append(f"Name: {quarantined_chunk.chunk.get_project_name()}")
        
        if project_info:
            panel_content.append(f"\n[bold]Project:[/bold] {', '.join(project_info)}")
        
        # Show resolution if resolved
        if quarantined_chunk.is_resolved():
            resolution = quarantined_chunk.resolution
            panel_content.append(f"\n[bold green]RESOLVED[/bold green]")
            panel_content.append(f"[bold]Action:[/bold] {resolution.action.value}")
            panel_content.append(f"[bold]Reason:[/bold] {resolution.reason}")
            panel_content.append(f"[bold]Resolved At:[/bold] {resolution.resolved_at}")
            if resolution.resolved_by:
                panel_content.append(f"[bold]Resolved By:[/bold] {resolution.resolved_by}")
            if resolution.notes:
                panel_content.append(f"[bold]Notes:[/bold] {resolution.notes}")
        
        console.print(Panel("\n".join(panel_content), title="Quarantined Chunk Details", border_style="yellow"))
        
    except Exception as e:
        console.print(f"[red]Error reviewing quarantined chunk: {str(e)}[/red]")


def do_quarantine_resolve(console: Console, 
                         config: Config,
                         chunk_id: str,
                         action: str,
                         reason: str,
                         resolved_by: Optional[str] = None,
                         notes: Optional[str] = None) -> None:
    """
    Resolve a quarantined chunk with the specified action.
    
    Args:
        console: Rich console for output
        config: Application configuration
        chunk_id: ID of the quarantined chunk to resolve
        action: Resolution action (accept, reject, merge, modify, defer, escalate)
        reason: Reason for the resolution
        resolved_by: Who is resolving the quarantine
        notes: Additional notes about the resolution
    """
    try:
        quarantine = KnowledgeQuarantine()
        quarantined_chunk = quarantine.get_quarantined_chunk(chunk_id)
        
        if not quarantined_chunk:
            console.print(f"[red]Quarantined chunk with ID '{chunk_id}' not found.[/red]")
            return
        
        if quarantined_chunk.is_resolved():
            console.print(f"[yellow]Chunk '{chunk_id}' is already resolved.[/yellow]")
            return
        
        # Validate action
        try:
            resolution_action = ResolutionAction(action.lower())
        except ValueError:
            console.print(f"[red]Invalid action: {action}[/red]")
            console.print(f"Valid actions: {', '.join([a.value for a in ResolutionAction])}")
            return
        
        # Create resolution
        resolution = Resolution(
            action=resolution_action,
            reason=reason,
            resolved_by=resolved_by,
            notes=notes
        )
        
        # Apply resolution
        success = quarantine.resolve_quarantine(chunk_id, resolution)
        
        if success:
            console.print(f"[green]Successfully resolved quarantined chunk '{chunk_id}' with action '{action}'.[/green]")
            
            # Show what happens next based on action
            if resolution_action == ResolutionAction.ACCEPT:
                console.print("[blue]The chunk will be accepted and added to the knowledge store.[/blue]")
            elif resolution_action == ResolutionAction.REJECT:
                console.print("[blue]The chunk will be permanently rejected and removed.[/blue]")
            elif resolution_action == ResolutionAction.MODIFY:
                console.print("[blue]The chunk requires modification before acceptance.[/blue]")
            elif resolution_action == ResolutionAction.DEFER:
                console.print("[blue]Resolution has been deferred for later consideration.[/blue]")
            elif resolution_action == ResolutionAction.ESCALATE:
                console.print("[blue]The issue has been escalated to higher authority.[/blue]")
        else:
            console.print(f"[red]Failed to resolve quarantined chunk '{chunk_id}'.[/red]")
        
    except Exception as e:
        console.print(f"[red]Error resolving quarantined chunk: {str(e)}[/red]")


def do_quarantine_stats(console: Console, config: Config) -> None:
    """
    Display quarantine system statistics.
    
    Args:
        console: Rich console for output
        config: Application configuration
    """
    try:
        quarantine = KnowledgeQuarantine()
        stats = quarantine.get_quarantine_stats()
        
        # Create statistics table
        table = Table(title="Quarantine System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        
        table.add_row("Total Quarantined", str(stats["total_quarantined"]))
        table.add_row("Unresolved", str(stats["unresolved"]))
        table.add_row("Resolved", str(stats["resolved"]))
        table.add_row("Average Age (days)", str(stats["average_age_days"]))
        table.add_row("Oldest Quarantine (days)", str(stats["oldest_quarantine_days"]))
        
        console.print(table)
        
        # Show breakdown by reason
        if stats["reason_breakdown"]:
            console.print("\n[bold]Breakdown by Reason:[/bold]")
            reason_table = Table()
            reason_table.add_column("Reason", style="magenta")
            reason_table.add_column("Count", style="yellow", justify="right")
            
            for reason, count in stats["reason_breakdown"].items():
                reason_table.add_row(reason.replace("_", " ").title(), str(count))
            
            console.print(reason_table)
        
        # Show breakdown by priority (unresolved only)
        if stats["priority_breakdown"]:
            console.print("\n[bold]Unresolved by Priority:[/bold]")
            priority_table = Table()
            priority_table.add_column("Priority", style="blue")
            priority_table.add_column("Count", style="yellow", justify="right")
            
            for priority, count in sorted(stats["priority_breakdown"].items()):
                priority_table.add_row(str(priority), str(count))
            
            console.print(priority_table)
        
        # Show high priority items
        high_priority = quarantine.get_high_priority_unresolved()
        if high_priority:
            console.print(f"\n[bold red]⚠ {len(high_priority)} high-priority unresolved items require attention![/bold red]")
        
    except Exception as e:
        console.print(f"[red]Error getting quarantine statistics: {str(e)}[/red]")


def do_quarantine_clear(console: Console, config: Config) -> None:
    """
    Clear all resolved quarantined chunks from the system.
    
    Args:
        console: Rich console for output
        config: Application configuration
    """
    try:
        quarantine = KnowledgeQuarantine()
        stats = quarantine.get_quarantine_stats()
        
        if stats["resolved"] == 0:
            console.print("[yellow]No resolved quarantined chunks to clear.[/yellow]")
            return
        
        # Confirm before clearing
        if not Confirm.ask(f"Clear {stats['resolved']} resolved quarantined chunks?"):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
        
        cleared_count = quarantine.clear_resolved_chunks()
        console.print(f"[green]Successfully cleared {cleared_count} resolved quarantined chunks.[/green]")
        
    except Exception as e:
        console.print(f"[red]Error clearing resolved chunks: {str(e)}[/red]")