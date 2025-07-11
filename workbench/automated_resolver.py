"""
Automated Conflict Resolver

This module provides automated conflict resolution for the workbench,
eliminating the need for user input during testing.
"""

import os
from typing import List, Optional
from rich.console import Console

from context_mixer.domain.conflict import Conflict


class AutomatedConflictResolver:
    """
    Automated conflict resolver that resolves conflicts without user input.

    This resolver uses predefined resolution strategies to automatically
    resolve conflicts during workbench testing.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize the automated resolver."""
        self.console = console or Console()

    def resolve_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """
        Automatically resolve conflicts using predefined strategies.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            List of resolved conflicts
        """
        resolved_conflicts = []

        for conflict in conflicts:
            # Display conflict details similar to interactive resolver
            self.console.print("\n[bold red]Conflict Detected![/bold red]")
            self.console.print(f"[bold]Description:[/bold] {conflict.description}")
            self.console.print("\n[bold]Conflicting Guidance:[/bold]")

            for i, guidance in enumerate(conflict.conflicting_guidance):
                self.console.print(f"\n[bold]{i + 1}. From {guidance.source}:[/bold]")
                self.console.print(f"{guidance.content}")

            self.console.print(f"\n[yellow]Auto-resolving conflict...[/yellow]")

            # Determine resolution strategy based on conflict content
            resolution = self._determine_resolution(conflict)

            if resolution:
                # Create a resolved conflict
                resolved_conflict = Conflict(
                    description=conflict.description,
                    conflicting_guidance=conflict.conflicting_guidance,
                    resolution=resolution
                )
                resolved_conflicts.append(resolved_conflict)
                self.console.print(f"[green]✓ Resolved with: {resolution}[/green]")
            else:
                self.console.print(f"[red]✗ Could not auto-resolve conflict[/red]")
                # Use the first guidance as fallback
                fallback_resolution = conflict.conflicting_guidance[0].content
                resolved_conflict = Conflict(
                    description=conflict.description,
                    conflicting_guidance=conflict.conflicting_guidance,
                    resolution=fallback_resolution
                )
                resolved_conflicts.append(resolved_conflict)
                self.console.print(f"[yellow]Using fallback: {fallback_resolution}[/yellow]")

        return resolved_conflicts

    def _determine_resolution(self, conflict: Conflict) -> Optional[str]:
        """
        Determine the appropriate resolution for a conflict using a generalized approach.

        This method uses a simple strategy that prefers existing content over new content.
        This is a general approach that works for any type of conflict without hardcoded
        rules for specific conflict types.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution string or None if no guidance available
        """
        # Prefer existing content over new content
        # This is a simple, general strategy that works for any conflict type
        if conflict.conflicting_guidance:
            # Look for existing content first
            for guidance in conflict.conflicting_guidance:
                if guidance.source == "existing":
                    return guidance.content.strip()

            # If no existing content found, use the first guidance
            return conflict.conflicting_guidance[0].content.strip()

        return None
