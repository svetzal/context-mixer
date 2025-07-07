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

        # Default resolution strategies
        self.resolution_strategies = {
            "indentation": "Use 4 spaces for indentation",
            "line_length": "Maximum line length is 80 characters",
            "quotes": "Use double quotes",
            "async": "Use async/await for database operations",
            "composition": "Prefer composition over inheritance",
        }

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
        Determine the appropriate resolution for a conflict.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution string or None if no strategy found
        """
        description_lower = conflict.description.lower()

        # Check for indentation conflicts
        if "indentation" in description_lower or "spaces" in description_lower:
            # Look for specific guidance in the conflicting content
            for guidance in conflict.conflicting_guidance:
                if "4 spaces" in guidance.content:
                    return "Use 4 spaces for indentation"
                elif "2 spaces" in guidance.content:
                    return "Use 2 spaces for indentation"
            return self.resolution_strategies.get("indentation")

        # Check for line length conflicts
        if "line length" in description_lower or "characters" in description_lower:
            # Look for specific guidance in the conflicting content
            for guidance in conflict.conflicting_guidance:
                if "80 characters" in guidance.content:
                    return "Maximum line length is 80 characters"
                elif "100 characters" in guidance.content:
                    return "Maximum line length is 100 characters"
            return self.resolution_strategies.get("line_length")

        # Check for async/sync conflicts
        if "async" in description_lower or "synchronous" in description_lower:
            return self.resolution_strategies.get("async")

        # Check for composition/inheritance conflicts
        if "composition" in description_lower or "inheritance" in description_lower:
            return self.resolution_strategies.get("composition")

        # Check for quote style conflicts
        if "quote" in description_lower:
            return self.resolution_strategies.get("quotes")

        # If no specific strategy found, try to extract a reasonable resolution
        # from the first piece of conflicting guidance
        if conflict.conflicting_guidance:
            return conflict.conflicting_guidance[0].content.strip()

        return None

    def add_resolution_strategy(self, key: str, resolution: str):
        """
        Add a new resolution strategy.

        Args:
            key: Strategy key (used for matching)
            resolution: Resolution text to use
        """
        self.resolution_strategies[key] = resolution
