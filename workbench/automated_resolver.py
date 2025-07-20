"""
Automated Conflict Resolver

This module provides automated conflict resolution for the workbench,
eliminating the need for user input during testing.
"""

from typing import List, Optional

from rich.console import Console

from context_mixer.commands.interactions.conflict_resolution_strategies import (
    ConflictResolutionStrategy,
    AutomaticResolutionStrategy,
    LLMBasedResolutionStrategy,
    ConflictResolutionContext,
    ConflictResolutionStrategyFactory
)
from context_mixer.domain.conflict import Conflict


class AutomatedConflictResolver:
    """
    Automated conflict resolver that resolves conflicts without user input.

    This resolver uses the Strategy pattern to automatically select and apply
    appropriate resolution strategies based on conflict characteristics.
    """

    def __init__(self, console: Optional[Console] = None, default_strategy: str = "automatic"):
        """
        Initialize the automated resolver.

        Args:
            console: Console for output
            default_strategy: Default strategy to use ("automatic", "llm", or specific strategy name)
        """
        self.console = console or Console()
        self.default_strategy = default_strategy
        self.context = ConflictResolutionContext(
            ConflictResolutionStrategyFactory.create_strategy(default_strategy)
        )

    def resolve_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """
        Automatically resolve conflicts using intelligent strategy selection.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            List of resolved conflicts
        """
        if not conflicts:
            return []

        # Group conflicts by type for batch processing with appropriate strategies
        conflict_groups = self._group_conflicts_by_characteristics(conflicts)

        all_resolved_conflicts = []

        for group_type, group_conflicts in conflict_groups.items():
            # Select the best strategy for this group of conflicts
            strategy = self._select_strategy_for_group(group_type, group_conflicts)

            # Update context with the selected strategy
            self.context.set_strategy(strategy)

            self.console.print(f"\n[blue]Processing {len(group_conflicts)} conflicts using {strategy.get_strategy_name()} strategy[/blue]")

            # Resolve conflicts using the selected strategy
            resolved_group = self.context.resolve_conflicts(group_conflicts, self.console)
            all_resolved_conflicts.extend(resolved_group)

        return all_resolved_conflicts

    def _group_conflicts_by_characteristics(self, conflicts: List[Conflict]) -> dict:
        """
        Group conflicts by their characteristics for intelligent strategy selection.

        Args:
            conflicts: List of conflicts to group

        Returns:
            Dictionary mapping group types to lists of conflicts
        """
        groups = {
            "simple": [],      # Simple conflicts with clear existing vs new patterns
            "complex": [],     # Complex conflicts that might benefit from LLM analysis
            "style": [],       # Style-related conflicts (indentation, naming, etc.)
            "default": []      # Fallback group
        }

        for conflict in conflicts:
            group_type = self._classify_conflict(conflict)
            groups[group_type].append(conflict)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _classify_conflict(self, conflict: Conflict) -> str:
        """
        Classify a conflict based on its characteristics.

        Args:
            conflict: The conflict to classify

        Returns:
            Group type string
        """
        description_lower = conflict.description.lower()

        # Check for style-related conflicts
        style_keywords = ["indentation", "spacing", "naming", "case", "format", "style"]
        if any(keyword in description_lower for keyword in style_keywords):
            return "style"

        # Check for simple conflicts (existing vs new pattern)
        sources = [guidance.source for guidance in conflict.conflicting_guidance]
        if "existing" in sources and len(conflict.conflicting_guidance) == 2:
            return "simple"

        # Check for complex conflicts that might benefit from LLM analysis
        if len(conflict.conflicting_guidance) > 2 or len(conflict.description) > 100:
            return "complex"

        return "default"

    def _select_strategy_for_group(self, group_type: str, conflicts: List[Conflict]) -> ConflictResolutionStrategy:
        """
        Select the most appropriate strategy for a group of conflicts.

        Args:
            group_type: The type of conflict group
            conflicts: The conflicts in the group

        Returns:
            ConflictResolutionStrategy instance
        """
        if group_type == "simple":
            # For simple conflicts, prefer existing content
            return AutomaticResolutionStrategy(prefer_existing=True, fallback_to_first=True)

        elif group_type == "style":
            # For style conflicts, use automatic resolution with specific preferences
            return AutomaticResolutionStrategy(prefer_existing=True, fallback_to_first=True)

        elif group_type == "complex":
            # For complex conflicts, try LLM-based resolution if available
            # This will fallback to automatic if no LLM gateway is available
            return LLMBasedResolutionStrategy()

        else:  # default
            # Use the default strategy specified in constructor
            return ConflictResolutionStrategyFactory.create_strategy(self.default_strategy)

    def set_strategy(self, strategy: ConflictResolutionStrategy):
        """
        Set a specific strategy to use for all conflicts.

        Args:
            strategy: The strategy to use
        """
        self.context.set_strategy(strategy)

    def set_strategy_by_name(self, strategy_name: str, **kwargs):
        """
        Set a strategy by name.

        Args:
            strategy_name: Name of the strategy ("automatic", "llm", "interactive")
            **kwargs: Additional arguments for strategy initialization
        """
        strategy = ConflictResolutionStrategyFactory.create_strategy(strategy_name, **kwargs)
        self.context.set_strategy(strategy)
