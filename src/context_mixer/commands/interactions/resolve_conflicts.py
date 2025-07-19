from typing import List, Optional, Protocol

from context_mixer.domain.conflict import Conflict
from context_mixer.commands.interactions.conflict_resolution_strategies import (
    ConflictResolutionStrategy,
    UserInteractiveResolutionStrategy,
    ConflictResolutionContext
)


class ConflictResolver(Protocol):
    """Protocol for conflict resolvers."""

    def resolve_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """Resolve a list of conflicts."""
        ...


def resolve_conflicts(conflicts: List[Conflict], console, resolver: Optional[ConflictResolver] = None) -> List[Conflict]:
    """
    Resolve conflicts by consulting the user or using an automated resolver.

    This function can either present each conflict to the user for interactive resolution,
    or use an automated resolver for unattended operation.

    Args:
        conflicts: A list of Conflict objects to resolve
        console: Rich console for output
        resolver: Optional automated conflict resolver. If provided, conflicts will be
                 resolved automatically without user input.

    Returns:
        The list of conflicts with resolutions set
    """
    # If conflicts is empty, return it as is
    if not conflicts:
        return []

    # If an automated resolver is provided, use it
    if resolver is not None:
        return resolver.resolve_conflicts(conflicts)

    # Otherwise, use interactive resolution with the new strategy pattern
    interactive_strategy = UserInteractiveResolutionStrategy()
    context = ConflictResolutionContext(interactive_strategy)
    return context.resolve_conflicts(conflicts, console)
