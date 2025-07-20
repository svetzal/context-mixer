"""
Conflict Resolution Strategies

This module implements the Strategy pattern for conflict resolution,
providing multiple approaches to resolve conflicts between guidance.
"""

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
from rich.console import Console

from context_mixer.domain.conflict import Conflict


class ConflictResolutionStrategy(ABC):
    """
    Abstract base class for conflict resolution strategies.

    This defines the interface that all concrete resolution strategies must implement.
    """

    @abstractmethod
    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Resolve a list of conflicts using this strategy.

        Args:
            conflicts: List of conflicts to resolve
            console: Optional console for output (used by interactive strategies)

        Returns:
            List of resolved conflicts
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass


class UserInteractiveResolutionStrategy(ConflictResolutionStrategy):
    """
    Interactive conflict resolution strategy that prompts the user for decisions.

    This strategy presents each conflict to the user and allows them to choose
    from the available options or provide a custom resolution.
    """

    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Resolve conflicts by consulting the user interactively.

        Args:
            conflicts: List of conflicts to resolve
            console: Console for user interaction

        Returns:
            List of resolved conflicts
        """
        if console is None:
            console = Console()

        if not conflicts:
            return []

        resolved_conflicts = []

        for conflict in conflicts:
            resolved_conflict = self._resolve_single_conflict(conflict, console)
            resolved_conflicts.append(resolved_conflict)

        return resolved_conflicts

    def _resolve_single_conflict(self, conflict: Conflict, console: Console) -> Conflict:
        """Resolve a single conflict interactively."""
        console.print("\n[bold red]Conflict Detected![/bold red]")
        console.print(f"[bold]Description:[/bold] {conflict.description}")
        console.print("\n[bold]Conflicting Guidance:[/bold]")

        for i, guidance in enumerate(conflict.conflicting_guidance):
            console.print(f"\n[bold]{i + 1}. From {guidance.source}:[/bold]")
            console.print(f"{guidance.content}")

        # Present options to the user
        console.print("\n[bold]Options:[/bold]")
        for i, guidance in enumerate(conflict.conflicting_guidance):
            console.print(f"{i + 1}. Choose the guidance from {guidance.source}")
        console.print(f"{len(conflict.conflicting_guidance) + 1}. Enter your own resolution")
        console.print(f"{len(conflict.conflicting_guidance) + 2}. This is not a conflict")

        # Ask the user to choose an option
        while True:
            choice = console.input("\n[bold]Select an option (Enter the number):[/bold] ")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(conflict.conflicting_guidance):
                    # Set the resolution to the chosen guidance
                    resolution = conflict.conflicting_guidance[choice_idx].content
                    break
                elif choice_idx == len(conflict.conflicting_guidance):
                    # User wants to enter their own resolution
                    resolution = self._get_multiline_input(console, conflict)
                    break
                elif choice_idx == len(conflict.conflicting_guidance) + 1:
                    # User indicates this is not a conflict
                    resolution = None
                    break
                else:
                    console.print("[red]Invalid choice. Please enter a valid number.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")

        # Return a new conflict with the resolution
        return Conflict(
            description=conflict.description,
            conflicting_guidance=conflict.conflicting_guidance,
            resolution=resolution
        )

    def _get_multiline_input(self, console: Console, conflict: Conflict) -> str:
        """
        Get multi-line input from the user.

        If the EDITOR environment variable is set, creates a temporary file,
        launches the editor, and reads the result. Otherwise, falls back to
        terminal input terminated by a single '.' on a line by itself.

        Args:
            console: Console for user interaction
            conflict: The conflict being resolved (used to populate editor with both chunks)

        Returns:
            The multi-line input as a single string with newlines preserved
        """
        editor = os.environ.get('EDITOR')

        if editor:
            return self._get_input_via_editor(console, editor, conflict)
        else:
            return self._get_input_via_terminal(console)

    def _get_input_via_editor(self, console: Console, editor: str, conflict: Conflict) -> str:
        """
        Get input via external editor.

        Args:
            console: Console for user interaction
            editor: Editor command to use
            conflict: The conflict being resolved (used to populate editor with both chunks)

        Returns:
            The content from the editor
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
            temp_file_path = temp_file.name
            # Write initial content to help the user
            temp_file.write("# Enter your conflict resolution below.\n")
            temp_file.write("# You can edit, delete, or combine the conflicting guidance shown below.\n")
            temp_file.write("# Lines starting with # will be ignored.\n")
            temp_file.write(f"# Conflict: {conflict.description}\n\n")

            # Write both conflicting guidance chunks
            for i, guidance in enumerate(conflict.conflicting_guidance):
                temp_file.write(f"# === Option {i + 1}: From {guidance.source} ===\n")
                temp_file.write(f"{guidance.content}\n\n")

            temp_file.write("# === End of conflicting guidance ===\n")
            temp_file.write("# Edit the content above to create your resolution.\n\n")

        try:
            console.print(f"\n[bold]Opening editor ({editor}) for conflict resolution...[/bold]")

            # Launch the editor
            result = subprocess.run([editor, temp_file_path], check=True)

            # Read the content back
            with open(temp_file_path, 'r') as temp_file:
                content = temp_file.read()

            # Filter out comment lines and clean up
            lines = []
            for line in content.split('\n'):
                if not line.strip().startswith('#'):
                    lines.append(line)

            # Remove leading/trailing empty lines
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            return '\n'.join(lines)

        except subprocess.CalledProcessError:
            console.print("[red]Editor was cancelled or failed. Falling back to terminal input.[/red]")
            return self._get_input_via_terminal(console)
        except FileNotFoundError:
            console.print(f"[red]Editor '{editor}' not found. Falling back to terminal input.[/red]")
            return self._get_input_via_terminal(console)
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File might already be deleted

    def _get_input_via_terminal(self, console: Console) -> str:
        """
        Get input via terminal (original behavior).

        Args:
            console: Console for user interaction

        Returns:
            The multi-line input as a single string with newlines preserved
        """
        console.print("\n[bold]Enter your custom resolution (end with a single '.' on a line by itself):[/bold]")
        lines = []

        while True:
            line = console.input("")
            if line.strip() == ".":
                break
            lines.append(line)

        return "\n".join(lines)

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "UserInteractive"


class AutomaticResolutionStrategy(ConflictResolutionStrategy):
    """
    Automatic conflict resolution strategy that uses predefined rules.

    This strategy automatically resolves conflicts using simple heuristics
    without requiring user input.
    """

    def __init__(self, prefer_existing: bool = True, fallback_to_first: bool = True):
        """
        Initialize the automatic resolution strategy.

        Args:
            prefer_existing: If True, prefer guidance from "existing" sources
            fallback_to_first: If True, use first guidance as fallback when no preference applies
        """
        self.prefer_existing = prefer_existing
        self.fallback_to_first = fallback_to_first

    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Automatically resolve conflicts using predefined rules.

        Args:
            conflicts: List of conflicts to resolve
            console: Optional console for logging (if provided)

        Returns:
            List of resolved conflicts
        """
        if not conflicts:
            return []

        resolved_conflicts = []

        for conflict in conflicts:
            resolution = self._determine_resolution(conflict)

            resolved_conflict = Conflict(
                description=conflict.description,
                conflicting_guidance=conflict.conflicting_guidance,
                resolution=resolution
            )
            resolved_conflicts.append(resolved_conflict)

            if console:
                console.print(f"[yellow]Auto-resolved conflict: {conflict.description}[/yellow]")
                console.print(f"[green]Resolution: {resolution}[/green]")

        return resolved_conflicts

    def _determine_resolution(self, conflict: Conflict) -> str:
        """
        Determine the appropriate resolution for a conflict.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution string
        """
        if not conflict.conflicting_guidance:
            return ""

        # Strategy 1: Prefer existing content over new content
        if self.prefer_existing:
            for guidance in conflict.conflicting_guidance:
                if guidance.source == "existing":
                    return guidance.content.strip()

        # Strategy 2: Fallback to first guidance
        if self.fallback_to_first:
            return conflict.conflicting_guidance[0].content.strip()

        return ""

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "Automatic"


class LLMBasedResolutionStrategy(ConflictResolutionStrategy):
    """
    LLM-based conflict resolution strategy that uses AI to resolve conflicts.

    This strategy leverages a language model to analyze conflicts and provide
    intelligent resolutions based on context and best practices.
    """

    def __init__(self, llm_gateway=None):
        """
        Initialize the LLM-based resolution strategy.

        Args:
            llm_gateway: Gateway for LLM interactions
        """
        self.llm_gateway = llm_gateway

    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Resolve conflicts using LLM analysis.

        Args:
            conflicts: List of conflicts to resolve
            console: Optional console for logging

        Returns:
            List of resolved conflicts
        """
        if not conflicts:
            return []

        if self.llm_gateway is None:
            # Fallback to automatic resolution if no LLM gateway available
            if console:
                console.print("[yellow]No LLM gateway available, falling back to automatic resolution[/yellow]")
            fallback_strategy = AutomaticResolutionStrategy()
            return fallback_strategy.resolve_conflicts(conflicts, None)  # Don't pass console to avoid double logging

        resolved_conflicts = []

        for conflict in conflicts:
            resolution = self._resolve_with_llm(conflict, console)

            resolved_conflict = Conflict(
                description=conflict.description,
                conflicting_guidance=conflict.conflicting_guidance,
                resolution=resolution
            )
            resolved_conflicts.append(resolved_conflict)

        return resolved_conflicts

    def _resolve_with_llm(self, conflict: Conflict, console: Optional[Console] = None) -> str:
        """
        Resolve a single conflict using LLM analysis.

        Args:
            conflict: The conflict to resolve
            console: Optional console for logging

        Returns:
            Resolution string
        """
        try:
            # Prepare the prompt for the LLM
            prompt = self._build_resolution_prompt(conflict)

            # TODO: Implement actual LLM call when LLM gateway is available
            # For now, use a simple heuristic as placeholder
            if console:
                console.print(f"[blue]LLM analyzing conflict: {conflict.description}[/blue]")

            # Placeholder implementation - in real implementation, this would call the LLM
            # response = self.llm_gateway.generate_text(prompt)
            # return self._extract_resolution_from_response(response)

            # For now, fall back to automatic resolution
            auto_strategy = AutomaticResolutionStrategy()
            return auto_strategy._determine_resolution(conflict)

        except Exception as e:
            if console:
                console.print(f"[red]LLM resolution failed: {e}[/red]")
            # Fallback to automatic resolution
            auto_strategy = AutomaticResolutionStrategy()
            return auto_strategy._determine_resolution(conflict)

    def _build_resolution_prompt(self, conflict: Conflict) -> str:
        """Build a prompt for LLM-based conflict resolution."""
        prompt = f"""
You are an expert at resolving conflicts in software development guidance.

Conflict Description: {conflict.description}

Conflicting Guidance:
"""
        for i, guidance in enumerate(conflict.conflicting_guidance, 1):
            prompt += f"\n{i}. From {guidance.source}:\n{guidance.content}\n"

        prompt += """
Please analyze these conflicting pieces of guidance and provide the best resolution that:
1. Maintains consistency with software development best practices
2. Considers the context and intent of both pieces of guidance
3. Provides a clear, actionable resolution

Resolution:"""

        return prompt

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "LLMBased"


class ConflictResolutionContext:
    """
    Context class that manages conflict resolution strategy selection and execution.

    This class implements the Context part of the Strategy pattern, allowing
    runtime selection and switching of resolution strategies.
    """

    def __init__(self, strategy: ConflictResolutionStrategy):
        """
        Initialize the context with a specific strategy.

        Args:
            strategy: The initial resolution strategy to use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ConflictResolutionStrategy):
        """
        Change the resolution strategy at runtime.

        Args:
            strategy: The new strategy to use
        """
        self._strategy = strategy

    def get_strategy(self) -> ConflictResolutionStrategy:
        """Get the current strategy."""
        return self._strategy

    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Resolve conflicts using the current strategy.

        Args:
            conflicts: List of conflicts to resolve
            console: Optional console for output

        Returns:
            List of resolved conflicts
        """
        return self._strategy.resolve_conflicts(conflicts, console)


# Strategy factory for easy strategy creation
class ConflictResolutionStrategyFactory:
    """Factory for creating conflict resolution strategies."""

    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> ConflictResolutionStrategy:
        """
        Create a conflict resolution strategy by type.

        Args:
            strategy_type: Type of strategy ("interactive", "automatic", "llm")
            **kwargs: Additional arguments for strategy initialization

        Returns:
            ConflictResolutionStrategy instance

        Raises:
            ValueError: If strategy_type is not recognized
        """
        strategy_type = strategy_type.lower()

        if strategy_type in ("interactive", "user", "manual"):
            return UserInteractiveResolutionStrategy()
        elif strategy_type in ("automatic", "auto", "rule-based"):
            return AutomaticResolutionStrategy(**kwargs)
        elif strategy_type in ("llm", "ai", "llm-based"):
            return LLMBasedResolutionStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types."""
        return ["interactive", "automatic", "llm"]
