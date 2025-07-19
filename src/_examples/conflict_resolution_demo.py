"""
Demonstration of enhanced conflict resolution features.

This script shows the enhanced UserInteractiveResolutionStrategy which now includes:
1. Multi-line input for custom resolutions (option #3)
2. A 4th option that allows users to indicate that something is not actually a conflict
"""

from rich.console import Console
from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from context_mixer.commands.interactions.conflict_resolution_strategies import UserInteractiveResolutionStrategy


def demo_conflict_resolution_options():
    """Demonstrate the conflict resolution options including the new 'This is not a conflict' option."""
    console = Console()

    # Create a sample conflict
    conflict = Conflict(
        description="Indentation style preference",
        conflicting_guidance=[
            ConflictingGuidance(
                content="Use tabs for indentation",
                source="existing"
            ),
            ConflictingGuidance(
                content="Use spaces for indentation", 
                source="new"
            )
        ]
    )

    console.print("[bold green]🚀 Enhanced Conflict Resolution Demo[/bold green]\n")

    console.print("This demo shows the enhanced conflict resolution features:")
    console.print("[bold cyan]1. Multi-line input for custom resolutions (option #3)[/bold cyan]")
    console.print("   - You can now enter multiple lines of text for your custom resolution")
    console.print("   - End your input with a single '.' on a line by itself")
    console.print("[bold cyan]2. 'This is not a conflict' option (option #4)[/bold cyan]")
    console.print("   - Allows you to indicate that something is not actually a conflict")
    console.print()

    console.print("When presented with a conflict, you now have these options:")
    console.print("1. Choose the guidance from existing")
    console.print("2. Choose the guidance from new") 
    console.print("3. Enter your own resolution [bold](now supports multi-line input!)[/bold]")
    console.print("4. This is not a conflict")
    console.print()

    console.print("The conflict will be presented below.")
    console.print("[bold yellow]Try option 3 to test multi-line input, or option 4 to test the 'not a conflict' feature![/bold yellow]")
    console.print()

    # Create the strategy and resolve the conflict
    strategy = UserInteractiveResolutionStrategy()

    try:
        resolved_conflicts = strategy.resolve_conflicts([conflict], console)

        console.print("\n[bold]Resolution Result:[/bold]")
        resolved_conflict = resolved_conflicts[0]

        if resolved_conflict.resolution is None:
            console.print("[green]✅ User indicated this is not a conflict - resolution set to None[/green]")
        else:
            console.print(f"[blue]Resolution: {resolved_conflict.resolution}[/blue]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    demo_conflict_resolution_options()
