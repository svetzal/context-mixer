from typing import List

from context_mixer.domain.conflict import Conflict


def resolve_conflicts(conflicts: List[Conflict], console) -> List[Conflict]:
    """
    Resolve conflicts by consulting the user.

    This function presents each conflict to the user and asks them to choose
    which guidance is correct or enter their own resolution.

    Args:
        conflicts: A list of Conflict objects to resolve
        console: Rich console for output

    Returns:
        The list of conflicts with resolutions set
    """
    # If conflicts is empty, return it as is
    if not conflicts:
        return []

    # Iterate through each conflict and resolve it
    for conflict in conflicts:
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

        # Ask the user to choose an option
        while True:
            choice = console.input("\n[bold]Select an option (Enter the number):[/bold] ")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(conflict.conflicting_guidance):
                    # Set the resolution to the chosen guidance
                    conflict.resolution = conflict.conflicting_guidance[choice_idx].content
                    break
                elif choice_idx == len(conflict.conflicting_guidance):
                    # User wants to enter their own resolution
                    custom_resolution = console.input("\n[bold]Enter your custom resolution:[/bold] ")
                    conflict.resolution = custom_resolution
                    break
                else:
                    console.print("[red]Invalid choice. Please enter a valid number.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")

    return conflicts
