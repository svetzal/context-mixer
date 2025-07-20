"""
Event-Driven Progress Tracking Demonstration

This script demonstrates how the new event-driven progress tracking system works,
showing how progress tracking can be decoupled from direct dependency injection
by using events instead.
"""

from unittest.mock import MagicMock

from rich.console import Console

from context_mixer.domain.events import get_event_bus
from context_mixer.utils.cli_progress import CLIProgressObserver
from context_mixer.utils.event_driven_progress import (
    EventPublishingProgressTracker, EventDrivenProgressTracker, create_cli_progress_tracker
)


def demo_basic_event_driven_progress():
    """Demonstrate basic event-driven progress tracking."""
    console = Console()
    console.print("[bold blue]Event-Driven Progress Tracking Demo[/bold blue]")
    console.print("=" * 50)
    
    # Get the global event bus
    event_bus = get_event_bus()
    event_bus.clear_handlers()  # Clear for clean demo
    
    # Create an event-driven progress tracker that will display progress
    progress_tracker = create_cli_progress_tracker(console, event_bus)
    
    # Create an event-publishing progress tracker that publishes progress events
    # This is what commands would use instead of direct ProgressTracker injection
    publishing_tracker = EventPublishingProgressTracker(
        event_bus=event_bus, 
        project_id="demo-project"
    )
    
    console.print("[cyan]✅ Created event-driven progress system[/cyan]")
    console.print("[dim]• EventDrivenProgressTracker subscribes to progress events[/dim]")
    console.print("[dim]• EventPublishingProgressTracker publishes progress events[/dim]")
    console.print("[dim]• No direct dependency injection required![/dim]")
    console.print()
    
    # Simulate a multi-step operation like file ingestion
    console.print("[bold]Simulating file ingestion process...[/bold]")
    console.print()
    
    # Step 1: File reading
    publishing_tracker.start_operation("file_reading", "Reading files", 5)
    for i in range(1, 6):
        publishing_tracker.update_progress("file_reading", i, f"Read file_{i}.txt")
        # Small delay to show progress
        import time
        time.sleep(0.2)
    publishing_tracker.complete_operation("file_reading")
    
    # Step 2: Chunking
    publishing_tracker.start_operation("chunking", "Creating chunks", 3)
    for i in range(1, 4):
        publishing_tracker.update_progress("chunking", i, f"Created chunk {i}")
        time.sleep(0.3)
    publishing_tracker.complete_operation("chunking")
    
    # Step 3: Validation
    publishing_tracker.start_operation("validation", "Validating chunks", 3)
    for i in range(1, 4):
        publishing_tracker.update_progress("validation", i, f"Validated chunk {i}")
        time.sleep(0.2)
    publishing_tracker.complete_operation("validation")
    
    console.print()
    console.print("[bold green]✅ Demo completed successfully![/bold green]")
    console.print()
    console.print("[bold]Key Benefits Demonstrated:[/bold]")
    console.print("• [green]No direct dependency injection required[/green]")
    console.print("• [green]Loose coupling through events[/green]")
    console.print("• [green]Same UI experience as before[/green]")
    console.print("• [green]Easy to test and extend[/green]")
    
    # Clean up
    progress_tracker.unsubscribe()
    event_bus.clear_handlers()


def demo_multiple_progress_observers():
    """Demonstrate multiple progress observers listening to the same events."""
    console = Console()
    console.print("\n[bold blue]Multiple Progress Observers Demo[/bold blue]")
    console.print("=" * 50)
    
    event_bus = get_event_bus()
    event_bus.clear_handlers()
    
    # Create multiple observers
    cli_observer = CLIProgressObserver(console)
    
    # Mock observer for logging/metrics
    mock_observer = MagicMock()
    
    # Create multiple event-driven trackers
    cli_tracker = EventDrivenProgressTracker(observer=cli_observer, event_bus=event_bus)
    metrics_tracker = EventDrivenProgressTracker(observer=mock_observer, event_bus=event_bus)
    
    # Create publisher
    publisher = EventPublishingProgressTracker(event_bus=event_bus, project_id="multi-demo")
    
    console.print("[cyan]✅ Created multiple progress observers[/cyan]")
    console.print("[dim]• CLI observer shows progress bars[/dim]")
    console.print("[dim]• Mock observer could log metrics[/dim]")
    console.print("[dim]• Both listen to the same events![/dim]")
    console.print()
    
    # Simulate operation
    console.print("[bold]Running operation with multiple observers...[/bold]")
    publisher.start_operation("multi_op", "Multi-observer operation", 3)
    
    import time
    for i in range(1, 4):
        publisher.update_progress("multi_op", i, f"Step {i}")
        time.sleep(0.3)
    
    publisher.complete_operation("multi_op")
    
    console.print()
    console.print("[bold green]✅ Multiple observers demo completed![/bold green]")
    console.print(f"[dim]Mock observer received {mock_observer.on_operation_start.call_count} start calls[/dim]")
    console.print(f"[dim]Mock observer received {mock_observer.on_progress_update.call_count} update calls[/dim]")
    console.print(f"[dim]Mock observer received {mock_observer.on_operation_complete.call_count} complete calls[/dim]")
    
    # Clean up
    cli_tracker.unsubscribe()
    metrics_tracker.unsubscribe()
    event_bus.clear_handlers()


def demo_error_handling():
    """Demonstrate error handling in event-driven progress tracking."""
    console = Console()
    console.print("\n[bold blue]Error Handling Demo[/bold blue]")
    console.print("=" * 50)
    
    event_bus = get_event_bus()
    event_bus.clear_handlers()
    
    # Create progress tracker
    progress_tracker = create_cli_progress_tracker(console, event_bus)
    publisher = EventPublishingProgressTracker(event_bus=event_bus, project_id="error-demo")
    
    console.print("[cyan]✅ Testing error handling[/cyan]")
    console.print()
    
    # Simulate operation that fails
    console.print("[bold]Simulating operation that fails...[/bold]")
    publisher.start_operation("failing_op", "Operation that will fail", 5)
    
    import time
    for i in range(1, 4):
        publisher.update_progress("failing_op", i, f"Processing step {i}")
        time.sleep(0.2)
    
    # Simulate failure
    publisher.fail_operation("failing_op", "Simulated error occurred")
    
    console.print()
    console.print("[bold green]✅ Error handling demo completed![/bold green]")
    console.print("[dim]Progress tracker properly handled the failure[/dim]")
    
    # Clean up
    progress_tracker.unsubscribe()
    event_bus.clear_handlers()


def demo_comparison_with_old_system():
    """Show the difference between old and new systems."""
    console = Console()
    console.print("\n[bold blue]Old vs New System Comparison[/bold blue]")
    console.print("=" * 50)
    
    console.print("[bold]Old System (Direct Dependency Injection):[/bold]")
    console.print("[red]❌ Commands required ProgressTracker parameter[/red]")
    console.print("[red]❌ CLI had to create and pass ProgressTracker[/red]")
    console.print("[red]❌ Tight coupling between components[/red]")
    console.print("[red]❌ Difficult to add multiple progress observers[/red]")
    console.print()
    
    console.print("[bold]New System (Event-Driven):[/bold]")
    console.print("[green]✅ Commands use EventPublishingProgressTracker internally[/green]")
    console.print("[green]✅ CLI creates EventDrivenProgressTracker that subscribes to events[/green]")
    console.print("[green]✅ Loose coupling through event bus[/green]")
    console.print("[green]✅ Easy to add multiple progress observers[/green]")
    console.print("[green]✅ Same user experience, better architecture[/green]")
    console.print()
    
    console.print("[bold]Migration Path:[/bold]")
    console.print("1. [cyan]Commands create EventPublishingProgressTracker internally[/cyan]")
    console.print("2. [cyan]CLI creates EventDrivenProgressTracker instead of passing ProgressTracker[/cyan]")
    console.print("3. [cyan]Remove progress_tracker parameters from command signatures[/cyan]")
    console.print("4. [cyan]All existing functionality preserved![/cyan]")


def run_demo():
    """Run all demonstrations."""
    console = Console()
    console.print("[bold magenta]Context Mixer Event-Driven Progress Demo[/bold magenta]")
    console.print("[dim]Demonstrating how progress tracking can be decoupled from dependency injection[/dim]")
    console.print()
    
    # Run all demos
    demo_basic_event_driven_progress()
    demo_multiple_progress_observers()
    demo_error_handling()
    demo_comparison_with_old_system()
    
    console.print("\n[bold green]🎉 All demos completed successfully![/bold green]")
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print("• Event-driven progress tracking eliminates the need for direct dependency injection")
    console.print("• Commands publish progress events instead of calling progress tracker methods directly")
    console.print("• UI components subscribe to progress events and display progress accordingly")
    console.print("• This provides loose coupling, better testability, and easier extensibility")
    console.print("• The user experience remains exactly the same!")


if __name__ == "__main__":
    run_demo()