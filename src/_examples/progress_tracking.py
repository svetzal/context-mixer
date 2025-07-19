#!/usr/bin/env python3
"""
Test script to verify progress tracking functionality.
"""
import asyncio
import tempfile
from pathlib import Path
from rich.console import Console

from context_mixer.utils.progress import ProgressTracker
from context_mixer.utils.cli_progress import CLIProgressObserver


async def show_progress_tracking():
    """Test the progress tracking system."""
    console = Console()
    
    # Create progress tracking system
    cli_progress_observer = CLIProgressObserver(console)
    progress_tracker = ProgressTracker(cli_progress_observer)
    
    console.print("[bold blue]Testing Progress Tracking System[/bold blue]")
    
    # Test 1: Simple operation
    console.print("\n[yellow]Test 1: Simple file processing simulation[/yellow]")
    progress_tracker.start_operation("test_files", "Processing files", 5)
    
    for i in range(5):
        await asyncio.sleep(0.5)  # Simulate work
        progress_tracker.update_progress("test_files", i + 1, f"Processed file {i + 1}")
    
    progress_tracker.complete_operation("test_files")
    
    # Test 2: Multiple concurrent operations
    console.print("\n[yellow]Test 2: Multiple concurrent operations[/yellow]")
    
    async def simulate_operation(op_id: str, op_name: str, total: int):
        progress_tracker.start_operation(op_id, op_name, total)
        for i in range(total):
            await asyncio.sleep(0.3)
            progress_tracker.update_progress(op_id, i + 1, f"Step {i + 1}")
        progress_tracker.complete_operation(op_id)
    
    # Run multiple operations concurrently
    await asyncio.gather(
        simulate_operation("chunking", "Creating chunks", 3),
        simulate_operation("validation", "Validating chunks", 4),
        simulate_operation("conflicts", "Checking conflicts", 2)
    )
    
    console.print("\n[green]✅ Progress tracking test completed successfully![/green]")


if __name__ == "__main__":
    asyncio.run(show_progress_tracking())