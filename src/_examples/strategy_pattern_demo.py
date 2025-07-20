"""
Strategy Pattern Demonstration

This script demonstrates the new Strategy pattern implementation for conflict resolution,
including auto-selection of strategies based on conflict characteristics.
"""

from rich.console import Console

from context_mixer.commands.interactions.conflict_resolution_strategies import (
    ConflictResolutionStrategyFactory,
    ConflictResolutionContext,
    AutomaticResolutionStrategy,
    LLMBasedResolutionStrategy
)
from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from workbench.automated_resolver import AutomatedConflictResolver


def demo_basic_strategies():
    """Demonstrate basic strategy usage."""
    console = Console()
    console.print("\n[bold blue]🎯 Basic Strategy Pattern Demo[/bold blue]")
    
    # Create sample conflicts
    conflicts = [
        Conflict(
            description="Indentation style conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Use tabs for indentation", source="existing"),
                ConflictingGuidance(content="Use 4 spaces for indentation", source="new")
            ]
        ),
        Conflict(
            description="Variable naming convention",
            conflicting_guidance=[
                ConflictingGuidance(content="Use camelCase for variables", source="style_guide"),
                ConflictingGuidance(content="Use snake_case for variables", source="existing")
            ]
        )
    ]
    
    # Demonstrate different strategies
    strategies = [
        ("Automatic", AutomaticResolutionStrategy()),
        ("LLM-Based", LLMBasedResolutionStrategy()),
    ]
    
    for strategy_name, strategy in strategies:
        console.print(f"\n[yellow]--- Using {strategy_name} Strategy ---[/yellow]")
        context = ConflictResolutionContext(strategy)
        resolved = context.resolve_conflicts(conflicts, console)
        
        for i, conflict in enumerate(resolved):
            console.print(f"Conflict {i+1} resolved: [green]{conflict.resolution}[/green]")


def demo_strategy_factory():
    """Demonstrate strategy factory usage."""
    console = Console()
    console.print("\n[bold blue]🏭 Strategy Factory Demo[/bold blue]")
    
    conflict = Conflict(
        description="Code formatting conflict",
        conflicting_guidance=[
            ConflictingGuidance(content="Format with black", source="existing"),
            ConflictingGuidance(content="Format with autopep8", source="new")
        ]
    )
    
    # Test different factory methods
    strategy_types = ["automatic", "llm", "interactive"]
    
    for strategy_type in strategy_types:
        console.print(f"\n[yellow]Creating {strategy_type} strategy...[/yellow]")
        try:
            strategy = ConflictResolutionStrategyFactory.create_strategy(strategy_type)
            console.print(f"✓ Created: {strategy.get_strategy_name()}")
            
            if strategy_type != "interactive":  # Skip interactive for demo
                context = ConflictResolutionContext(strategy)
                resolved = context.resolve_conflicts([conflict], console)
                console.print(f"Resolution: [green]{resolved[0].resolution}[/green]")
        except Exception as e:
            console.print(f"✗ Error: {e}")


def demo_auto_selection():
    """Demonstrate intelligent auto-selection of strategies."""
    console = Console()
    console.print("\n[bold blue]🧠 Auto-Selection Demo[/bold blue]")
    
    # Create conflicts of different types to trigger different strategies
    conflicts = [
        # Simple conflict - should use automatic strategy
        Conflict(
            description="Simple existing vs new conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Original implementation", source="existing"),
                ConflictingGuidance(content="New implementation", source="new")
            ]
        ),
        
        # Style conflict - should use automatic strategy with style preferences
        Conflict(
            description="Indentation style preference",
            conflicting_guidance=[
                ConflictingGuidance(content="Use tabs", source="existing"),
                ConflictingGuidance(content="Use spaces", source="style_guide")
            ]
        ),
        
        # Complex conflict - should attempt LLM strategy
        Conflict(
            description="Complex architectural decision involving multiple design patterns and performance considerations that requires careful analysis",
            conflicting_guidance=[
                ConflictingGuidance(content="Use singleton pattern for database connection", source="existing"),
                ConflictingGuidance(content="Use dependency injection for database connection", source="new"),
                ConflictingGuidance(content="Use factory pattern for database connection", source="alternative")
            ]
        ),
        
        # Default conflict
        Conflict(
            description="General conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Option A", source="source1"),
                ConflictingGuidance(content="Option B", source="source2")
            ]
        )
    ]
    
    # Use the enhanced AutomatedConflictResolver
    resolver = AutomatedConflictResolver(console=console, default_strategy="automatic")
    
    console.print(f"\n[cyan]Processing {len(conflicts)} conflicts with auto-selection...[/cyan]")
    resolved_conflicts = resolver.resolve_conflicts(conflicts)
    
    console.print(f"\n[green]✓ Successfully resolved {len(resolved_conflicts)} conflicts[/green]")


def demo_runtime_strategy_switching():
    """Demonstrate runtime strategy switching."""
    console = Console()
    console.print("\n[bold blue]🔄 Runtime Strategy Switching Demo[/bold blue]")
    
    conflict = Conflict(
        description="Runtime switching test",
        conflicting_guidance=[
            ConflictingGuidance(content="Original approach", source="existing"),
            ConflictingGuidance(content="New approach", source="new")
        ]
    )
    
    # Start with automatic strategy
    context = ConflictResolutionContext(AutomaticResolutionStrategy())
    console.print(f"Initial strategy: {context.get_strategy().get_strategy_name()}")
    
    result1 = context.resolve_conflicts([conflict], console)
    console.print(f"Result 1: [green]{result1[0].resolution}[/green]")
    
    # Switch to LLM strategy
    context.set_strategy(LLMBasedResolutionStrategy())
    console.print(f"Switched to: {context.get_strategy().get_strategy_name()}")
    
    result2 = context.resolve_conflicts([conflict], console)
    console.print(f"Result 2: [green]{result2[0].resolution}[/green]")


def demo_workbench_integration():
    """Demonstrate integration with workbench scenarios."""
    console = Console()
    console.print("\n[bold blue]🔬 Workbench Integration Demo[/bold blue]")
    
    # Create a resolver with different default strategies
    strategies_to_test = ["automatic", "llm"]
    
    conflicts = [
        Conflict(
            description="Workbench test conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Existing workbench behavior", source="existing"),
                ConflictingGuidance(content="New workbench behavior", source="new")
            ]
        )
    ]
    
    for strategy_name in strategies_to_test:
        console.print(f"\n[yellow]--- Testing with {strategy_name} default strategy ---[/yellow]")
        resolver = AutomatedConflictResolver(console=console, default_strategy=strategy_name)
        
        resolved = resolver.resolve_conflicts(conflicts)
        console.print(f"Resolved with: [green]{resolved[0].resolution}[/green]")
        
        # Demonstrate manual strategy override
        console.print("Manually setting to automatic strategy...")
        resolver.set_strategy_by_name("automatic", prefer_existing=False)
        
        resolved2 = resolver.resolve_conflicts(conflicts)
        console.print(f"After override: [green]{resolved2[0].resolution}[/green]")


def main():
    """Run all demonstrations."""
    console = Console()
    console.print("[bold green]🚀 Strategy Pattern for Conflict Resolution - Demo[/bold green]")
    console.print("This demo shows the new Strategy pattern implementation with auto-selection capabilities.")
    
    try:
        demo_basic_strategies()
        demo_strategy_factory()
        demo_auto_selection()
        demo_runtime_strategy_switching()
        demo_workbench_integration()
        
        console.print("\n[bold green]✅ All demonstrations completed successfully![/bold green]")
        console.print("\n[bold cyan]Key Features Demonstrated:[/bold cyan]")
        console.print("• Multiple resolution strategies (Automatic, LLM-based, Interactive)")
        console.print("• Strategy factory for easy creation")
        console.print("• Intelligent auto-selection based on conflict characteristics")
        console.print("• Runtime strategy switching")
        console.print("• Workbench integration with configurable defaults")
        console.print("• Backward compatibility with existing code")
        
    except Exception as e:
        console.print(f"[red]Demo failed with error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()