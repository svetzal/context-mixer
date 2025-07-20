#!/usr/bin/env python3
"""
Example demonstrating how to use the ClusteringBasedResolutionStrategy 
in user-facing tools for intelligent conflict resolution.

This example shows how clustering can be used as a conflict resolution strategy
to automatically resolve conflicts using semantic similarity analysis.
"""

import asyncio
import tempfile
from pathlib import Path
from rich.console import Console

from context_mixer.commands.ingest import IngestCommand
from context_mixer.commands.base import CommandContext
from context_mixer.commands.interactions.conflict_resolution_strategies import (
    ClusteringBasedResolutionStrategy,
    ConflictResolutionContext,
    ConflictResolutionStrategyFactory
)
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory
from context_mixer.domain.clustering import ClusteringConfig
from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from context_mixer.config import Config


def demo_clustering_strategy_creation():
    """Demonstrate different ways to create clustering strategies."""
    console = Console()
    console.print("[bold blue]🧩 Clustering Strategy Creation Demo[/bold blue]\n")
    
    # Method 1: Using the factory
    console.print("[yellow]1. Creating clustering strategy via factory:[/yellow]")
    strategy1 = ConflictResolutionStrategyFactory.create_strategy("clustering")
    console.print(f"   Strategy: {strategy1.get_strategy_name()}")
    
    # Method 2: Direct instantiation with configuration
    console.print("\n[yellow]2. Creating with custom configuration:[/yellow]")
    strategy2 = ClusteringBasedResolutionStrategy()
    console.print(f"   Strategy: {strategy2.get_strategy_name()}")
    console.print(f"   Clustering available: {strategy2._clustering_available}")
    
    # Method 3: With fallback strategy
    console.print("\n[yellow]3. Creating with custom fallback:[/yellow]")
    from context_mixer.commands.interactions.conflict_resolution_strategies import AutomaticResolutionStrategy
    fallback = AutomaticResolutionStrategy(prefer_existing=False)
    strategy3 = ConflictResolutionStrategyFactory.create_strategy(
        "clustering", 
        fallback_strategy=fallback
    )
    console.print(f"   Strategy: {strategy3.get_strategy_name()}")
    console.print(f"   Fallback: {strategy3.fallback_strategy.get_strategy_name()}")


def demo_clustering_conflict_resolution():
    """Demonstrate clustering-based conflict resolution."""
    console = Console()
    console.print("\n[bold blue]🤖 Clustering-Based Conflict Resolution Demo[/bold blue]\n")
    
    # Create sample conflicts with different levels of similarity
    conflicts = [
        Conflict(
            description="Code formatting - similar guidance",
            conflicting_guidance=[
                ConflictingGuidance(
                    content="Use spaces for indentation in Python code", 
                    source="existing"
                ),
                ConflictingGuidance(
                    content="Always use space characters for indenting Python files", 
                    source="new"
                )
            ]
        ),
        Conflict(
            description="Variable naming - different approaches",
            conflicting_guidance=[
                ConflictingGuidance(
                    content="Use camelCase for JavaScript variables", 
                    source="existing"
                ),
                ConflictingGuidance(
                    content="Use snake_case for all variables", 
                    source="new"
                )
            ]
        ),
        Conflict(
            description="Documentation style - comprehensive vs brief",
            conflicting_guidance=[
                ConflictingGuidance(
                    content="Write brief comments", 
                    source="new"
                ),
                ConflictingGuidance(
                    content="Write comprehensive documentation with examples, usage notes, and detailed parameter descriptions", 
                    source="existing"
                )
            ]
        )
    ]
    
    # Create clustering strategy
    strategy = ClusteringBasedResolutionStrategy()
    
    # Resolve conflicts using clustering
    console.print("[yellow]Resolving conflicts with clustering strategy...[/yellow]")
    resolved_conflicts = strategy.resolve_conflicts(conflicts, console)
    
    # Display results
    console.print(f"\n[green]✅ Resolved {len(resolved_conflicts)} conflicts:[/green]")
    for i, conflict in enumerate(resolved_conflicts, 1):
        console.print(f"\n[bold]{i}. {conflict.description}[/bold]")
        console.print(f"   Resolution: {conflict.resolution}")


def demo_strategy_integration_with_context():
    """Demonstrate using clustering strategy with ConflictResolutionContext."""
    console = Console()
    console.print("\n[bold blue]🔄 Strategy Context Integration Demo[/bold blue]\n")
    
    # Create sample conflicts
    conflicts = [
        Conflict(
            description="Testing approach conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Use unit tests for all functions", source="existing"),
                ConflictingGuidance(content="Write unit tests for core functionality", source="new")
            ]
        )
    ]
    
    # Demonstrate strategy switching
    console.print("[yellow]1. Starting with automatic strategy:[/yellow]")
    from context_mixer.commands.interactions.conflict_resolution_strategies import AutomaticResolutionStrategy
    context = ConflictResolutionContext(AutomaticResolutionStrategy())
    result1 = context.resolve_conflicts(conflicts, console)
    console.print(f"   Resolution: {result1[0].resolution}")
    
    console.print("\n[yellow]2. Switching to clustering strategy:[/yellow]")
    context.set_strategy(ClusteringBasedResolutionStrategy())
    result2 = context.resolve_conflicts(conflicts, console)
    console.print(f"   Resolution: {result2[0].resolution}")
    
    console.print(f"\n[green]Current strategy: {context.get_strategy().get_strategy_name()}[/green]")


async def demo_clustering_with_ingest_command():
    """Demonstrate using clustering strategy with IngestCommand."""
    console = Console()
    console.print("\n[bold blue]📥 Clustering Strategy with Ingest Command Demo[/bold blue]\n")
    
    # Create a temporary directory with sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample context files that might conflict
        file1 = temp_path / "context1.md"
        file1.write_text("""
# Coding Standards

## Indentation
Use spaces for indentation in all Python code.

## Variable Naming  
Use descriptive variable names in snake_case.
""")
        
        file2 = temp_path / "context2.md"
        file2.write_text("""
# Development Guidelines

## Code Formatting
Always use space characters for indenting Python files.

## Naming Conventions
Use clear and descriptive variable names.
""")
        
        # Create a temporary knowledge store
        db_path = temp_path / "test_knowledge.db"
        
        # Configure clustering for the knowledge store
        clustering_config = ClusteringConfig(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        
        knowledge_store = KnowledgeStoreFactory.create_vector_store(
            db_path=db_path,
            clustering_config=clustering_config,
            enable_clustering=True
        )
        
        # Create clustering strategy as resolver
        clustering_strategy = ClusteringBasedResolutionStrategy()
        
        # Create a resolver adapter that implements ConflictResolver protocol
        class StrategyResolver:
            def __init__(self, strategy, console):
                self.strategy = strategy
                self.console = console
            
            def resolve_conflicts(self, conflicts):
                return self.strategy.resolve_conflicts(conflicts, self.console)
        
        resolver = StrategyResolver(clustering_strategy, console)
        
        # Create ingest command with clustering-enabled knowledge store
        ingest_command = IngestCommand(knowledge_store)
        
        # Create command context
        config = Config()
        context = CommandContext(
            config=config,
            console=console,
            knowledge_store=knowledge_store,
            parameters={
                'path': temp_path,
                'project_id': "clustering-demo",
                'project_name': "Clustering Strategy Demo",
                'resolver': resolver  # Use clustering strategy for conflict resolution
            }
        )
        
        console.print("[yellow]Ingesting files with clustering strategy enabled...[/yellow]")
        try:
            result = await ingest_command.execute(context)
            console.print(f"[green]✅ Ingest completed: {result.message}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Ingest failed: {e}[/red]")
            # This is expected in demo since we don't have full LLM setup
            console.print("[dim]This is expected in demo environment[/dim]")


def demo_available_strategies():
    """Show all available conflict resolution strategies."""
    console = Console()
    console.print("\n[bold blue]📋 Available Conflict Resolution Strategies[/bold blue]\n")
    
    strategies = ConflictResolutionStrategyFactory.get_available_strategies()
    console.print("[yellow]Available strategies:[/yellow]")
    for strategy in strategies:
        console.print(f"   • {strategy}")
    
    console.print("\n[yellow]Creating instances of each strategy:[/yellow]")
    for strategy_type in strategies:
        try:
            instance = ConflictResolutionStrategyFactory.create_strategy(strategy_type)
            console.print(f"   ✅ {strategy_type}: {instance.get_strategy_name()}")
        except Exception as e:
            console.print(f"   ❌ {strategy_type}: {e}")


async def main():
    """Run all clustering strategy demonstrations."""
    console = Console()
    console.print("[bold green]🎯 Context Mixer - Clustering Strategy Demo[/bold green]")
    console.print("Demonstrating how to use clustering as a conflict resolution strategy\n")
    
    # Run all demos
    demo_clustering_strategy_creation()
    demo_clustering_conflict_resolution()
    demo_strategy_integration_with_context()
    await demo_clustering_with_ingest_command()
    demo_available_strategies()
    
    console.print("\n[bold green]✨ Demo completed![/bold green]")
    console.print("\n[dim]Key takeaways:")
    console.print("• Clustering is now available as a conflict resolution strategy")
    console.print("• Strategies can be easily switched at runtime")
    console.print("• Clustering provides intelligent conflict resolution using semantic similarity")
    console.print("• Full integration with user-facing tools like ingest command")
    console.print("• Graceful fallback when clustering is unavailable")


if __name__ == "__main__":
    asyncio.run(main())