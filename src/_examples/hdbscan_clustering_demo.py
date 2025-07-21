"""
HDBSCAN Clustering Performance Demo

This demo showcases the HDBSCAN clustering optimization for conflict detection,
demonstrating the performance benefits and integration capabilities.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Set up logging to reduce noise
logging.basicConfig(level=logging.WARN)

from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, TemporalScope, 
    GranularityLevel, ProvenanceInfo
)
from context_mixer.domain.clustering_integration import (
    ClusterOptimizedConflictDetector, ClusteringConfig, ClusteringStatistics
)
from context_mixer.domain.clustering_service import ClusteringService
from context_mixer.gateways.hdbscan_gateway import create_hdbscan_gateway, MockHDBSCANGateway
from context_mixer.gateways.llm import LLMGateway


@dataclass
class DemoResults:
    """Results from the clustering demo."""
    traditional_time: float
    clustered_time: float
    traditional_conflicts: int
    clustered_conflicts: int
    chunks_processed: int
    clusters_created: int
    optimization_percentage: float
    clustering_stats: ClusteringStatistics


class MockLLMGateway(LLMGateway):
    """Mock LLM gateway for demo purposes."""
    
    def __init__(self, conflict_rate: float = 0.1):
        """Initialize with configurable conflict rate."""
        self.conflict_rate = conflict_rate
        self.call_count = 0
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Mock text generation."""
        self.call_count += 1
        # Simulate some processing time
        await asyncio.sleep(0.01)  
        return "Mock response"
    
    async def detect_conflict(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
        """Mock conflict detection with configurable rate."""
        self.call_count += 1
        # Simulate processing time
        await asyncio.sleep(0.02)
        # Return conflict based on configured rate
        return np.random.random() < self.conflict_rate


class ClusteringDemo:
    """Demonstration of HDBSCAN clustering optimization."""
    
    def __init__(self):
        self.console = Console()
        self.llm_gateway = MockLLMGateway(conflict_rate=0.15)
        
    def create_demo_chunks(self, count: int) -> List[KnowledgeChunk]:
        """Create diverse knowledge chunks for demonstration."""
        chunks = []
        
        # Create chunks with different domains and authority levels
        domains = ["technical", "business", "operational", "security"]
        authorities = [AuthorityLevel.EXPERIMENTAL, AuthorityLevel.FOUNDATIONAL, AuthorityLevel.OFFICIAL]
        
        for i in range(count):
            domain = domains[i % len(domains)]
            authority = authorities[i % len(authorities)]
            
            # Create content that will cluster well
            if domain == "technical":
                content_templates = [
                    "Implement {} pattern in the codebase",
                    "Configure {} for optimal performance", 
                    "Test {} functionality thoroughly",
                    "Document {} implementation details"
                ]
                features = ["authentication", "logging", "caching", "validation", "routing"]
            elif domain == "business":
                content_templates = [
                    "Define {} requirements clearly",
                    "Analyze {} business impact",
                    "Prioritize {} development tasks",
                    "Review {} compliance needs"
                ]
                features = ["user management", "billing", "reporting", "analytics", "workflows"]
            elif domain == "operational":
                content_templates = [
                    "Monitor {} system health",
                    "Deploy {} to production",
                    "Scale {} infrastructure",
                    "Backup {} data regularly"
                ]
                features = ["databases", "services", "networks", "storage", "compute"]
            else:  # security
                content_templates = [
                    "Secure {} against threats",
                    "Audit {} access controls", 
                    "Encrypt {} data transmission",
                    "Validate {} input sanitization"
                ]
                features = ["APIs", "databases", "user sessions", "file uploads", "communications"]
            
            template = content_templates[i % len(content_templates)]
            feature = features[i % len(features)]
            content = template.format(feature)
            
            chunk = KnowledgeChunk(
                id=f"chunk-{i:03d}",
                content=content,
                metadata=ChunkMetadata(
                    domains=[domain],
                    authority=authority,
                    scope=["general"],
                    granularity=GranularityLevel.DETAILED,
                    temporal=TemporalScope.CURRENT,
                    provenance=ProvenanceInfo(
                        source=f"{domain}/{feature}.md",
                        created_at="2024-01-01T00:00:00Z"
                    )
                )
            )
            chunks.append(chunk)
        
        return chunks

    async def run_traditional_conflict_detection(self, target_chunk: KnowledgeChunk, 
                                                existing_chunks: List[KnowledgeChunk]) -> tuple[List[KnowledgeChunk], float]:
        """Run traditional O(n²) conflict detection."""
        start_time = time.time()
        conflicts = []
        
        # Reset LLM call count
        self.llm_gateway.call_count = 0
        
        # Traditional approach: check all pairs
        for chunk in existing_chunks:
            if await self.llm_gateway.detect_conflict(target_chunk, chunk):
                conflicts.append(chunk)
        
        end_time = time.time()
        return conflicts, end_time - start_time

    async def run_clustered_conflict_detection(self, target_chunk: KnowledgeChunk,
                                             existing_chunks: List[KnowledgeChunk]) -> tuple[List[KnowledgeChunk], float, ClusteringStatistics]:
        """Run optimized clustering-based conflict detection."""
        # Create clustering components
        hdbscan_gateway = MockHDBSCANGateway(deterministic=True)
        clustering_service = ClusteringService(
            llm_gateway=self.llm_gateway,
            hdbscan_gateway=hdbscan_gateway
        )
        
        config = ClusteringConfig(
            enabled=True,
            min_cluster_size=3,
            batch_size=5,
            fallback_to_traditional=True
        )
        
        detector = ClusterOptimizedConflictDetector(
            clustering_service,
            self.llm_gateway,
            config
        )
        
        # Reset LLM call count
        self.llm_gateway.call_count = 0
        start_time = time.time()
        
        # Run optimized detection
        conflicts, stats = await detector.detect_conflicts_optimized(
            target_chunk,
            existing_chunks,
            use_cache=False,
            max_candidates=20
        )
        
        end_time = time.time()
        stats.conflict_detection_time = end_time - start_time
        
        return conflicts, end_time - start_time, stats

    def display_results(self, results: DemoResults):
        """Display demo results in a nice format."""
        
        # Performance Summary Panel
        performance_table = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
        performance_table.add_column("Metric", style="cyan", width=25)
        performance_table.add_column("Traditional O(n²)", style="red", width=20)
        performance_table.add_column("Clustered O(k*log(k))", style="green", width=20)
        performance_table.add_column("Improvement", style="bold yellow", width=15)
        
        time_improvement = ((results.traditional_time - results.clustered_time) / results.traditional_time) * 100
        performance_table.add_row(
            "Processing Time",
            f"{results.traditional_time:.3f}s",
            f"{results.clustered_time:.3f}s", 
            f"{time_improvement:.1f}% faster"
        )
        
        performance_table.add_row(
            "LLM API Calls",
            f"{results.chunks_processed}",
            f"{results.chunks_processed - int(results.chunks_processed * results.optimization_percentage / 100)}",
            f"{results.optimization_percentage:.1f}% reduction"
        )
        
        performance_table.add_row(
            "Conflicts Found",
            f"{results.traditional_conflicts}",
            f"{results.clustered_conflicts}",
            "Same accuracy" if results.traditional_conflicts == results.clustered_conflicts else "Different"
        )
        
        self.console.print()
        self.console.print(Panel(performance_table, title="🚀 HDBSCAN Clustering Optimization Results", border_style="blue"))
        
        # Clustering Details Panel
        clustering_table = Table(title="Clustering Analysis", show_header=True, header_style="bold blue")
        clustering_table.add_column("Metric", style="cyan", width=30)
        clustering_table.add_column("Value", style="white", width=20)
        clustering_table.add_column("Description", style="dim", width=40)
        
        clustering_table.add_row(
            "Total Chunks Clustered",
            str(results.clustering_stats.total_chunks_clustered),
            "Number of chunks processed for clustering"
        )
        
        clustering_table.add_row(
            "Clusters Created", 
            str(results.clusters_created),
            "Semantic groups discovered by HDBSCAN"
        )
        
        clustering_table.add_row(
            "Comparisons Avoided",
            str(results.clustering_stats.traditional_comparisons_avoided),
            "LLM calls eliminated through clustering"
        )
        
        clustering_table.add_row(
            "Clustering Time",
            f"{results.clustering_stats.clustering_time:.3f}s",
            "Time spent on HDBSCAN clustering"
        )
        
        clustering_table.add_row(
            "Conflict Detection Time",
            f"{results.clustering_stats.conflict_detection_time:.3f}s",
            "Time spent on actual conflict detection"
        )
        
        self.console.print()
        self.console.print(Panel(clustering_table, title="📊 Clustering Analysis Details", border_style="green"))

    async def run_demo(self, chunk_count: int = 50):
        """Run the complete clustering demonstration."""
        
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold blue]HDBSCAN Clustering Performance Demo[/bold blue]\n"
            f"Demonstrating conflict detection optimization with {chunk_count} knowledge chunks",
            border_style="blue"
        ))
        
        # Create demo data
        self.console.print(f"\n[cyan]Creating {chunk_count} diverse knowledge chunks...[/cyan]")
        chunks = self.create_demo_chunks(chunk_count)
        
        # Separate target chunk from existing chunks
        target_chunk = chunks[0]
        existing_chunks = chunks[1:]
        
        self.console.print(f"[green]✓[/green] Created chunks across domains: technical, business, operational, security")
        self.console.print(f"[green]✓[/green] Target chunk: {target_chunk.id} - '{target_chunk.content[:50]}...'")
        
        # Run traditional detection
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            
            task1 = progress.add_task("[red]Running traditional O(n²) conflict detection...", total=100)
            traditional_conflicts, traditional_time = await self.run_traditional_conflict_detection(
                target_chunk, existing_chunks
            )
            traditional_calls = self.llm_gateway.call_count
            progress.update(task1, completed=100)
            
            task2 = progress.add_task("[green]Running clustered O(k*log(k)) detection...", total=100)
            clustered_conflicts, clustered_time, clustering_stats = await self.run_clustered_conflict_detection(
                target_chunk, existing_chunks
            )
            clustered_calls = self.llm_gateway.call_count
            progress.update(task2, completed=100)
        
        # Calculate optimization metrics
        optimization_percentage = ((traditional_calls - clustered_calls) / traditional_calls) * 100 if traditional_calls > 0 else 0
        
        results = DemoResults(
            traditional_time=traditional_time,
            clustered_time=clustered_time,
            traditional_conflicts=len(traditional_conflicts),
            clustered_conflicts=len(clustered_conflicts),
            chunks_processed=len(existing_chunks),
            clusters_created=clustering_stats.clusters_created,
            optimization_percentage=optimization_percentage,
            clustering_stats=clustering_stats
        )
        
        # Display results
        self.display_results(results)
        
        # Show scaling potential
        self.console.print()
        scaling_panel = Panel(
            f"[bold green]Scaling Benefits[/bold green]\n\n"
            f"• With {chunk_count} chunks: {optimization_percentage:.1f}% reduction in LLM calls\n"
            f"• With 1,000 chunks: ~70-80% reduction expected\n" 
            f"• With 10,000 chunks: ~85-95% reduction expected\n\n"
            f"[dim]The optimization becomes more significant with larger knowledge bases,\n"
            f"making large-scale ingestion practical and cost-effective.[/dim]",
            title="📈 Performance Scaling",
            border_style="yellow"
        )
        self.console.print(scaling_panel)
        
        return results


async def main():
    """Run the clustering demo."""
    demo = ClusteringDemo()
    
    # Run demo with different sizes to show scaling
    for size in [20, 50, 100]:
        await demo.run_demo(size)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())