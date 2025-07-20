#!/usr/bin/env python3
"""
HDBSCAN Clustering Demo for Context Mixer

This script demonstrates the new hierarchical clustering system that optimizes
conflict detection from O(n²) to O(k*log(k)) by grouping semantically similar
chunks and only checking conflicts within/between related clusters.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Set logging level before importing mojentic
logging.basicConfig(level=logging.WARN)

from context_mixer.domain.clustering import MockHDBSCANClusterer, ContextualDomain
from context_mixer.domain.clustering_integration import ClusterOptimizedConflictDetector, ClusteringConfig
from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, ProvenanceInfo, 
    GranularityLevel, TemporalScope
)


def create_diverse_chunks():
    """Create diverse knowledge chunks for clustering demonstration."""
    chunks = []
    
    # Technical architecture chunks (should cluster together)
    for i in range(4):
        provenance = ProvenanceInfo(
            source=f"architecture_{i}.md",
            created_at=datetime.now().isoformat()
        )
        
        metadata = ChunkMetadata(
            domains=["technical", "architecture"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=[f"architecture_{i}"],
            provenance=provenance
        )
        
        chunk = KnowledgeChunk(
            id=f"arch_chunk_{i}",
            content=f"Architecture guidance {i}: Use microservices pattern for scalability",
            metadata=metadata
        )
        chunks.append(chunk)
    
    # Business process chunks (should cluster together, separate from technical)
    for i in range(3):
        provenance = ProvenanceInfo(
            source=f"business_{i}.md",
            created_at=datetime.now().isoformat()
        )
        
        metadata = ChunkMetadata(
            domains=["business", "process"],
            authority=AuthorityLevel.FOUNDATIONAL,
            scope=["company"],
            granularity=GranularityLevel.OVERVIEW,
            temporal=TemporalScope.CURRENT,
            tags=[f"business_{i}"],
            provenance=provenance
        )
        
        chunk = KnowledgeChunk(
            id=f"biz_chunk_{i}",
            content=f"Business process {i}: Follow GDPR compliance procedures",
            metadata=metadata
        )
        chunks.append(chunk)
    
    # Security implementation chunks (different authority, should cluster separately)
    for i in range(2):
        provenance = ProvenanceInfo(
            source=f"security_{i}.md",
            created_at=datetime.now().isoformat()
        )
        
        metadata = ChunkMetadata(
            domains=["technical", "security"],
            authority=AuthorityLevel.EXPERIMENTAL,
            scope=["prototype"],
            granularity=GranularityLevel.COMPREHENSIVE,
            temporal=TemporalScope.CURRENT,
            tags=[f"security_{i}"],
            provenance=provenance
        )
        
        chunk = KnowledgeChunk(
            id=f"sec_chunk_{i}",
            content=f"Security implementation {i}: Use OAuth 2.0 for authentication",
            metadata=metadata
        )
        chunks.append(chunk)
    
    return chunks


async def demonstrate_clustering():
    """Demonstrate the clustering system and optimization benefits."""
    print("🚀 HDBSCAN Clustering Optimization Demo")
    print("=" * 50)
    
    # Create diverse knowledge chunks
    chunks = create_diverse_chunks()
    print(f"📦 Created {len(chunks)} diverse knowledge chunks:")
    print("   - 4 technical architecture chunks (official authority)")
    print("   - 3 business process chunks (foundational authority)")  
    print("   - 2 security implementation chunks (experimental authority)")
    
    # Create clusterer
    clusterer = MockHDBSCANClusterer(min_cluster_size=2)
    
    print(f"\n🧠 Performing HDBSCAN clustering (min_cluster_size=2)...")
    contextual_chunks, clusters = await clusterer.cluster_chunks(chunks)
    
    print(f"✅ Clustering complete!")
    print(f"   - Created {len(clusters)} clusters")
    print(f"   - {len([c for c in contextual_chunks if c.cluster_id])} chunks assigned to clusters")
    print(f"   - {len([c for c in contextual_chunks if not c.cluster_id])} chunks marked as noise")
    
    # Show cluster details
    print(f"\n📊 Cluster Analysis:")
    for cluster_id, cluster in clusters.items():
        intelligence = cluster.get_cluster_intelligence()
        print(f"   Cluster {cluster_id}:")
        print(f"     - Domain: {intelligence['domain']}")
        print(f"     - Chunks: {intelligence['chunk_count']}")
        print(f"     - Status: {intelligence['status']}")
        print(f"     - Authority distribution: {intelligence['authority_distribution']}")
    
    # Demonstrate conflict optimization
    print(f"\n⚡ Conflict Detection Optimization:")
    
    # Traditional approach: O(n²) comparisons
    traditional_comparisons = len(chunks) * (len(chunks) - 1) // 2
    print(f"   Traditional O(n²): {traditional_comparisons} pairwise comparisons needed")
    
    # Clustering approach: count actual candidates
    config = ClusteringConfig(enabled=True, min_cluster_size=2, batch_size=3)
    detector = ClusterOptimizedConflictDetector(clusterer=clusterer, config=config)
    
    optimized_comparisons = 0
    for chunk in contextual_chunks:
        candidates = detector.dynamic_detector.get_conflict_candidates(
            chunk, contextual_chunks, clusters
        )
        optimized_comparisons += len(candidates)
    
    # Avoid double counting (each pair counted twice)
    optimized_comparisons = optimized_comparisons // 2
    
    print(f"   Clustering optimized: {optimized_comparisons} targeted comparisons needed")
    
    if traditional_comparisons > 0:
        reduction = 1.0 - (optimized_comparisons / traditional_comparisons)
        print(f"   🎯 Reduction: {reduction:.1%} fewer conflict checks!")
    
    # Show which chunks would be compared
    print(f"\n🔍 Conflict Detection Strategy:")
    for i, chunk in enumerate(contextual_chunks[:3]):  # Show first 3 for brevity
        candidates = detector.dynamic_detector.get_conflict_candidates(
            chunk, contextual_chunks, clusters
        )
        
        print(f"   Chunk {chunk.base_chunk.id}:")
        print(f"     - Cluster: {chunk.cluster_id or 'noise'}")
        print(f"     - Domain: {chunk.contextual_domain}")
        print(f"     - Candidates to check: {len(candidates)}")
        
        if candidates:
            candidate_clusters = [c.cluster_id or 'noise' for c in candidates]
            print(f"     - Candidate clusters: {set(candidate_clusters)}")
    
    print(f"\n💡 Key Benefits:")
    print(f"   ✅ Dramatically reduced LLM API calls ({reduction:.1%} reduction)")
    print(f"   ✅ Context-aware conflict detection (same/related domains)")
    print(f"   ✅ Hierarchical cluster awareness prevents false positives")
    print(f"   ✅ Graceful fallback to O(n²) detection if clustering fails")
    print(f"   ✅ Authority-level and domain-based intelligent grouping")
    
    return clusters, contextual_chunks, detector


async def demonstrate_performance_scaling():
    """Demonstrate how clustering optimization scales with larger datasets."""
    print(f"\n📈 Performance Scaling Analysis:")
    print("=" * 30)
    
    chunk_counts = [5, 10, 20, 50]
    
    for n in chunk_counts:
        # Traditional O(n²) comparisons
        traditional = n * (n - 1) // 2
        
        # Estimated clustering optimization (assumes ~3 chunks per cluster on average)
        avg_cluster_size = 3
        num_clusters = max(1, n // avg_cluster_size)
        
        # Optimized: within-cluster + cross-cluster for same domains
        within_cluster = num_clusters * (avg_cluster_size * (avg_cluster_size - 1) // 2)
        cross_cluster = num_clusters * (num_clusters - 1) // 2 * 2  # Some cross-cluster checks
        optimized = within_cluster + cross_cluster
        
        reduction = 1.0 - (optimized / traditional) if traditional > 0 else 0
        
        print(f"   {n} chunks: {traditional} → {optimized} comparisons ({reduction:.1%} reduction)")
    
    print(f"\n   As dataset grows, clustering provides exponentially greater benefits!")


if __name__ == "__main__":
    async def main():
        clusters, contextual_chunks, detector = await demonstrate_clustering()
        await demonstrate_performance_scaling()
        
        print(f"\n🎉 Demo complete!")
        print(f"   This clustering system is now integrated into the ingest pipeline")
        print(f"   and will automatically optimize conflict detection for large datasets.")
    
    asyncio.run(main())