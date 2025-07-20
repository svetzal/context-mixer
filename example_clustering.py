#!/usr/bin/env python3
"""
Example demonstrating HDBSCAN clustering optimization in Context Mixer.

This example shows how to use the new clustering features to optimize
conflict detection for large knowledge bases.
"""

import asyncio
from pathlib import Path
import tempfile
import logging

# Set up logging to see clustering information
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_example_knowledge_store():
    """
    Create a VectorKnowledgeStore with clustering enabled.
    
    This example shows how to configure clustering parameters
    for optimal performance based on your knowledge base size.
    """
    from context_mixer.domain.knowledge_store import KnowledgeStoreFactory
    from context_mixer.domain.clustering import ClusteringConfig
    
    # Configure clustering for your knowledge base
    clustering_config = ClusteringConfig(
        min_cluster_size=5,      # Minimum 5 chunks per cluster
        min_samples=3,           # Minimum 3 samples for core points
        cluster_selection_epsilon=0.0,  # No cluster merging threshold
        metric='euclidean',      # Distance metric
        alpha=1.0,              # Distance scaling parameter
        prediction_data=True     # Enable prediction for new chunks
    )
    
    # Create temporary directory for this example
    db_path = Path(tempfile.mkdtemp()) / "example_knowledge_db"
    
    # Create knowledge store with clustering enabled
    store = KnowledgeStoreFactory.create_vector_store(
        db_path=db_path,
        clustering_config=clustering_config,
        enable_clustering=True  # Enable clustering optimization
    )
    
    print(f"✅ Created VectorKnowledgeStore with clustering at {db_path}")
    print(f"📊 Clustering config: {clustering_config.model_dump()}")
    
    return store

def create_sample_chunks():
    """Create sample knowledge chunks for demonstration."""
    from context_mixer.domain.knowledge import (
        KnowledgeChunk, ChunkMetadata, AuthorityLevel, 
        GranularityLevel, TemporalScope, ProvenanceInfo
    )
    from datetime import datetime
    
    chunks = []
    
    # Create chunks with different domains and content
    chunk_data = [
        ("chunk_1", "Python best practices for web development", ["technical", "python", "web"]),
        ("chunk_2", "React component architecture patterns", ["technical", "frontend", "react"]),
        ("chunk_3", "Database indexing strategies", ["technical", "database", "performance"]),
        ("chunk_4", "Security considerations for API design", ["technical", "security", "api"]),
        ("chunk_5", "DevOps deployment pipelines", ["technical", "devops", "deployment"]),
        ("chunk_6", "Project management methodologies", ["business", "management", "process"]),
        ("chunk_7", "Code review best practices", ["technical", "process", "quality"]),
        ("chunk_8", "Testing strategies for microservices", ["technical", "testing", "architecture"]),
        ("chunk_9", "Performance optimization techniques", ["technical", "performance", "optimization"]),
        ("chunk_10", "Documentation standards", ["process", "documentation", "standards"]),
    ]
    
    current_time = datetime.now().isoformat()
    
    for chunk_id, content, domains in chunk_data:
        # Create complete provenance information
        provenance = ProvenanceInfo(
            source=f"/example/source/{chunk_id}.md",
            project_id="example_project",
            project_name="Example Project",
            project_path="/example/project",
            created_at=current_time,
            author="example_user"
        )
        
        # Create complete metadata
        metadata = ChunkMetadata(
            domains=domains,
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise", "development"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=provenance,
            tags=domains  # Use domains as tags for simplicity
        )
        
        chunk = KnowledgeChunk(
            id=chunk_id,
            content=content,
            metadata=metadata
        )
        chunks.append(chunk)
    
    print(f"📝 Created {len(chunks)} sample knowledge chunks")
    return chunks

async def demonstrate_clustering_workflow():
    """Demonstrate the complete clustering workflow."""
    print("\n🚀 HDBSCAN Clustering Example")
    print("=" * 50)
    
    try:
        # Create knowledge store with clustering
        store = create_example_knowledge_store()
        
        # Create and store sample chunks
        chunks = create_sample_chunks()
        
        print(f"\n📥 Storing {len(chunks)} chunks...")
        await store.store_chunks(chunks)
        
        # Get initial stats
        print(f"\n📊 Initial knowledge store stats:")
        stats = await store.get_stats()
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Clustering enabled: {stats['clustering_enabled']}")
        
        if stats.get('clustering'):
            clustering_stats = stats['clustering']
            print(f"   Clusters formed: {clustering_stats.get('total_clusters', 'Not available')}")
            print(f"   Noise chunks: {clustering_stats.get('noise_chunks', 'Not available')}")
        
        # Demonstrate conflict detection with clustering
        print(f"\n🔍 Testing conflict detection with clustering optimization...")
        
        # Create a new chunk that might conflict
        from context_mixer.domain.knowledge import (
            KnowledgeChunk, ChunkMetadata, AuthorityLevel,
            GranularityLevel, TemporalScope, ProvenanceInfo
        )
        from datetime import datetime
        
        current_time = datetime.now().isoformat()
        
        new_provenance = ProvenanceInfo(
            source="/example/source/new_chunk.md",
            project_id="example_project",
            project_name="Example Project", 
            project_path="/example/project",
            created_at=current_time,
            author="example_user"
        )
        
        new_chunk_metadata = ChunkMetadata(
            domains=["technical", "python"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise", "development"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=new_provenance,
            tags=["technical", "python", "alternative"]
        )
        
        new_chunk = KnowledgeChunk(
            id="new_chunk",
            content="Alternative Python web development best practices",
            metadata=new_chunk_metadata
        )
        
        # Detect conflicts using clustering optimization
        conflicts = await store.detect_conflicts(new_chunk)
        print(f"   Found {len(conflicts)} potential conflicts")
        
        for conflict in conflicts:
            print(f"   ⚠️  Conflict with: {conflict.id}")
        
        # Get cluster information for a specific chunk
        print(f"\n🎯 Cluster information for 'chunk_1':")
        cluster_info = await store.get_cluster_info("chunk_1")
        if cluster_info:
            print(f"   Cluster ID: {cluster_info['cluster_id']}")
            print(f"   Cluster size: {cluster_info['cluster_size']}")
            print(f"   Is noise: {cluster_info['is_noise']}")
            print(f"   Related chunks: {len(cluster_info['cluster_chunks'])}")
        else:
            print("   Clustering information not available")
        
        # Manually rebuild clusters
        print(f"\n🔄 Manually rebuilding clusters...")
        rebuild_stats = await store.rebuild_clusters()
        if 'error' not in rebuild_stats:
            print(f"   Rebuild successful!")
            print(f"   Total clusters: {rebuild_stats.get('total_clusters', 'N/A')}")
            print(f"   Total chunks: {rebuild_stats.get('total_chunks', 'N/A')}")
        else:
            print(f"   Rebuild failed: {rebuild_stats['error']}")
        
        # Final stats
        print(f"\n📈 Final knowledge store stats:")
        final_stats = await store.get_stats()
        if final_stats.get('clustering'):
            clustering_stats = final_stats['clustering']
            print(f"   Clusters: {clustering_stats.get('total_clusters', 0)}")
            print(f"   Average cluster size: {clustering_stats.get('avg_cluster_size', 0):.1f}")
            print(f"   Clustering quality: {clustering_stats.get('relative_validity', 'N/A')}")
        
        print(f"\n✅ Clustering optimization demonstration complete!")
        print(f"   The system is now ready to handle large-scale conflict detection")
        print(f"   with significant performance improvements.")
        
    except ImportError as e:
        print(f"\n❌ Clustering not available: {e}")
        print(f"   Install HDBSCAN to enable clustering: pip install hdbscan")
        print(f"   The system will fall back to domain-based conflict detection.")
    
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise
    
    finally:
        # Clean up
        if 'store' in locals():
            await store.close()

def show_configuration_examples():
    """Show different clustering configuration examples."""
    from context_mixer.domain.clustering import ClusteringConfig
    
    print(f"\n⚙️  Clustering Configuration Examples")
    print("=" * 40)
    
    configs = [
        ("Small knowledge base (< 100 chunks)", ClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean'
        )),
        ("Medium knowledge base (100-1000 chunks)", ClusteringConfig(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean'
        )),
        ("Large knowledge base (> 1000 chunks)", ClusteringConfig(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_epsilon=0.1,
            metric='euclidean'
        )),
        ("High-precision clustering", ClusteringConfig(
            min_cluster_size=8,
            min_samples=6,
            cluster_selection_epsilon=0.0,
            metric='cosine'  # Better for text embeddings
        ))
    ]
    
    for description, config in configs:
        print(f"\n{description}:")
        print(f"   min_cluster_size: {config.min_cluster_size}")
        print(f"   min_samples: {config.min_samples}")
        print(f"   cluster_selection_epsilon: {config.cluster_selection_epsilon}")
        print(f"   metric: {config.metric}")

async def main():
    """Run the complete clustering demonstration."""
    show_configuration_examples()
    await demonstrate_clustering_workflow()

if __name__ == "__main__":
    asyncio.run(main())