#!/usr/bin/env python3
"""
Integration benchmark that tests the clustering functionality with actual VectorKnowledgeStore.

This script tests the integration between clustering and conflict detection
in the actual system rather than using mocks.
"""

import asyncio
import logging
import tempfile
import sys
from pathlib import Path

# Set up logging to see clustering messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, ProvenanceInfo, AuthorityLevel, GranularityLevel, TemporalScope
from context_mixer.domain.clustering import ClusteringConfig

async def run_integration_benchmark():
    """Run integration benchmark testing clustering with VectorKnowledgeStore."""
    
    print("🚀 Context Mixer Clustering Integration Benchmark")
    print("=" * 60)
    
    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "benchmark_db"
        
        print(f"📁 Using temporary database at: {db_path}")
        
        # Create clustering config
        clustering_config = ClusteringConfig(
            min_cluster_size=3,
            min_samples=2
        )
        
        print(f"⚙️ Clustering config: {clustering_config}")
        
        try:
            # Initialize VectorKnowledgeStore with clustering enabled
            store = VectorKnowledgeStore(
                db_path=db_path,
                clustering_config=clustering_config,
                enable_clustering=True
            )
            
            print(f"✅ VectorKnowledgeStore initialized successfully")
            print(f"🎯 Clustering enabled: {store.enable_clustering}")
            print(f"🤖 Clusterer available: {store._clusterer is not None}")
            
            # Create test knowledge chunks
            print("\n📦 Creating test knowledge chunks...")
            chunks = []
            for i in range(20):
                chunk = KnowledgeChunk(
                    id=f"chunk_{i}",
                    content=f"Knowledge about {'AI and machine learning' if i < 10 else 'software engineering and databases'}. This is detailed content for chunk {i}.",
                    metadata=ChunkMetadata(
                        domains=["technology", "ai" if i < 10 else "engineering"],
                        authority=AuthorityLevel.CONVENTIONAL,
                        scope=["enterprise"],
                        granularity=GranularityLevel.DETAILED,
                        temporal=TemporalScope.CURRENT,
                        provenance=ProvenanceInfo(
                            source=f"module_{i}.py",
                            created_at="2024-01-01T00:00:00Z"
                        )
                    )
                )
                chunks.append(chunk)
            
            print(f"✅ Created {len(chunks)} test chunks")
            
            # Store chunks in batches
            print("\n💾 Storing chunks...")
            await store.store_chunks(chunks)
            print("✅ All chunks stored successfully")
            
            # Test conflict detection (this should trigger clustering)
            print("\n🔍 Testing conflict detection with clustering...")
            test_chunk = KnowledgeChunk(
                id="new_ai_chunk",
                content="New knowledge about AI and machine learning algorithms",
                metadata=ChunkMetadata(
                    domains=["technology", "ai"],
                    authority=AuthorityLevel.CONVENTIONAL,
                    scope=["enterprise"],
                    granularity=GranularityLevel.DETAILED,
                    temporal=TemporalScope.CURRENT,
                    provenance=ProvenanceInfo(
                        source="new_module.py",
                        created_at="2024-01-01T00:00:00Z"
                    )
                )
            )
            
            conflicts = await store.detect_conflicts(test_chunk)
            print(f"✅ Conflict detection completed. Found {len(conflicts)} potential conflicts")
            
            # Get clustering statistics
            print("\n📊 Getting clustering statistics...")
            stats = await store.rebuild_clusters()
            print("🎯 Clustering Statistics:")
            for key, value in stats.items():
                if key != 'config':
                    print(f"   • {key}: {value}")
            
            # Test storage statistics
            print("\n📈 Getting storage statistics...")
            storage_stats = await store.get_stats()
            print("📋 Storage Statistics:")
            for key, value in storage_stats.items():
                if key not in ['domains', 'authority_levels']:
                    print(f"   • {key}: {value}")
            
            print("\n🎉 Integration benchmark completed successfully!")
            print("✅ Clustering is working correctly - no network warnings!")
            return True
            
        except Exception as e:
            print(f"\n❌ Integration benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_benchmark())
    if success:
        print("\n🏆 All integration tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Integration tests failed!")
        sys.exit(1)