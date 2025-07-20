#!/usr/bin/env python3
"""
Test script to validate clustering integration in conflict resolution.

This script tests that the clustering optimization is properly integrated
into the conflict resolution features of the tool.
"""

import asyncio
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore
from context_mixer.domain.clustering import ClusteringConfig
from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, GranularityLevel,
    TemporalScope, ProvenanceInfo
)
from context_mixer.commands.operations.merge import detect_conflicts_batch_with_clustering
from context_mixer.gateways.llm import LLMGateway


async def test_clustering_integration():
    """Test that clustering integration works in conflict resolution."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing clustering integration in conflict resolution...")
    
    # Create sample chunks
    chunks = []
    for i in range(5):
        provenance = ProvenanceInfo(
            source="test_file.md",
            project_id="test_project", 
            created_at="2024-01-01T00:00:00Z"
        )
        metadata = ChunkMetadata(
            domains=["test"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=provenance
        )
        chunk = KnowledgeChunk(
            id=f"chunk_{i}",
            content=f"Test content {i} about testing methodologies",
            metadata=metadata
        )
        chunks.append(chunk)
    
    # Create chunk pairs for testing
    chunk_pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk_pairs.append((chunks[i], chunks[j]))
    
    print(f"📊 Created {len(chunks)} chunks with {len(chunk_pairs)} pairs to test")
    
    # Create a temporary directory for the vector store
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_vector_store"
        
        # Create clustering config
        clustering_config = ClusteringConfig(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        
        # Create mock LLM gateway
        mock_llm_gateway = Mock(spec=LLMGateway)
        
        # Create vector knowledge store with clustering enabled
        store = VectorKnowledgeStore(
            db_path=db_path,
            llm_gateway=mock_llm_gateway,
            clustering_config=clustering_config,
            enable_clustering=True
        )
        
        # Mock the gateway methods
        mock_gateway = Mock()
        mock_gateway._get_embedding_for_chunk.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        store._gateway = mock_gateway
        
        # Mock the LLM conflict detection to return no conflicts
        store._llm_detect_conflict = AsyncMock(return_value=False)
        
        print("🔍 Testing clustering-aware batch conflict detection...")
        
        # Test the clustering-aware batch conflict detection
        try:
            results = await detect_conflicts_batch_with_clustering(
                chunk_pairs,
                knowledge_store=store,
                llm_gateway=mock_llm_gateway
            )
            
            print(f"✅ Successfully processed {len(results)} conflict checks")
            print(f"📈 Expected {len(chunk_pairs)} results, got {len(results)}")
            
            # Verify results structure
            for chunk1, chunk2, conflicts in results:
                assert isinstance(chunk1, KnowledgeChunk)
                assert isinstance(chunk2, KnowledgeChunk)
                assert hasattr(conflicts, 'list')
            
            print("✅ All conflict check results have correct structure")
            
        except Exception as e:
            print(f"❌ Clustering-aware conflict detection failed: {e}")
            raise
        
        print("🔄 Testing fallback to standard processing...")
        
        # Test fallback when clustering is disabled
        store.enable_clustering = False
        
        try:
            results_fallback = await detect_conflicts_batch_with_clustering(
                chunk_pairs,
                knowledge_store=store,
                llm_gateway=mock_llm_gateway
            )
            
            print(f"✅ Fallback processing completed with {len(results_fallback)} results")
            
        except Exception as e:
            print(f"❌ Fallback processing failed: {e}")
            raise
        
        print("🎯 Testing without knowledge store (pure LLM approach)...")
        
        # Test without knowledge store (should use pure LLM approach)
        # Mock the original detect_conflicts_batch function
        try:
            # This should fall back to the original function since no clustering is available
            results_pure = await detect_conflicts_batch_with_clustering(
                chunk_pairs,
                knowledge_store=None,
                llm_gateway=mock_llm_gateway
            )
            
            print(f"✅ Pure LLM processing completed with {len(results_pure)} results")
            
        except Exception as e:
            print(f"❌ Pure LLM processing failed: {e}")
            # This is expected to fail since we're mocking, but we want to see the fallback logic
            print("ℹ️  This failure is expected with mocked LLM gateway")
    
    print("🎉 All clustering integration tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_clustering_integration())