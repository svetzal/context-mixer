"""
Tests for HDBSCAN clustering integration in VectorKnowledgeStore.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore
from context_mixer.domain.clustering import ClusteringConfig
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, AuthorityLevel


class DescribeVectorKnowledgeStoreWithClustering:
    """Test cases for VectorKnowledgeStore with clustering optimization."""

    @pytest.fixture
    def mock_gateway(self, mocker):
        """Mock ChromaGateway for testing."""
        gateway = mocker.MagicMock()
        gateway.store_knowledge_chunks = mocker.MagicMock()
        gateway.get_knowledge_chunk = mocker.MagicMock()
        gateway.search_knowledge = mocker.MagicMock()
        gateway.get_all_chunks = mocker.MagicMock(return_value=[])
        gateway.get_collection_stats = mocker.MagicMock(return_value={"total_chunks": 0, "collection_name": "test"})
        gateway.get_pool_stats = mocker.MagicMock(return_value={})
        gateway._get_embedding_for_chunk = mocker.MagicMock(return_value=[0.1, 0.2, 0.3])
        gateway._get_embeddings_for_chunks = mocker.MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        return gateway

    @pytest.fixture
    def clustering_config(self):
        """Clustering configuration for testing."""
        return ClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean'
        )

    @pytest.fixture
    def sample_chunks(self):
        """Sample knowledge chunks for testing."""
        chunks = []
        for i in range(5):
            metadata = ChunkMetadata(
                domains=["test"],
                authority=AuthorityLevel.OFFICIAL,
                project_id="test_project"
            )
            chunk = KnowledgeChunk(
                id=f"chunk_{i}",
                content=f"Test content {i}",
                metadata=metadata
            )
            chunks.append(chunk)
        return chunks

    @pytest.fixture
    def store_with_clustering(self, mock_gateway, clustering_config):
        """VectorKnowledgeStore with clustering enabled."""
        store = VectorKnowledgeStore(
            db_path=Path("/tmp/test"),
            clustering_config=clustering_config,
            enable_clustering=True
        )
        store._gateway = mock_gateway
        return store

    @pytest.fixture
    def store_without_clustering(self, mock_gateway):
        """VectorKnowledgeStore with clustering disabled."""
        store = VectorKnowledgeStore(
            db_path=Path("/tmp/test"),
            enable_clustering=False
        )
        store._gateway = mock_gateway
        return store

    def should_initialize_with_clustering_enabled(self, clustering_config):
        store = VectorKnowledgeStore(
            db_path=Path("/tmp/test"),
            clustering_config=clustering_config,
            enable_clustering=True
        )
        
        assert store.enable_clustering
        assert store.clustering_config == clustering_config
        assert store._clusterer is not None

    def should_initialize_with_clustering_disabled(self):
        store = VectorKnowledgeStore(
            db_path=Path("/tmp/test"),
            enable_clustering=False
        )
        
        assert not store.enable_clustering

    @patch('context_mixer.domain.vector_knowledge_store.KnowledgeClusterer')
    def should_handle_missing_hdbscan_gracefully(self, mock_clusterer_class):
        mock_clusterer_class.side_effect = ImportError("HDBSCAN not available")
        
        store = VectorKnowledgeStore(
            db_path=Path("/tmp/test"),
            enable_clustering=True
        )
        
        assert not store.enable_clustering

    async def should_mark_clusters_dirty_after_storing_chunks(self, store_with_clustering, sample_chunks):
        store_with_clustering._clusters_dirty = False
        
        await store_with_clustering.store_chunks(sample_chunks)
        
        assert store_with_clustering._clusters_dirty

    async def should_use_cluster_based_conflict_detection_when_enabled(self, store_with_clustering, sample_chunks):
        # Mock cluster-based detection method
        store_with_clustering._cluster_based_conflict_detection = AsyncMock(return_value=[])
        store_with_clustering._domain_based_conflict_detection = AsyncMock(return_value=[])
        
        chunk = sample_chunks[0]
        await store_with_clustering.detect_conflicts(chunk)
        
        store_with_clustering._cluster_based_conflict_detection.assert_called_once_with(chunk)
        store_with_clustering._domain_based_conflict_detection.assert_not_called()

    async def should_fallback_to_domain_based_detection_when_clustering_disabled(self, store_without_clustering, sample_chunks):
        # Mock domain-based detection method
        store_without_clustering._domain_based_conflict_detection = AsyncMock(return_value=[])
        
        chunk = sample_chunks[0]
        await store_without_clustering.detect_conflicts(chunk)
        
        store_without_clustering._domain_based_conflict_detection.assert_called_once_with(chunk)

    async def should_rebuild_clusters_when_dirty(self, store_with_clustering, sample_chunks):
        # Mock the clusterer
        store_with_clustering._clusterer.fit = Mock()
        store_with_clustering._clusterer.get_cluster_stats = Mock(return_value={"fitted": True})
        store_with_clustering._gateway.get_all_chunks.return_value = sample_chunks
        
        # Setup embeddings
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        store_with_clustering._gateway._get_embeddings_for_chunks.return_value = embeddings
        
        store_with_clustering._clusters_dirty = True
        
        await store_with_clustering._ensure_clusters_updated()
        
        assert not store_with_clustering._clusters_dirty
        store_with_clustering._clusterer.fit.assert_called_once()

    async def should_skip_clustering_when_insufficient_data(self, store_with_clustering):
        # Mock small dataset
        small_chunks = [Mock(id="chunk_1"), Mock(id="chunk_2")]
        store_with_clustering._gateway.get_all_chunks.return_value = small_chunks
        
        store_with_clustering._clusters_dirty = True
        
        await store_with_clustering._ensure_clusters_updated()
        
        # Should remain dirty since clustering was skipped
        assert store_with_clustering._clusters_dirty

    async def should_get_cluster_info_for_chunk(self, store_with_clustering, sample_chunks):
        # Mock fitted clusterer
        store_with_clustering._clusterer._fitted = True
        store_with_clustering._clusterer.get_cluster_for_chunk = Mock(return_value=1)
        store_with_clustering._clusterer.get_chunks_in_cluster = Mock(return_value={"chunk_0", "chunk_1"})
        store_with_clustering._clusters_dirty = False
        
        info = await store_with_clustering.get_cluster_info("chunk_0")
        
        assert info is not None
        assert info["chunk_id"] == "chunk_0"
        assert info["cluster_id"] == 1
        assert info["cluster_size"] == 2
        assert not info["is_noise"]

    async def should_return_none_for_cluster_info_when_clustering_disabled(self, store_without_clustering):
        info = await store_without_clustering.get_cluster_info("chunk_0")
        assert info is None

    async def should_include_clustering_stats_in_store_stats(self, store_with_clustering):
        # Mock clusterer stats
        store_with_clustering._clusterer.get_cluster_stats = Mock(return_value={
            "fitted": True,
            "total_clusters": 2,
            "total_chunks": 5
        })
        store_with_clustering._clusters_dirty = False
        
        stats = await store_with_clustering.get_stats()
        
        assert stats["clustering_enabled"]
        assert "clustering" in stats
        assert stats["clustering"]["fitted"]

    async def should_handle_cluster_rebuild_errors_gracefully(self, store_with_clustering):
        # Mock error during clustering
        store_with_clustering._gateway.get_all_chunks.side_effect = Exception("Test error")
        store_with_clustering._clusters_dirty = True
        
        # Should not raise exception
        await store_with_clustering._ensure_clusters_updated()
        
        # Should remain dirty due to error
        assert store_with_clustering._clusters_dirty

    async def should_manually_rebuild_clusters(self, store_with_clustering, sample_chunks):
        # Mock successful clustering
        store_with_clustering._clusterer.get_cluster_stats = Mock(return_value={
            "fitted": True,
            "total_clusters": 2
        })
        store_with_clustering._gateway.get_all_chunks.return_value = sample_chunks
        store_with_clustering._gateway._get_embeddings_for_chunks.return_value = [[0.1, 0.2]] * 5
        store_with_clustering._clusterer.fit = Mock()
        
        stats = await store_with_clustering.rebuild_clusters()
        
        assert stats["fitted"]
        assert stats["total_clusters"] == 2

    async def should_return_error_when_rebuilding_clusters_with_clustering_disabled(self, store_without_clustering):
        stats = await store_without_clustering.rebuild_clusters()
        assert "error" in stats
        assert "disabled" in stats["error"]