"""
Tests for HDBSCAN-based clustering functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from context_mixer.domain.clustering import (
    KnowledgeClusterer,
    ClusteringConfig,
    ClusterMetadata
)


class DescribeKnowledgeClusterer:
    """Test cases for KnowledgeClusterer."""

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        # Create some clustered data
        np.random.seed(42)
        
        # Cluster 1: around (0, 0)
        cluster1 = np.random.normal(0, 0.2, (8, 2))
        
        # Cluster 2: around (2, 2)  
        cluster2 = np.random.normal(2, 0.2, (6, 2))
        
        # Some noise points
        noise = np.random.uniform(-1, 3, (3, 2))
        
        embeddings = np.vstack([cluster1, cluster2, noise])
        chunk_ids = [f"chunk_{i}" for i in range(len(embeddings))]
        
        return embeddings, chunk_ids

    @pytest.fixture
    def small_embeddings(self):
        """Small dataset for testing edge cases."""
        embeddings = np.array([[0, 0], [1, 1]])
        chunk_ids = ["chunk_0", "chunk_1"]
        return embeddings, chunk_ids

    @pytest.fixture
    def config(self):
        """Default clustering configuration."""
        return ClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            cluster_selection_epsilon=0.0,
            metric='euclidean'
        )

    def should_initialize_with_default_config(self):
        clusterer = KnowledgeClusterer()
        assert clusterer.config.min_cluster_size == 5
        assert clusterer.config.min_samples == 3
        assert not clusterer._fitted

    def should_initialize_with_custom_config(self, config):
        clusterer = KnowledgeClusterer(config)
        assert clusterer.config.min_cluster_size == 3
        assert clusterer.config.min_samples == 2

    @patch('context_mixer.domain.clustering.hdbscan', None)
    def should_raise_error_when_hdbscan_not_available(self):
        with pytest.raises(ImportError, match="HDBSCAN is required"):
            KnowledgeClusterer()

    def should_fit_clusterer_on_sample_data(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        
        cluster_metadata = clusterer.fit(embeddings, chunk_ids)
        
        assert clusterer._fitted
        assert len(cluster_metadata) > 0
        
        # Should have found some clusters
        valid_clusters = [c for c in cluster_metadata.keys() if c != -1]
        assert len(valid_clusters) >= 1

    def should_handle_small_dataset_gracefully(self, small_embeddings, config):
        embeddings, chunk_ids = small_embeddings
        clusterer = KnowledgeClusterer(config)
        
        cluster_metadata = clusterer.fit(embeddings, chunk_ids)
        
        assert clusterer._fitted
        # Should treat all as noise due to insufficient data
        assert -1 in cluster_metadata
        assert cluster_metadata[-1].size == 2

    def should_raise_error_for_mismatched_inputs(self, config):
        embeddings = np.array([[1, 2], [3, 4]])
        chunk_ids = ["chunk_1"]  # Wrong length
        
        clusterer = KnowledgeClusterer(config)
        
        with pytest.raises(ValueError, match="same length"):
            clusterer.fit(embeddings, chunk_ids)

    def should_predict_cluster_for_new_embedding(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        clusterer.fit(embeddings, chunk_ids)
        
        # Test prediction near cluster 1 (around 0, 0)
        new_embedding = np.array([0.1, 0.1])
        cluster_id, confidence = clusterer.predict_cluster(new_embedding)
        
        assert isinstance(cluster_id, int)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def should_raise_error_when_predicting_without_fitting(self, config):
        clusterer = KnowledgeClusterer(config)
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            clusterer.predict_cluster(np.array([1, 2]))

    def should_get_cluster_for_chunk(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        clusterer.fit(embeddings, chunk_ids)
        
        cluster_id = clusterer.get_cluster_for_chunk(chunk_ids[0])
        assert isinstance(cluster_id, int)

    def should_return_minus_one_for_unknown_chunk(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        clusterer.fit(embeddings, chunk_ids)
        
        cluster_id = clusterer.get_cluster_for_chunk("unknown_chunk")
        assert cluster_id == -1

    def should_get_chunks_in_cluster(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        cluster_metadata = clusterer.fit(embeddings, chunk_ids)
        
        # Get a valid cluster ID
        valid_clusters = [c for c in cluster_metadata.keys() if c != -1]
        if valid_clusters:
            cluster_id = valid_clusters[0]
            chunks_in_cluster = clusterer.get_chunks_in_cluster(cluster_id)
            
            assert isinstance(chunks_in_cluster, set)
            assert len(chunks_in_cluster) > 0
            assert all(chunk_id in chunk_ids for chunk_id in chunks_in_cluster)

    def should_return_empty_set_for_invalid_cluster(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        clusterer.fit(embeddings, chunk_ids)
        
        chunks = clusterer.get_chunks_in_cluster(999)
        assert chunks == set()

    def should_get_nearby_clusters(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        cluster_metadata = clusterer.fit(embeddings, chunk_ids)
        
        valid_clusters = [c for c in cluster_metadata.keys() if c != -1]
        if len(valid_clusters) > 1:
            nearby = clusterer.get_nearby_clusters(valid_clusters[0])
            assert isinstance(nearby, list)
            # Should not include the source cluster itself
            assert valid_clusters[0] not in nearby

    def should_get_cluster_stats(self, sample_embeddings, config):
        embeddings, chunk_ids = sample_embeddings
        clusterer = KnowledgeClusterer(config)
        
        # Before fitting
        stats = clusterer.get_cluster_stats()
        assert not stats["fitted"]
        
        # After fitting
        clusterer.fit(embeddings, chunk_ids)
        stats = clusterer.get_cluster_stats()
        
        assert stats["fitted"]
        assert "total_clusters" in stats
        assert "total_chunks" in stats
        assert "noise_chunks" in stats
        assert "config" in stats
        assert stats["total_chunks"] == len(chunk_ids)


class DescribeClusteringConfig:
    """Test cases for ClusteringConfig."""

    def should_create_with_defaults(self):
        config = ClusteringConfig()
        assert config.min_cluster_size == 5
        assert config.min_samples == 3
        assert config.cluster_selection_epsilon == 0.0
        assert config.metric == 'euclidean'
        assert config.alpha == 1.0
        assert config.prediction_data

    def should_validate_min_cluster_size(self):
        with pytest.raises(ValueError):
            ClusteringConfig(min_cluster_size=1)  # Should be >= 2

    def should_validate_min_samples(self):
        with pytest.raises(ValueError):
            ClusteringConfig(min_samples=0)  # Should be >= 1

    def should_validate_alpha(self):
        with pytest.raises(ValueError):
            ClusteringConfig(alpha=0.0)  # Should be > 0


class DescribeClusterMetadata:
    """Test cases for ClusterMetadata."""

    def should_create_cluster_metadata(self):
        chunk_ids = {"chunk_1", "chunk_2", "chunk_3"}
        metadata = ClusterMetadata(
            cluster_id=1,
            size=3,
            persistence=0.8,
            representative_chunk_id="chunk_1",
            chunk_ids=chunk_ids
        )
        
        assert metadata.cluster_id == 1
        assert metadata.size == 3
        assert metadata.persistence == 0.8
        assert metadata.representative_chunk_id == "chunk_1"
        assert metadata.chunk_ids == chunk_ids

    def should_create_minimal_cluster_metadata(self):
        metadata = ClusterMetadata(
            cluster_id=-1,
            size=0,
            persistence=0.0
        )
        
        assert metadata.cluster_id == -1
        assert metadata.size == 0
        assert metadata.persistence == 0.0
        assert metadata.representative_chunk_id is None
        assert metadata.chunk_ids is None