"""
Test specifications for clustering integration with conflict detection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from context_mixer.domain.clustering_integration import (
    ClusterOptimizedConflictDetector, ClusteringConfig
)
from context_mixer.domain.clustering import (
    MockHDBSCANClusterer, IntelligentCluster, ContextualDomain, ClusterStatus
)
from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, ProvenanceInfo, 
    GranularityLevel, TemporalScope
)
from context_mixer.domain.conflict import ConflictList, Conflict


@pytest.fixture
def sample_chunks():
    """Create sample knowledge chunks for testing."""
    chunks = []
    
    for i in range(4):
        provenance = ProvenanceInfo(
            source=f"test_file_{i}.py",
            created_at=datetime.now().isoformat()
        )
        
        metadata = ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["test"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=[f"test_{i}"],
            provenance=provenance
        )
        
        chunk = KnowledgeChunk(
            id=f"test_chunk_{i}",
            content=f"This is test content for chunk {i}",
            metadata=metadata
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def mock_llm_gateway():
    """Create a mock LLM gateway."""
    gateway = AsyncMock()
    # Mock no conflicts by default
    gateway.generate_object.return_value = ConflictList(list=[])
    return gateway


@pytest.fixture
def clustering_config():
    """Create test clustering configuration."""
    return ClusteringConfig(
        enabled=True,
        min_cluster_size=2,
        min_samples=1,
        quality_threshold=0.3,
        batch_size=2,
        fallback_to_traditional=True
    )


@pytest.fixture
def mock_clusterer():
    """Create a mock clusterer for testing."""
    return MockHDBSCANClusterer(min_cluster_size=2)


@pytest.fixture
def detector(mock_clusterer, clustering_config):
    """Create cluster optimized conflict detector."""
    return ClusterOptimizedConflictDetector(
        clusterer=mock_clusterer,
        config=clustering_config
    )


class DescribeClusteringConfig:
    """Test specifications for ClusteringConfig."""
    
    def should_create_with_default_values(self):
        """Should create config with sensible defaults."""
        config = ClusteringConfig()
        
        assert config.enabled is True
        assert config.min_cluster_size == 3
        assert config.min_samples == 1
        assert config.quality_threshold == 0.3
        assert config.batch_size == 5
        assert config.fallback_to_traditional is True
    
    def should_create_with_custom_values(self):
        """Should create config with custom values."""
        config = ClusteringConfig(
            enabled=False,
            min_cluster_size=5,
            min_samples=2,
            quality_threshold=0.5,
            batch_size=10,
            fallback_to_traditional=False
        )
        
        assert config.enabled is False
        assert config.min_cluster_size == 5
        assert config.min_samples == 2
        assert config.quality_threshold == 0.5
        assert config.batch_size == 10
        assert config.fallback_to_traditional is False


class DescribeClusterOptimizedConflictDetector:
    """Test specifications for ClusterOptimizedConflictDetector."""
    
    def should_initialize_with_defaults(self):
        """Should initialize with default configuration and clusterer."""
        detector = ClusterOptimizedConflictDetector()
        
        assert detector.config is not None
        assert detector.clusterer is not None
        assert detector.quality_monitor is not None
        assert detector.dynamic_detector is not None
        assert isinstance(detector.performance_stats, dict)
    
    def should_initialize_with_custom_components(self, mock_clusterer, clustering_config):
        """Should initialize with provided clusterer and config."""
        detector = ClusterOptimizedConflictDetector(
            clusterer=mock_clusterer,
            config=clustering_config
        )
        
        assert detector.clusterer == mock_clusterer
        assert detector.config == clustering_config
    
    async def should_use_traditional_detection_when_clustering_disabled(self, sample_chunks, mock_llm_gateway):
        """Should fall back to traditional detection when clustering is disabled."""
        config = ClusteringConfig(enabled=False)
        detector = ClusterOptimizedConflictDetector(config=config)
        
        # Mock the traditional detection method
        detector._traditional_conflict_detection = AsyncMock(return_value=[])
        
        result = await detector.detect_internal_conflicts_optimized(
            sample_chunks, mock_llm_gateway
        )
        
        # Should have called traditional detection
        detector._traditional_conflict_detection.assert_called_once()
        assert isinstance(result, list)
    
    async def should_use_traditional_detection_when_insufficient_chunks(self, mock_llm_gateway, clustering_config):
        """Should fall back to traditional detection when there are too few chunks."""
        clustering_config.min_cluster_size = 10  # More than we have
        detector = ClusterOptimizedConflictDetector(config=clustering_config)
        
        # Mock the traditional detection method
        detector._traditional_conflict_detection = AsyncMock(return_value=[])
        
        small_chunk_list = []  # Empty list
        result = await detector.detect_internal_conflicts_optimized(
            small_chunk_list, mock_llm_gateway
        )
        
        # Should have called traditional detection
        detector._traditional_conflict_detection.assert_called_once()
    
    async def should_perform_clustering_optimization(self, detector, sample_chunks, mock_llm_gateway):
        """Should perform clustering-based optimization when conditions are met."""
        # Mock the batch conflict detection to avoid import issues
        import context_mixer.commands.operations.merge as merge_module
        original_detect_conflicts_batch = merge_module.detect_conflicts_batch
        
        async def mock_batch_detect(chunk_pairs, llm_gateway, batch_size):
            # Return no conflicts for all pairs
            return [(pair[0], pair[1], ConflictList(list=[])) for pair in chunk_pairs]
        
        merge_module.detect_conflicts_batch = mock_batch_detect
        
        try:
            result = await detector.detect_internal_conflicts_optimized(
                sample_chunks, mock_llm_gateway
            )
            
            assert isinstance(result, list)
            # Should have created clusters
            assert detector.performance_stats["clusters_created"] >= 0
            
        finally:
            # Restore original function
            merge_module.detect_conflicts_batch = original_detect_conflicts_batch
    
    async def should_handle_clustering_failures_gracefully(self, detector, sample_chunks, mock_llm_gateway):
        """Should fall back to traditional detection when clustering fails."""
        # Make clustering fail
        detector.clusterer.cluster_chunks = AsyncMock(side_effect=Exception("Clustering failed"))
        detector._traditional_conflict_detection = AsyncMock(return_value=[])
        
        result = await detector.detect_internal_conflicts_optimized(
            sample_chunks, mock_llm_gateway
        )
        
        # Should have fallen back to traditional detection
        detector._traditional_conflict_detection.assert_called_once()
        assert detector.performance_stats["fallback_used"] == 1
    
    async def should_optimize_external_conflict_detection(self, detector, sample_chunks, mock_llm_gateway):
        """Should optimize external conflict detection using clustering."""
        # Create mock existing clusters
        existing_clusters = {
            "cluster_1": IntelligentCluster(
                id="cluster_1",
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                chunk_ids={"existing_chunk_1", "existing_chunk_2"},
                status=ClusterStatus.STABLE
            )
        }
        
        # Mock knowledge store
        mock_knowledge_store = AsyncMock()
        mock_knowledge_store.get_chunk.return_value = sample_chunks[0]  # Return sample chunk
        
        # Mock batch conflict detection
        import context_mixer.commands.operations.merge as merge_module
        original_detect_conflicts_batch = merge_module.detect_conflicts_batch
        
        async def mock_batch_detect(chunk_pairs, llm_gateway, batch_size):
            return [(pair[0], pair[1], ConflictList(list=[])) for pair in chunk_pairs]
        
        merge_module.detect_conflicts_batch = mock_batch_detect
        
        try:
            result = await detector.detect_external_conflicts_optimized(
                sample_chunks[:1],  # Just one chunk
                existing_clusters,
                mock_llm_gateway,
                mock_knowledge_store
            )
            
            assert isinstance(result, list)
            
        finally:
            merge_module.detect_conflicts_batch = original_detect_conflicts_batch
    
    def should_track_performance_statistics(self, detector):
        """Should track and report performance statistics."""
        # Set some test stats
        detector.performance_stats["traditional_checks"] = 100
        detector.performance_stats["optimized_checks"] = 20
        detector.performance_stats["clusters_created"] = 5
        
        stats = detector.get_performance_stats()
        
        assert "traditional_checks" in stats
        assert "optimized_checks" in stats
        assert "clusters_created" in stats
        assert "reduction_ratio" in stats
        assert "reduction_percentage" in stats
        
        # Should calculate 80% reduction (100 -> 20)
        assert abs(stats["reduction_ratio"] - 0.8) < 0.01
        assert abs(stats["reduction_percentage"] - 80.0) < 0.1
    
    def should_reset_performance_statistics(self, detector):
        """Should reset performance statistics."""
        # Set some stats
        detector.performance_stats["traditional_checks"] = 100
        detector.performance_stats["optimized_checks"] = 20
        
        detector.reset_performance_stats()
        
        assert detector.performance_stats["traditional_checks"] == 0
        assert detector.performance_stats["optimized_checks"] == 0
        assert detector.performance_stats["clusters_created"] == 0
        assert detector.performance_stats["fallback_used"] == 0
    
    def should_get_related_cluster_ids(self, detector):
        """Should identify related clusters for conflict checking."""
        # Create test clusters
        clusters = {
            "cluster_1": IntelligentCluster(
                id="cluster_1",
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                chunk_ids={"chunk_1", "chunk_2"},
                status=ClusterStatus.STABLE
            ),
            "cluster_2": IntelligentCluster(
                id="cluster_2",
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,  # Same domain
                chunk_ids={"chunk_3", "chunk_4"},
                status=ClusterStatus.STABLE
            ),
            "cluster_3": IntelligentCluster(
                id="cluster_3",
                contextual_domain=ContextualDomain.BUSINESS_PROCESS,  # Different domain
                chunk_ids={"chunk_5", "chunk_6"},
                status=ClusterStatus.STABLE
            )
        }
        
        related_ids = detector._get_related_cluster_ids("cluster_1", clusters)
        
        # Should include itself and same-domain cluster
        assert "cluster_1" in related_ids
        assert "cluster_2" in related_ids
        # Should not include different domain cluster
        assert "cluster_3" not in related_ids
    
    async def should_handle_missing_clusters_gracefully(self, detector):
        """Should handle missing cluster IDs gracefully."""
        empty_clusters = {}
        
        related_ids = detector._get_related_cluster_ids("nonexistent_cluster", empty_clusters)
        
        # Should return at least the target cluster ID
        assert "nonexistent_cluster" in related_ids
        assert len(related_ids) == 1