"""
Tests for clustering integration with conflict detection optimization.

This module tests the integration of HDBSCAN clustering with the existing
conflict detection system to ensure proper optimization and fallback behavior.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import numpy as np

from context_mixer.domain.clustering_integration import (
    ClusteringConfig, ClusterOptimizedConflictDetector, ClusteringStatistics
)
from context_mixer.domain.clustering import KnowledgeCluster, ClusteringResult, ClusterType, ClusterMetadata
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, AuthorityLevel, TemporalScope, ProvenanceInfo, GranularityLevel
from context_mixer.gateways.llm import LLMGateway


class DescribeClusteringConfig:
    """Test cases for ClusteringConfig."""

    def should_create_default_config(self):
        config = ClusteringConfig()
        
        assert config.enabled is True
        assert config.min_cluster_size == 3
        assert config.min_samples == 1
        assert config.quality_threshold == 0.3
        assert config.batch_size == 5
        assert config.fallback_to_traditional is True

    def should_create_custom_config(self):
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
    """Test cases for ClusterOptimizedConflictDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_clustering_service = MagicMock()
        self.mock_clustering_service.cluster_knowledge_chunks = AsyncMock()
        self.mock_llm_gateway = MagicMock(spec=LLMGateway)
        self.config = ClusteringConfig(enabled=True, batch_size=2)
        
        self.detector = ClusterOptimizedConflictDetector(
            self.mock_clustering_service,
            self.mock_llm_gateway,
            self.config
        )
        
        # Create test chunks
        self.target_chunk = KnowledgeChunk(
            id="target-1",
            content="Target chunk content",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["project"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=ProvenanceInfo(
                    source="test.py",
                    created_at="2024-01-01T00:00:00Z"
                )
            )
        )
        
        self.existing_chunks = [
            KnowledgeChunk(
                id=f"chunk-{i}",
                content=f"Existing chunk {i}",
                metadata=ChunkMetadata(
                    domains=["technical"],
                    authority=AuthorityLevel.OFFICIAL,
                    scope=["project"],
                    granularity=GranularityLevel.DETAILED,
                    temporal=TemporalScope.CURRENT,
                    provenance=ProvenanceInfo(
                        source=f"test{i}.py",
                        created_at="2024-01-01T00:00:00Z"
                    )
                )
            )
            for i in range(5)
        ]

    @pytest.mark.asyncio
    async def should_detect_conflicts_with_clustering_optimization(self):
        """Test successful conflict detection using clustering."""
        # Mock clustering result
        cluster = KnowledgeCluster(
            id="cluster-1",
            chunk_ids=[self.target_chunk.id, self.existing_chunks[0].id, self.existing_chunks[1].id],
            metadata=ClusterMetadata(
                cluster_type=ClusterType.SEMANTIC_CLUSTER,
                domains=["technical"],
                authority_levels={AuthorityLevel.OFFICIAL},
                scopes=["project"],
                created_at="2024-01-01T00:00:00",
                chunk_count=3
            )
        )
        
        clustering_result = ClusteringResult(
            clusters=[cluster],
            noise_chunk_ids=[],
            clustering_params={},
            performance_metrics={}
        )
        
        # Mock the clustering service
        self.mock_clustering_service.cluster_knowledge_chunks = AsyncMock(return_value=clustering_result)
        
        # Mock conflict detection to return one conflict
        from context_mixer.domain.conflict import ConflictList, Conflict, ConflictingGuidance
        from context_mixer.domain.context import Context
        
        conflict_list = ConflictList(list=[
            Conflict(
                description="Test conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="existing", source="existing"),
                    ConflictingGuidance(content="new", source="new")
                ]
            )
        ])
        
        mock_detect_conflicts_batch = AsyncMock(return_value=[
            (self.target_chunk, self.existing_chunks[0], conflict_list)
        ])
        
        # Patch the detect_conflicts_batch function
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                'context_mixer.domain.clustering_integration.detect_conflicts_batch',
                mock_detect_conflicts_batch
            )
            
            conflicts, stats = await self.detector.detect_conflicts_optimized(
                self.target_chunk,
                self.existing_chunks,
                use_cache=False,
                max_candidates=10
            )
        
        # Verify results
        assert len(conflicts) == 1
        assert conflicts[0].id == "chunk-0"
        
        # Verify statistics
        assert stats.total_chunks_clustered == 6  # target + 5 existing
        assert stats.clusters_created == 1
        assert stats.traditional_comparisons_avoided == 3  # 5 - 2 candidates
        assert stats.optimization_percentage == 60.0  # 3/5 * 100
        assert stats.fallback_used is False
        
        # Verify clustering service was called
        self.mock_clustering_service.cluster_knowledge_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def should_fallback_when_clustering_disabled(self):
        """Test fallback to traditional detection when clustering is disabled."""
        config = ClusteringConfig(enabled=False)
        detector = ClusterOptimizedConflictDetector(
            self.mock_clustering_service,
            self.mock_llm_gateway,
            config
        )
        
        # Mock traditional conflict detection
        from context_mixer.domain.conflict import ConflictList, Conflict, ConflictingGuidance
        
        conflict_list = ConflictList(list=[
            Conflict(
                description="Test conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="existing", source="existing"),
                    ConflictingGuidance(content="new", source="new")
                ]
            )
        ])
        
        mock_detect_conflicts_batch = AsyncMock(return_value=[
            (self.target_chunk, self.existing_chunks[0], conflict_list)
        ])
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                'context_mixer.domain.clustering_integration.detect_conflicts_batch',
                mock_detect_conflicts_batch
            )
            
            conflicts, stats = await detector.detect_conflicts_optimized(
                self.target_chunk,
                self.existing_chunks
            )
        
        # Verify fallback was used
        assert stats.fallback_used is True
        assert len(conflicts) == 1
        
        # Verify clustering service was not called
        self.mock_clustering_service.cluster_knowledge_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def should_fallback_when_clustering_fails(self):
        """Test fallback when clustering operation fails."""
        # Mock clustering service to raise an exception
        self.mock_clustering_service.cluster_knowledge_chunks = AsyncMock(
            side_effect=Exception("Clustering failed")
        )
        
        # Mock traditional conflict detection
        from context_mixer.domain.conflict import ConflictList
        
        empty_conflict_list = ConflictList(list=[])
        
        mock_detect_conflicts_batch = AsyncMock(return_value=[
            (self.target_chunk, self.existing_chunks[0], empty_conflict_list)
        ])
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                'context_mixer.domain.clustering_integration.detect_conflicts_batch',
                mock_detect_conflicts_batch
            )
            
            conflicts, stats = await self.detector.detect_conflicts_optimized(
                self.target_chunk,
                self.existing_chunks
            )
        
        # Verify fallback was used
        assert stats.fallback_used is True
        assert conflicts == []

    @pytest.mark.asyncio
    async def should_handle_empty_existing_chunks(self):
        """Test handling of empty existing chunks list."""
        conflicts, stats = await self.detector.detect_conflicts_optimized(
            self.target_chunk,
            [],
            use_cache=False
        )
        
        assert conflicts == []
        assert stats.fallback_used is True

    @pytest.mark.asyncio
    async def should_use_cluster_cache_when_enabled(self):
        """Test that cluster caching works correctly."""
        # First call - should cluster and cache
        cluster = KnowledgeCluster(
            id="cluster-1",
            chunk_ids=[self.target_chunk.id],
            metadata=ClusterMetadata(
                cluster_type=ClusterType.SEMANTIC_CLUSTER,
                domains=["technical"],
                authority_levels={AuthorityLevel.OFFICIAL},
                scopes=["project"],
                created_at="2024-01-01T00:00:00",
                chunk_count=1
            )
        )
        
        clustering_result = ClusteringResult(
            clusters=[cluster],
            noise_chunk_ids=[],
            clustering_params={},
            performance_metrics={}
        )
        
        self.mock_clustering_service.cluster_knowledge_chunks = AsyncMock(return_value=clustering_result)
        
        # Mock conflict detection
        from context_mixer.domain.conflict import ConflictList
        
        empty_conflict_list = ConflictList(list=[])
        
        mock_detect_conflicts_batch = AsyncMock(return_value=[
            (self.target_chunk, self.existing_chunks[0], empty_conflict_list)
        ])
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                'context_mixer.domain.clustering_integration.detect_conflicts_batch',
                mock_detect_conflicts_batch
            )
            
            # First call
            await self.detector.detect_conflicts_optimized(
                self.target_chunk,
                self.existing_chunks,
                use_cache=True
            )
            
            # Second call with same chunks - should use cache
            await self.detector.detect_conflicts_optimized(
                self.target_chunk,
                self.existing_chunks,
                use_cache=True
            )
        
        # Clustering service should only be called once due to caching
        assert self.mock_clustering_service.cluster_knowledge_chunks.call_count == 1

    def should_clear_cache(self):
        """Test cache clearing functionality."""
        # Add something to cache first
        self.detector._cluster_cache["test"] = "value"
        assert len(self.detector._cluster_cache) == 1
        
        # Clear cache
        self.detector.clear_cache()
        assert len(self.detector._cluster_cache) == 0

    def should_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.detector.get_statistics()
        assert isinstance(stats, ClusteringStatistics)
        assert stats.total_chunks_clustered == 0
        assert stats.clusters_created == 0


class DescribeClusteringStatistics:
    """Test cases for ClusteringStatistics."""

    def should_create_default_statistics(self):
        stats = ClusteringStatistics()
        
        assert stats.total_chunks_clustered == 0
        assert stats.clusters_created == 0
        assert stats.traditional_comparisons_avoided == 0
        assert stats.clustering_time == 0.0
        assert stats.conflict_detection_time == 0.0
        assert stats.optimization_percentage == 0.0
        assert stats.fallback_used is False

    def should_calculate_optimization_percentage(self):
        stats = ClusteringStatistics()
        stats.traditional_comparisons_avoided = 75
        
        # Simulate calculating optimization percentage
        total_possible = 100
        stats.optimization_percentage = (stats.traditional_comparisons_avoided / total_possible) * 100.0
        
        assert stats.optimization_percentage == 75.0