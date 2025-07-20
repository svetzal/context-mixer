"""
Test specifications for hierarchical knowledge clustering system.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List

from context_mixer.domain.clustering import (
    ContextualChunk, IntelligentCluster, ClusterQualityMonitor,
    DynamicConflictDetector, MockHDBSCANClusterer, ContextualDomain,
    ClusterStatus, ClusterQualityMetrics
)
from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, ProvenanceInfo, 
    GranularityLevel, TemporalScope
)


@pytest.fixture
def sample_chunks():
    """Create sample knowledge chunks for testing."""
    chunks = []
    
    for i in range(5):
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
def contextual_chunk(sample_chunks):
    """Create a sample contextual chunk."""
    return ContextualChunk(
        base_chunk=sample_chunks[0],
        cluster_id="test_cluster_1",
        contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
        boundary_confidence=0.8
    )


@pytest.fixture
def intelligent_cluster():
    """Create a sample intelligent cluster."""
    return IntelligentCluster(
        id="test_cluster_1",
        contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
        chunk_ids={"chunk_1", "chunk_2", "chunk_3"},
        knowledge_summary="Test cluster for technical implementation",
        status=ClusterStatus.STABLE
    )


class DescribeContextualChunk:
    """Test specifications for ContextualChunk."""
    
    def should_get_hierarchical_context(self, contextual_chunk):
        """Should return comprehensive hierarchical context information."""
        context = contextual_chunk.get_hierarchical_context()
        
        assert "chunk_id" in context
        assert "cluster_id" in context
        assert "domain" in context
        assert "authority" in context
        assert "boundary_confidence" in context
        assert context["chunk_id"] == contextual_chunk.base_chunk.id
        assert context["cluster_id"] == contextual_chunk.cluster_id
        assert context["domain"] == contextual_chunk.contextual_domain
    
    def should_detect_contextual_relationship_same_cluster(self, sample_chunks):
        """Should detect chunks in same cluster as contextually related."""
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1], 
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        assert chunk1.is_contextually_related(chunk2)
    
    def should_detect_contextual_relationship_same_parent(self, sample_chunks):
        """Should detect chunks with same parent cluster as contextually related."""
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_2", 
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        assert chunk1.is_contextually_related(chunk2)
    
    def should_detect_contextual_relationship_same_domain(self, sample_chunks):
        """Should detect chunks in same contextual domain as related."""
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_2",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        assert chunk1.is_contextually_related(chunk2)
    
    def should_not_detect_relationship_different_domains(self, sample_chunks):
        """Should not detect chunks in different domains as related by default."""
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_2", 
            contextual_domain=ContextualDomain.BUSINESS_PROCESS
        )
        
        assert not chunk1.is_contextually_related(chunk2)


class DescribeIntelligentCluster:
    """Test specifications for IntelligentCluster."""
    
    def should_get_cluster_intelligence(self, intelligent_cluster):
        """Should return comprehensive cluster intelligence information."""
        intelligence = intelligent_cluster.get_cluster_intelligence()
        
        assert "id" in intelligence
        assert "domain" in intelligence
        assert "chunk_count" in intelligence
        assert "knowledge_summary" in intelligence
        assert "authority_distribution" in intelligence
        assert "status" in intelligence
        
        assert intelligence["id"] == intelligent_cluster.id
        assert intelligence["chunk_count"] == len(intelligent_cluster.chunk_ids)
    
    def should_check_conflicts_with_same_cluster(self, intelligent_cluster):
        """Should always check conflicts within the same cluster."""
        assert intelligent_cluster.should_check_conflicts_with(intelligent_cluster)
    
    def should_check_conflicts_with_parent_child(self):
        """Should check conflicts between parent and child clusters."""
        parent_cluster = IntelligentCluster(
            id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_ARCHITECTURE,
            chunk_ids={"chunk_1", "chunk_2"}
        )
        
        child_cluster = IntelligentCluster(
            id="child_1",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_3", "chunk_4"}
        )
        
        assert parent_cluster.should_check_conflicts_with(child_cluster)
        assert child_cluster.should_check_conflicts_with(parent_cluster)
    
    def should_check_conflicts_with_siblings(self):
        """Should check conflicts between sibling clusters."""
        sibling1 = IntelligentCluster(
            id="sibling_1",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2"}
        )
        
        sibling2 = IntelligentCluster(
            id="sibling_2",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_3", "chunk_4"}
        )
        
        assert sibling1.should_check_conflicts_with(sibling2)
        assert sibling2.should_check_conflicts_with(sibling1)
    
    def should_check_conflicts_with_same_domain(self):
        """Should check conflicts between clusters in same contextual domain."""
        cluster1 = IntelligentCluster(
            id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2"}
        )
        
        cluster2 = IntelligentCluster(
            id="cluster_2",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_3", "chunk_4"}
        )
        
        assert cluster1.should_check_conflicts_with(cluster2)
    
    def should_not_check_conflicts_different_domains(self):
        """Should not check conflicts between unrelated clusters in different domains."""
        cluster1 = IntelligentCluster(
            id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2"},
            status=ClusterStatus.STABLE
        )
        
        cluster2 = IntelligentCluster(
            id="cluster_2", 
            contextual_domain=ContextualDomain.BUSINESS_PROCESS,
            chunk_ids={"chunk_3", "chunk_4"},
            status=ClusterStatus.STABLE  # Both clusters stable, different domains
        )
        
        # Should NOT check conflicts between different unrelated domains
        assert not cluster1.should_check_conflicts_with(cluster2)
    
    def should_get_graduated_similarity_thresholds(self):
        """Should return graduated similarity thresholds based on relationships."""
        # Same cluster
        cluster = IntelligentCluster(
            id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1"}
        )
        assert cluster.get_similarity_threshold(cluster) == 0.7
        
        # Sibling clusters (same parent)
        sibling1 = IntelligentCluster(
            id="sibling_1",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1"}
        )
        sibling2 = IntelligentCluster(
            id="sibling_2",
            parent_cluster_id="parent_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_2"}
        )
        assert sibling1.get_similarity_threshold(sibling2) == 0.8
        
        # Same domain, different parents
        domain_cluster1 = IntelligentCluster(
            id="domain_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1"}
        )
        domain_cluster2 = IntelligentCluster(
            id="domain_2",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_2"}
        )
        assert domain_cluster1.get_similarity_threshold(domain_cluster2) == 0.85
        
        # Different domains
        different_domain = IntelligentCluster(
            id="different",
            contextual_domain=ContextualDomain.BUSINESS_PROCESS,
            chunk_ids={"chunk_3"}
        )
        assert domain_cluster1.get_similarity_threshold(different_domain) == 0.9


class DescribeClusterQualityMonitor:
    """Test specifications for ClusterQualityMonitor."""
    
    def should_evaluate_single_chunk_cluster(self, sample_chunks):
        """Should handle single chunk clusters appropriately."""
        monitor = ClusterQualityMonitor()
        
        cluster = IntelligentCluster(
            id="single_cluster",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1"}
        )
        
        contextual_chunks = [ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="single_cluster",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            semantic_embedding=[0.1, 0.2, 0.3]
        )]
        
        embeddings = np.array([[0.1, 0.2, 0.3]])
        
        metrics = monitor.evaluate_cluster_quality(cluster, contextual_chunks, embeddings)
        
        assert metrics.silhouette_score == 0.0  # Single chunk, no silhouette
        assert metrics.domain_coherence == 1.0  # Single chunk is coherent
        assert metrics.size == 1
    
    def should_calculate_domain_coherence(self, sample_chunks):
        """Should calculate domain coherence based on chunk domains."""
        monitor = ClusterQualityMonitor()
        
        cluster = IntelligentCluster(
            id="test_cluster",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2", "chunk_3"}
        )
        
        # 2 chunks same domain, 1 different
        contextual_chunks = [
            ContextualChunk(
                base_chunk=sample_chunks[0],
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                semantic_embedding=[0.1, 0.2, 0.3]
            ),
            ContextualChunk(
                base_chunk=sample_chunks[1],
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                semantic_embedding=[0.2, 0.3, 0.4]
            ),
            ContextualChunk(
                base_chunk=sample_chunks[2],
                contextual_domain=ContextualDomain.BUSINESS_PROCESS,
                semantic_embedding=[0.3, 0.4, 0.5]
            )
        ]
        
        embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        
        metrics = monitor.evaluate_cluster_quality(cluster, contextual_chunks, embeddings)
        
        # 2 out of 3 chunks in dominant domain
        assert abs(metrics.domain_coherence - (2/3)) < 0.01
    
    def should_update_and_maintain_metrics_history(self):
        """Should update metrics history and maintain reasonable size."""
        monitor = ClusterQualityMonitor()
        cluster_id = "test_cluster"
        
        # Add multiple metrics
        for i in range(25):  # More than the 20 limit
            metrics = ClusterQualityMetrics(silhouette_score=0.5 + i * 0.01)
            monitor.update_metrics(cluster_id, metrics)
        
        # Should maintain only recent history
        assert len(monitor.metrics_history[cluster_id]) == 20
        
        # Should have the most recent metrics
        latest_score = monitor.metrics_history[cluster_id][-1].silhouette_score
        assert abs(latest_score - 0.74) < 0.01  # 0.5 + 24 * 0.01


class DescribeDynamicConflictDetector:
    """Test specifications for DynamicConflictDetector."""
    
    def should_skip_self_comparison(self, sample_chunks):
        """Should not check conflicts between a chunk and itself."""
        monitor = ClusterQualityMonitor()
        detector = DynamicConflictDetector(monitor)
        
        chunk = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        assert not detector.should_check_conflict(chunk, chunk)
    
    def should_use_cluster_level_logic_when_available(self, sample_chunks):
        """Should use cluster-level logic when clusters are provided."""
        monitor = ClusterQualityMonitor()
        detector = DynamicConflictDetector(monitor)
        
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        cluster1 = IntelligentCluster(
            id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2"}
        )
        
        # Should use cluster logic (same cluster = check conflicts)
        assert detector.should_check_conflict(chunk1, chunk2, cluster1, cluster1)
    
    def should_cache_conflict_decisions(self, sample_chunks):
        """Should cache conflict detection decisions for performance."""
        monitor = ClusterQualityMonitor()
        detector = DynamicConflictDetector(monitor)
        
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        # First call - should compute and cache
        result1 = detector.should_check_conflict(chunk1, chunk2)
        
        # Second call - should use cache
        result2 = detector.should_check_conflict(chunk1, chunk2)
        
        assert result1 == result2
        
        # Check cache was populated
        cache_key = tuple(sorted([chunk1.base_chunk.id, chunk2.base_chunk.id]))
        assert cache_key in detector.conflict_cache
    
    def should_optimize_conflict_candidates(self, sample_chunks):
        """Should return optimized subset of chunks for conflict checking."""
        monitor = ClusterQualityMonitor()
        detector = DynamicConflictDetector(monitor)
        
        # Create chunks in different clusters/domains
        target_chunk = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        related_chunk = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_1",  # Same cluster
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        unrelated_chunk = ContextualChunk(
            base_chunk=sample_chunks[2],
            cluster_id="cluster_2",
            contextual_domain=ContextualDomain.BUSINESS_PROCESS  # Different domain
        )
        
        all_chunks = [target_chunk, related_chunk, unrelated_chunk]
        
        clusters = {
            "cluster_1": IntelligentCluster(
                id="cluster_1",
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                chunk_ids={"chunk_1", "chunk_2"}
            ),
            "cluster_2": IntelligentCluster(
                id="cluster_2", 
                contextual_domain=ContextualDomain.BUSINESS_PROCESS,
                chunk_ids={"chunk_3"}
            )
        }
        
        candidates = detector.get_conflict_candidates(target_chunk, all_chunks, clusters)
        
        # Should only include related chunks
        assert len(candidates) == 1
        assert candidates[0].base_chunk.id == related_chunk.base_chunk.id
    
    def should_return_graduated_similarity_thresholds(self, sample_chunks):
        """Should return appropriate similarity thresholds based on relationships."""
        monitor = ClusterQualityMonitor()
        detector = DynamicConflictDetector(monitor)
        
        # Same cluster chunks
        chunk1 = ContextualChunk(
            base_chunk=sample_chunks[0],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        chunk2 = ContextualChunk(
            base_chunk=sample_chunks[1],
            cluster_id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION
        )
        
        cluster = IntelligentCluster(
            id="cluster_1",
            contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
            chunk_ids={"chunk_1", "chunk_2"}
        )
        
        threshold = detector.get_similarity_threshold(chunk1, chunk2, cluster, cluster)
        assert threshold == 0.7  # Same cluster threshold


class DescribeMockHDBSCANClusterer:
    """Test specifications for MockHDBSCANClusterer."""
    
    async def should_cluster_chunks_by_domain_and_authority(self, sample_chunks):
        """Should group chunks by domain and authority level."""
        clusterer = MockHDBSCANClusterer(min_cluster_size=2)
        
        # Add chunk with different authority
        different_authority_chunk = KnowledgeChunk(
            id="different_chunk",
            content="Different authority content",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.EXPERIMENTAL,  # Different authority
                scope=["test"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                tags=["different"],
                provenance=ProvenanceInfo(
                    source="different.py",
                    created_at=datetime.now().isoformat()
                )
            )
        )
        
        all_chunks = sample_chunks + [different_authority_chunk]
        
        contextual_chunks, clusters = await clusterer.cluster_chunks(all_chunks)
        
        # Should create clusters for groups that meet min_cluster_size
        assert len(contextual_chunks) == len(all_chunks)
        
        # Should have at least one cluster for the main group
        assert len(clusters) >= 1
        
        # Check that chunks in same domain/authority are clustered together
        main_group_chunks = [cc for cc in contextual_chunks if cc.cluster_id is not None]
        assert len(main_group_chunks) >= len(sample_chunks)  # Main group should be clustered
    
    async def should_assign_new_chunk_to_compatible_cluster(self, sample_chunks):
        """Should assign new chunk to most compatible existing cluster."""
        clusterer = MockHDBSCANClusterer()
        
        # Create existing clusters
        existing_clusters = {
            "cluster_1": IntelligentCluster(
                id="cluster_1",
                contextual_domain=ContextualDomain.TECHNICAL_IMPLEMENTATION,
                chunk_ids={"chunk_1", "chunk_2"},
                authority_distribution={AuthorityLevel.OFFICIAL: 2},
                status=ClusterStatus.STABLE,
                quality_metrics=ClusterQualityMetrics(silhouette_score=0.8)
            )
        }
        
        # Create new chunk that should match
        new_chunk = KnowledgeChunk(
            id="new_chunk",
            content="New technical content",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["test"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                tags=["new"],
                provenance=ProvenanceInfo(
                    source="new.py",
                    created_at=datetime.now().isoformat()
                )
            )
        )
        
        contextual_chunk = await clusterer.assign_chunk_to_cluster(new_chunk, existing_clusters)
        
        # Should assign to existing compatible cluster
        assert contextual_chunk.cluster_id == "cluster_1"
        assert contextual_chunk.contextual_domain == ContextualDomain.TECHNICAL_IMPLEMENTATION
        assert contextual_chunk.boundary_confidence > 0.5
    
    async def should_create_noise_for_incompatible_chunks(self, sample_chunks):
        """Should create noise assignment for chunks that don't fit existing clusters."""
        clusterer = MockHDBSCANClusterer()
        
        # Create existing cluster with different domain/authority
        existing_clusters = {
            "cluster_1": IntelligentCluster(
                id="cluster_1",
                contextual_domain=ContextualDomain.BUSINESS_PROCESS,
                chunk_ids={"chunk_1"},
                authority_distribution={AuthorityLevel.FOUNDATIONAL: 1},
                status=ClusterStatus.STABLE
            )
        }
        
        # Create incompatible chunk
        incompatible_chunk = KnowledgeChunk(
            id="incompatible",
            content="Technical experimental content",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.EXPERIMENTAL,
                scope=["test"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                tags=["experimental"],
                provenance=ProvenanceInfo(
                    source="experimental.py",
                    created_at=datetime.now().isoformat()
                )
            )
        )
        
        contextual_chunk = await clusterer.assign_chunk_to_cluster(incompatible_chunk, existing_clusters)
        
        # Should not assign to existing cluster
        assert contextual_chunk.cluster_id is None
        assert contextual_chunk.boundary_confidence < 0.5
    
    def should_map_domains_to_contextual_domains(self):
        """Should correctly map knowledge domains to contextual domains."""
        clusterer = MockHDBSCANClusterer()
        
        test_cases = [
            (["technical"], ContextualDomain.TECHNICAL_IMPLEMENTATION),
            (["architecture"], ContextualDomain.TECHNICAL_ARCHITECTURE),
            (["business"], ContextualDomain.BUSINESS_PROCESS),
            (["security"], ContextualDomain.SECURITY_FRAMEWORK),
            (["unknown_domain"], ContextualDomain.UNKNOWN)
        ]
        
        for domains, expected_domain in test_cases:
            result = clusterer._map_to_contextual_domain(domains)
            assert result == expected_domain
    
    def should_return_quality_threshold(self):
        """Should return minimum acceptable cluster quality threshold."""
        clusterer = MockHDBSCANClusterer()
        threshold = clusterer.get_cluster_quality_threshold()
        assert threshold == 0.3
        assert isinstance(threshold, float)