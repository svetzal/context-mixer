"""
Tests for clustering domain models.

This module tests the HDBSCAN clustering domain models and their functionality.
"""

import pytest
from datetime import datetime
from typing import List

from context_mixer.domain.clustering import (
    ClusterType, ClusterMetadata, KnowledgeCluster, ClusteringResult,
    ConflictDetectionCandidate, ClusterRelationship
)
from context_mixer.domain.knowledge import AuthorityLevel


class DescribeClusterType:
    """Test ClusterType enumeration."""

    def should_have_correct_values(self):
        assert ClusterType.KNOWLEDGE_DOMAIN == "knowledge_domain"
        assert ClusterType.CONTEXTUAL_SUBDOMAIN == "contextual_subdomain"
        assert ClusterType.SEMANTIC_CLUSTER == "semantic_cluster"


class DescribeClusterMetadata:
    """Test ClusterMetadata model."""

    @pytest.fixture
    def sample_metadata(self):
        return ClusterMetadata(
            cluster_type=ClusterType.SEMANTIC_CLUSTER,
            domains=["technical", "testing"],
            authority_levels={AuthorityLevel.OFFICIAL, AuthorityLevel.CONVENTIONAL},
            scopes=["backend", "api"],
            created_at=datetime.utcnow().isoformat(),
            chunk_count=5
        )

    def should_create_valid_metadata(self, sample_metadata):
        assert sample_metadata.cluster_type == ClusterType.SEMANTIC_CLUSTER
        assert "technical" in sample_metadata.domains
        assert "testing" in sample_metadata.domains
        assert AuthorityLevel.OFFICIAL in sample_metadata.authority_levels
        assert sample_metadata.chunk_count == 5

    def should_handle_empty_authority_levels(self):
        metadata = ClusterMetadata(
            cluster_type=ClusterType.KNOWLEDGE_DOMAIN,
            domains=["business"],
            authority_levels=set(),
            scopes=[],
            created_at=datetime.utcnow().isoformat()
        )
        assert len(metadata.authority_levels) == 0


class DescribeKnowledgeCluster:
    """Test KnowledgeCluster model."""

    @pytest.fixture
    def sample_cluster(self):
        metadata = ClusterMetadata(
            cluster_type=ClusterType.SEMANTIC_CLUSTER,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=["backend"],
            created_at=datetime.utcnow().isoformat(),
            chunk_count=3
        )

        return KnowledgeCluster(
            id="cluster-123",
            chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
            centroid=[0.1, 0.2, 0.3],
            summary="Technical backend guidelines",
            metadata=metadata,
            hdbscan_cluster_id=0,
            stability_score=0.85
        )

    def should_create_valid_cluster(self, sample_cluster):
        assert sample_cluster.id == "cluster-123"
        assert len(sample_cluster.chunk_ids) == 3
        assert sample_cluster.centroid == [0.1, 0.2, 0.3]
        assert sample_cluster.summary == "Technical backend guidelines"
        assert sample_cluster.hdbscan_cluster_id == 0
        assert sample_cluster.stability_score == 0.85

    def should_identify_leaf_cluster(self, sample_cluster):
        assert sample_cluster.is_leaf_cluster() is True

    def should_identify_non_leaf_cluster(self):
        metadata = ClusterMetadata(
            cluster_type=ClusterType.KNOWLEDGE_DOMAIN,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat(),
            chunk_count=0
        )

        cluster = KnowledgeCluster(
            id="parent-cluster",
            child_cluster_ids=["child-1", "child-2"],
            chunk_ids=[],  # No direct chunks
            metadata=metadata
        )

        assert cluster.is_leaf_cluster() is False

    def should_get_correct_conflict_detection_threshold(self):
        # Test semantic cluster threshold
        semantic_metadata = ClusterMetadata(
            cluster_type=ClusterType.SEMANTIC_CLUSTER,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat()
        )
        semantic_cluster = KnowledgeCluster(id="semantic", metadata=semantic_metadata, chunk_ids=["chunk-1"])
        assert semantic_cluster.get_conflict_detection_threshold() == 0.7

        # Test contextual subdomain threshold
        subdomain_metadata = ClusterMetadata(
            cluster_type=ClusterType.CONTEXTUAL_SUBDOMAIN,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat()
        )
        subdomain_cluster = KnowledgeCluster(id="subdomain", metadata=subdomain_metadata, chunk_ids=["chunk-1"])
        assert subdomain_cluster.get_conflict_detection_threshold() == 0.8

        # Test knowledge domain threshold
        domain_metadata = ClusterMetadata(
            cluster_type=ClusterType.KNOWLEDGE_DOMAIN,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat()
        )
        domain_cluster = KnowledgeCluster(id="domain", metadata=domain_metadata, chunk_ids=["chunk-1"])
        assert domain_cluster.get_conflict_detection_threshold() == 0.9


class DescribeClusteringResult:
    """Test ClusteringResult model."""

    @pytest.fixture
    def sample_clusters(self):
        metadata1 = ClusterMetadata(
            cluster_type=ClusterType.SEMANTIC_CLUSTER,
            domains=["technical"],
            authority_levels={AuthorityLevel.OFFICIAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat(),
            chunk_count=2
        )

        metadata2 = ClusterMetadata(
            cluster_type=ClusterType.KNOWLEDGE_DOMAIN,
            domains=["business"],
            authority_levels={AuthorityLevel.FOUNDATIONAL},
            scopes=[],
            created_at=datetime.utcnow().isoformat(),
            chunk_count=0
        )

        cluster1 = KnowledgeCluster(
            id="cluster-1",
            chunk_ids=["chunk-1", "chunk-2"],
            metadata=metadata1
        )

        cluster2 = KnowledgeCluster(
            id="cluster-2",
            child_cluster_ids=["cluster-1"],
            chunk_ids=[],
            metadata=metadata2
        )

        return [cluster1, cluster2]

    @pytest.fixture
    def sample_clustering_result(self, sample_clusters):
        return ClusteringResult(
            clusters=sample_clusters,
            noise_chunk_ids=["noise-1", "noise-2"],
            clustering_params={"min_cluster_size": 3},
            performance_metrics={"num_clusters": 2, "noise_ratio": 0.2}
        )

    def should_create_valid_clustering_result(self, sample_clustering_result):
        assert len(sample_clustering_result.clusters) == 2
        assert len(sample_clustering_result.noise_chunk_ids) == 2
        assert sample_clustering_result.clustering_params["min_cluster_size"] == 3
        assert sample_clustering_result.performance_metrics["num_clusters"] == 2

    def should_get_cluster_by_id(self, sample_clustering_result):
        cluster = sample_clustering_result.get_cluster_by_id("cluster-1")
        assert cluster is not None
        assert cluster.id == "cluster-1"

        missing_cluster = sample_clustering_result.get_cluster_by_id("nonexistent")
        assert missing_cluster is None

    def should_get_clusters_by_type(self, sample_clustering_result):
        semantic_clusters = sample_clustering_result.get_clusters_by_type(ClusterType.SEMANTIC_CLUSTER)
        assert len(semantic_clusters) == 1
        assert semantic_clusters[0].id == "cluster-1"

        domain_clusters = sample_clustering_result.get_clusters_by_type(ClusterType.KNOWLEDGE_DOMAIN)
        assert len(domain_clusters) == 1
        assert domain_clusters[0].id == "cluster-2"

    def should_get_leaf_clusters(self, sample_clustering_result):
        leaf_clusters = sample_clustering_result.get_leaf_clusters()
        assert len(leaf_clusters) == 1
        assert leaf_clusters[0].id == "cluster-1"  # Only cluster with chunks


class DescribeConflictDetectionCandidate:
    """Test ConflictDetectionCandidate model."""

    @pytest.fixture
    def sample_candidate(self):
        return ConflictDetectionCandidate(
            chunk1_id="chunk-1",
            chunk2_id="chunk-2",
            cluster1_id="cluster-1",
            cluster2_id="cluster-1",
            similarity_threshold=0.7,
            priority_score=1.0,
            relationship_context="same-cluster"
        )

    def should_create_valid_candidate(self, sample_candidate):
        assert sample_candidate.chunk1_id == "chunk-1"
        assert sample_candidate.chunk2_id == "chunk-2"
        assert sample_candidate.cluster1_id == "cluster-1"
        assert sample_candidate.cluster2_id == "cluster-1"
        assert sample_candidate.similarity_threshold == 0.7
        assert sample_candidate.priority_score == 1.0
        assert sample_candidate.relationship_context == "same-cluster"

    def should_handle_noise_chunk_candidate(self):
        candidate = ConflictDetectionCandidate(
            chunk1_id="noise-chunk",
            chunk2_id="cluster-chunk",
            cluster1_id=None,  # Noise chunk
            cluster2_id="cluster-1",
            similarity_threshold=0.9,
            priority_score=0.3,
            relationship_context="noise-to-cluster"
        )

        assert candidate.cluster1_id is None
        assert candidate.cluster2_id == "cluster-1"
        assert candidate.similarity_threshold == 0.9
        assert candidate.priority_score == 0.3


class DescribeClusterRelationship:
    """Test ClusterRelationship model."""

    def should_create_valid_relationship(self):
        relationship = ClusterRelationship(
            cluster1_id="cluster-1",
            cluster2_id="cluster-2",
            relationship_type="sibling",
            similarity_score=0.6,
            conflict_detection_threshold=0.8
        )

        assert relationship.cluster1_id == "cluster-1"
        assert relationship.cluster2_id == "cluster-2"
        assert relationship.relationship_type == "sibling"
        assert relationship.similarity_score == 0.6
        assert relationship.conflict_detection_threshold == 0.8
