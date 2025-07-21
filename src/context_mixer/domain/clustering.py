"""
Domain models for HDBSCAN-based knowledge clustering.

This module implements hierarchical domain-aware clustering to optimize conflict detection
by grouping semantically related knowledge chunks and reducing expensive pairwise comparisons.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field
from .knowledge import KnowledgeChunk, AuthorityLevel


class ClusterType(str, Enum):
    """Types of clusters in the hierarchical clustering system."""
    KNOWLEDGE_DOMAIN = "knowledge_domain"      # Top-level: business, technical, operational, legal
    CONTEXTUAL_SUBDOMAIN = "contextual_subdomain"  # Mid-level: architecture, testing, deployment
    SEMANTIC_CLUSTER = "semantic_cluster"      # Bottom-level: specific semantic groupings


class ClusterMetadata(BaseModel):
    """Metadata for knowledge clusters."""
    cluster_type: ClusterType = Field(..., description="Type of cluster in hierarchy")
    domains: List[str] = Field(..., description="Knowledge domains represented")
    authority_levels: Set[AuthorityLevel] = Field(..., description="Authority levels in cluster")
    scopes: List[str] = Field(..., description="Applicable scopes")
    created_at: str = Field(..., description="When cluster was created")
    updated_at: Optional[str] = Field(None, description="When cluster was last updated")
    chunk_count: int = Field(0, description="Number of chunks in this cluster")


class KnowledgeCluster(BaseModel):
    """
    A hierarchical cluster of semantically related knowledge chunks.
    
    Implements the hierarchical clustering architecture from PLAN.md:
    Knowledge Domains → Contextual Sub-domains → Semantic Clusters → Knowledge Chunks
    """
    id: str = Field(..., description="Unique identifier for this cluster")
    parent_cluster_id: Optional[str] = Field(None, description="Parent cluster in hierarchy")
    child_cluster_ids: List[str] = Field(default_factory=list, description="Child clusters")
    chunk_ids: List[str] = Field(default_factory=list, description="Knowledge chunks in this cluster")
    centroid: Optional[List[float]] = Field(None, description="Cluster centroid embedding")
    summary: Optional[str] = Field(None, description="Concise summary of knowledge in cluster")
    metadata: ClusterMetadata = Field(..., description="Cluster metadata")
    
    # HDBSCAN-specific properties
    hdbscan_cluster_id: Optional[int] = Field(None, description="Original HDBSCAN cluster ID")
    stability_score: Optional[float] = Field(None, description="HDBSCAN stability score")
    
    def get_hierarchical_level(self) -> int:
        """Get the hierarchical level (0=root, 1=domain, 2=subdomain, 3=semantic)."""
        if self.parent_cluster_id is None:
            return 0
        # This would need to be calculated by traversing up the hierarchy
        return 1  # Simplified for now
    
    def is_leaf_cluster(self) -> bool:
        """Check if this is a leaf cluster (contains chunks, not other clusters)."""
        return len(self.chunk_ids) > 0 and len(self.child_cluster_ids) == 0
    
    def get_conflict_detection_threshold(self) -> float:
        """
        Get the similarity threshold for conflict detection based on cluster relationships.
        
        From PLAN.md:
        - Same semantic cluster: 0.7 (high semantic similarity expected)
        - Cross-cluster, same architectural domain: 0.8 (moderate restriction)
        - Cross-domain contexts: 0.9 (high restriction, rare valid conflicts)
        """
        if self.metadata.cluster_type == ClusterType.SEMANTIC_CLUSTER:
            return 0.7
        elif self.metadata.cluster_type == ClusterType.CONTEXTUAL_SUBDOMAIN:
            return 0.8
        else:  # KNOWLEDGE_DOMAIN
            return 0.9


class ClusterRelationship(BaseModel):
    """Represents a relationship between two clusters."""
    cluster1_id: str = Field(..., description="First cluster ID")
    cluster2_id: str = Field(..., description="Second cluster ID")
    relationship_type: str = Field(..., description="Type of relationship (parent-child, sibling, cross-domain)")
    similarity_score: Optional[float] = Field(None, description="Semantic similarity between clusters")
    conflict_detection_threshold: float = Field(..., description="Threshold for conflict detection")


class ClusteringResult(BaseModel):
    """Result of HDBSCAN clustering operation."""
    clusters: List[KnowledgeCluster] = Field(..., description="Generated clusters")
    noise_chunk_ids: List[str] = Field(default_factory=list, description="Chunks not assigned to any cluster")
    clustering_params: Dict[str, Any] = Field(..., description="HDBSCAN parameters used")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Clustering performance metrics")
    
    def get_cluster_by_id(self, cluster_id: str) -> Optional[KnowledgeCluster]:
        """Get a cluster by its ID."""
        return next((c for c in self.clusters if c.id == cluster_id), None)
    
    def get_clusters_by_type(self, cluster_type: ClusterType) -> List[KnowledgeCluster]:
        """Get all clusters of a specific type."""
        return [c for c in self.clusters if c.metadata.cluster_type == cluster_type]
    
    def get_leaf_clusters(self) -> List[KnowledgeCluster]:
        """Get all leaf clusters (those containing actual chunks)."""
        return [c for c in self.clusters if c.is_leaf_cluster()]


class ConflictDetectionCandidate(BaseModel):
    """A candidate pair for conflict detection with clustering context."""
    chunk1_id: str = Field(..., description="First chunk ID")
    chunk2_id: str = Field(..., description="Second chunk ID")
    cluster1_id: Optional[str] = Field(None, description="Cluster of first chunk")
    cluster2_id: Optional[str] = Field(None, description="Cluster of second chunk")
    similarity_threshold: float = Field(..., description="Required similarity threshold")
    priority_score: float = Field(..., description="Priority for conflict detection (higher = more important)")
    relationship_context: str = Field(..., description="Context of relationship (same-cluster, cross-cluster, etc.)")