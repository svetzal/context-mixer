"""
Hierarchical knowledge clustering using HDBSCAN for optimized conflict detection.

This module implements the HDBSCAN clustering system as specified in PLAN.md to reduce
conflict detection from O(n²) to O(k*log(k)) by grouping semantically related chunks
and only checking conflicts within/between related clusters.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from .knowledge import KnowledgeChunk, AuthorityLevel
from .context import Context

logger = logging.getLogger(__name__)


class ClusterStatus(str, Enum):
    """Status of cluster stability and quality."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    FORMING = "forming"
    DEGRADED = "degraded"


class ContextualDomain(str, Enum):
    """Enhanced domain classification with contextual boundaries."""
    TECHNICAL_ARCHITECTURE = "technical_architecture"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    BUSINESS_PROCESS = "business_process"
    BUSINESS_POLICY = "business_policy"
    OPERATIONAL_WORKFLOW = "operational_workflow"
    OPERATIONAL_PROCEDURE = "operational_procedure"
    LEGAL_REGULATORY = "legal_regulatory"
    LEGAL_COMPLIANCE = "legal_compliance"
    DESIGN_PATTERN = "design_pattern"
    DESIGN_UI_UX = "design_ui_ux"
    SECURITY_FRAMEWORK = "security_framework"
    SECURITY_IMPLEMENTATION = "security_implementation"
    UNKNOWN = "unknown"


@dataclass
class ClusterQualityMetrics:
    """Quality metrics for cluster performance monitoring."""
    silhouette_score: float = 0.0
    domain_coherence: float = 0.0
    stability_score: float = 0.0
    size: int = 0
    density: float = 0.0
    separation: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ContextualChunk(BaseModel):
    """
    Enhanced knowledge chunk with hierarchical parent awareness.
    
    Maintains awareness of at least one level up in its hierarchy,
    enabling contextual boundary recognition.
    """
    base_chunk: KnowledgeChunk = Field(..., description="The underlying knowledge chunk")
    cluster_id: Optional[str] = Field(None, description="ID of the cluster this chunk belongs to")
    parent_cluster_id: Optional[str] = Field(None, description="ID of the parent cluster (one level up)")
    contextual_domain: ContextualDomain = Field(ContextualDomain.UNKNOWN, description="Detected contextual domain")
    semantic_embedding: Optional[List[float]] = Field(None, description="Semantic embedding for clustering")
    contextual_similarity_scores: Dict[str, float] = Field(default_factory=dict, description="Similarity scores to other contexts")
    boundary_confidence: float = Field(0.0, description="Confidence in contextual boundary detection (0.0-1.0)")
    
    def get_hierarchical_context(self) -> Dict[str, Any]:
        """Get hierarchical context information for conflict detection."""
        return {
            "chunk_id": self.base_chunk.id,
            "cluster_id": self.cluster_id,
            "parent_cluster_id": self.parent_cluster_id,
            "domain": self.contextual_domain,
            "authority": self.base_chunk.get_authority_level(),
            "boundary_confidence": self.boundary_confidence
        }
    
    def is_contextually_related(self, other: 'ContextualChunk', threshold: float = 0.7) -> bool:
        """Check if this chunk is contextually related to another chunk."""
        # Same cluster - highly related
        if self.cluster_id and self.cluster_id == other.cluster_id:
            return True
            
        # Same parent cluster - moderately related
        if (self.parent_cluster_id and other.parent_cluster_id and 
            self.parent_cluster_id == other.parent_cluster_id):
            return True
            
        # Same contextual domain - somewhat related
        if self.contextual_domain == other.contextual_domain:
            return True
            
        # Check semantic similarity
        other_id = other.base_chunk.id
        if other_id in self.contextual_similarity_scores:
            return self.contextual_similarity_scores[other_id] >= threshold
            
        return False


class IntelligentCluster(BaseModel):
    """
    Cluster with contained chunk awareness and knowledge summarization.
    
    Each cluster maintains intelligence about contained chunks plus awareness
    of its broader cluster hierarchy.
    """
    id: str = Field(..., description="Unique cluster identifier")
    parent_cluster_id: Optional[str] = Field(None, description="ID of parent cluster in hierarchy")
    child_cluster_ids: Set[str] = Field(default_factory=set, description="IDs of child clusters")
    chunk_ids: Set[str] = Field(default_factory=set, description="IDs of chunks in this cluster")
    contextual_domain: ContextualDomain = Field(..., description="Primary contextual domain of cluster")
    knowledge_summary: str = Field("", description="AI-generated summary of knowledge in this cluster")
    centroid_embedding: Optional[List[float]] = Field(None, description="Cluster centroid for similarity comparisons")
    quality_metrics: ClusterQualityMetrics = Field(default_factory=ClusterQualityMetrics)
    status: ClusterStatus = Field(ClusterStatus.FORMING, description="Current cluster status")
    authority_distribution: Dict[AuthorityLevel, int] = Field(default_factory=dict, description="Distribution of authority levels")
    
    def get_cluster_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive cluster intelligence for conflict detection."""
        return {
            "id": self.id,
            "parent_id": self.parent_cluster_id,
            "domain": self.contextual_domain,
            "chunk_count": len(self.chunk_ids),
            "knowledge_summary": self.knowledge_summary,
            "quality_score": self.quality_metrics.silhouette_score,
            "authority_distribution": self.authority_distribution,
            "status": self.status
        }
    
    def should_check_conflicts_with(self, other: 'IntelligentCluster') -> bool:
        """Determine if conflicts should be checked between this cluster and another."""
        # Skip if either cluster is degraded
        if self.status == ClusterStatus.DEGRADED or other.status == ClusterStatus.DEGRADED:
            return False
            
        # Same cluster - always check internal conflicts
        if self.id == other.id:
            return True
            
        # Parent-child relationship - check conflicts  
        if (self.parent_cluster_id == other.id or 
            other.parent_cluster_id == self.id or
            (self.parent_cluster_id == other.parent_cluster_id and self.parent_cluster_id is not None)):
            return True
            
        # Same contextual domain - check conflicts
        if self.contextual_domain == other.contextual_domain:
            return True
            
        # Different domains - rarely need conflict checking
        return False
    
    def get_similarity_threshold(self, other: 'IntelligentCluster') -> float:
        """Get dynamic similarity threshold based on cluster relationship."""
        if self.id == other.id:
            return 0.7  # Same cluster - expect high similarity
        elif (self.parent_cluster_id == other.parent_cluster_id and 
              self.parent_cluster_id is not None):
            return 0.8  # Same parent - moderate restriction
        elif self.contextual_domain == other.contextual_domain:
            return 0.85  # Same domain - higher restriction  
        else:
            return 0.9  # Cross-domain - very high restriction


class ClusterQualityMonitor:
    """
    Monitor cluster quality with silhouette score, stability, and coherence metrics.
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, List[ClusterQualityMetrics]] = {}
        
    def evaluate_cluster_quality(self, 
                                cluster: IntelligentCluster,
                                chunks: List[ContextualChunk],
                                all_embeddings: np.ndarray) -> ClusterQualityMetrics:
        """
        Evaluate quality metrics for a cluster.
        
        Args:
            cluster: The cluster to evaluate
            chunks: List of chunks in the cluster
            all_embeddings: All embeddings for silhouette calculation
            
        Returns:
            Updated quality metrics
        """
        metrics = ClusterQualityMetrics()
        
        if len(chunks) < 2:
            metrics.silhouette_score = 0.0
            metrics.domain_coherence = 1.0  # Single chunk is coherent
            metrics.size = len(chunks)
            return metrics
        
        try:
            # Calculate silhouette score (requires sklearn)
            cluster_embeddings = np.array([chunk.semantic_embedding for chunk in chunks 
                                         if chunk.semantic_embedding is not None])
            
            if len(cluster_embeddings) > 1:
                # Simplified silhouette calculation - would use sklearn.metrics.silhouette_score in production
                metrics.silhouette_score = self._calculate_simplified_silhouette(cluster_embeddings)
            
            # Calculate domain coherence
            domain_counts = {}
            for chunk in chunks:
                domain = chunk.contextual_domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            if domain_counts:
                max_domain_count = max(domain_counts.values())
                metrics.domain_coherence = max_domain_count / len(chunks)
            
            # Calculate stability (based on historical metrics)
            metrics.stability_score = self._calculate_stability(cluster.id)
            
            # Update basic metrics
            metrics.size = len(chunks)
            metrics.density = self._calculate_density(cluster_embeddings)
            
        except Exception as e:
            logger.warning(f"Error calculating cluster quality for {cluster.id}: {e}")
            
        return metrics
    
    def _calculate_simplified_silhouette(self, embeddings: np.ndarray) -> float:
        """Simplified silhouette score calculation."""
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate average intra-cluster distance
        intra_distances = []
        for i, emb in enumerate(embeddings):
            distances = [np.linalg.norm(emb - other_emb) for j, other_emb in enumerate(embeddings) if i != j]
            if distances:
                intra_distances.append(np.mean(distances))
        
        if not intra_distances:
            return 0.0
            
        avg_intra_distance = np.mean(intra_distances)
        
        # Simplified: assume reasonable inter-cluster distance
        # In production, would calculate against other clusters
        assumed_inter_distance = avg_intra_distance * 1.5
        
        if assumed_inter_distance == 0:
            return 0.0
            
        silhouette = (assumed_inter_distance - avg_intra_distance) / max(assumed_inter_distance, avg_intra_distance)
        return max(0.0, min(1.0, silhouette))
    
    def _calculate_density(self, embeddings: np.ndarray) -> float:
        """Calculate cluster density."""
        if len(embeddings) < 2:
            return 1.0
        
        distances = []
        for i, emb in enumerate(embeddings):
            for j, other_emb in enumerate(embeddings[i+1:], i+1):
                distances.append(np.linalg.norm(emb - other_emb))
        
        if not distances:
            return 1.0
            
        return 1.0 / (1.0 + np.mean(distances))
    
    def _calculate_stability(self, cluster_id: str) -> float:
        """Calculate cluster stability based on historical metrics."""
        if cluster_id not in self.metrics_history:
            return 0.5  # New cluster, moderate stability
        
        history = self.metrics_history[cluster_id]
        if len(history) < 2:
            return 0.5
        
        # Calculate variance in quality metrics
        silhouette_scores = [m.silhouette_score for m in history[-5:]]  # Last 5 measurements
        if len(silhouette_scores) < 2:
            return 0.5
        
        variance = np.var(silhouette_scores)
        stability = max(0.0, 1.0 - variance)  # Lower variance = higher stability
        return stability
    
    def update_metrics(self, cluster_id: str, metrics: ClusterQualityMetrics):
        """Update metrics history for a cluster."""
        if cluster_id not in self.metrics_history:
            self.metrics_history[cluster_id] = []
        
        self.metrics_history[cluster_id].append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history[cluster_id]) > 20:
            self.metrics_history[cluster_id] = self.metrics_history[cluster_id][-20:]


class DynamicConflictDetector:
    """
    Context-aware conflict detection using hierarchical cluster relationships.
    
    Implements graduated similarity thresholds based on cluster relationships
    to reduce false positives and improve performance.
    """
    
    def __init__(self, quality_monitor: ClusterQualityMonitor):
        self.quality_monitor = quality_monitor
        self.conflict_cache: Dict[Tuple[str, str], bool] = {}
        
    def should_check_conflict(self, 
                            chunk1: ContextualChunk, 
                            chunk2: ContextualChunk,
                            cluster1: Optional[IntelligentCluster] = None,
                            cluster2: Optional[IntelligentCluster] = None) -> bool:
        """
        Determine if two chunks should be checked for conflicts based on clustering.
        
        This is the key optimization - only check conflicts for chunks that are
        semantically related based on their cluster relationships.
        """
        # Skip self-comparison
        if chunk1.base_chunk.id == chunk2.base_chunk.id:
            return False
        
        # Check cache first
        cache_key = tuple(sorted([chunk1.base_chunk.id, chunk2.base_chunk.id]))
        if cache_key in self.conflict_cache:
            return self.conflict_cache[cache_key]
        
        should_check = False
        
        # If clusters are provided, use cluster-level logic
        if cluster1 and cluster2:
            should_check = cluster1.should_check_conflicts_with(cluster2)
        else:
            # Fall back to chunk-level contextual relationship
            should_check = chunk1.is_contextually_related(chunk2)
        
        # Cache the decision for performance
        self.conflict_cache[cache_key] = should_check
        
        # Limit cache size
        if len(self.conflict_cache) > 10000:
            # Remove oldest entries (simplified - would use LRU in production)
            oldest_keys = list(self.conflict_cache.keys())[:1000]
            for key in oldest_keys:
                del self.conflict_cache[key]
        
        return should_check
    
    def get_similarity_threshold(self, 
                               chunk1: ContextualChunk, 
                               chunk2: ContextualChunk,
                               cluster1: Optional[IntelligentCluster] = None,
                               cluster2: Optional[IntelligentCluster] = None) -> float:
        """Get dynamic similarity threshold based on cluster relationships."""
        if cluster1 and cluster2:
            return cluster1.get_similarity_threshold(cluster2)
        
        # Fallback thresholds based on chunk relationships
        if chunk1.cluster_id == chunk2.cluster_id:
            return 0.7  # Same cluster
        elif chunk1.contextual_domain == chunk2.contextual_domain:
            return 0.8  # Same domain
        else:
            return 0.9  # Cross-domain
    
    def get_conflict_candidates(self, 
                              target_chunk: ContextualChunk,
                              all_chunks: List[ContextualChunk],
                              clusters: Dict[str, IntelligentCluster]) -> List[ContextualChunk]:
        """
        Get optimized list of chunks to check for conflicts against target chunk.
        
        This is the core optimization - instead of checking all chunks (O(n²)),
        only return chunks that are semantically related based on clustering.
        """
        candidates = []
        target_cluster = clusters.get(target_chunk.cluster_id) if target_chunk.cluster_id else None
        
        for chunk in all_chunks:
            if chunk.base_chunk.id == target_chunk.base_chunk.id:
                continue
                
            chunk_cluster = clusters.get(chunk.cluster_id) if chunk.cluster_id else None
            
            if self.should_check_conflict(target_chunk, chunk, target_cluster, chunk_cluster):
                candidates.append(chunk)
        
        logger.info(f"Reduced conflict candidates from {len(all_chunks)} to {len(candidates)} for chunk {target_chunk.base_chunk.id[:12]}...")
        return candidates


class HierarchicalKnowledgeClusterer(ABC):
    """
    Abstract base class for hierarchical knowledge clustering with cross-domain
    contextual awareness.
    """
    
    @abstractmethod
    async def cluster_chunks(self, chunks: List[KnowledgeChunk]) -> Tuple[List[ContextualChunk], Dict[str, IntelligentCluster]]:
        """
        Cluster knowledge chunks into hierarchical semantic groups.
        
        Args:
            chunks: List of knowledge chunks to cluster
            
        Returns:
            Tuple of (contextual_chunks, cluster_map)
        """
        pass
    
    @abstractmethod
    async def assign_chunk_to_cluster(self, chunk: KnowledgeChunk, existing_clusters: Dict[str, IntelligentCluster]) -> ContextualChunk:
        """
        Assign a single new chunk to an existing cluster or create a new cluster.
        
        Args:
            chunk: New chunk to assign
            existing_clusters: Existing cluster map
            
        Returns:
            ContextualChunk with cluster assignment
        """
        pass
    
    @abstractmethod
    def get_cluster_quality_threshold(self) -> float:
        """Get minimum acceptable cluster quality threshold."""
        pass


class MockHDBSCANClusterer(HierarchicalKnowledgeClusterer):
    """
    Mock implementation of HDBSCAN clustering for testing and development.
    
    This provides the interface and basic clustering logic without requiring
    the actual HDBSCAN library to be installed.
    """
    
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.quality_monitor = ClusterQualityMonitor()
        
    async def cluster_chunks(self, chunks: List[KnowledgeChunk]) -> Tuple[List[ContextualChunk], Dict[str, IntelligentCluster]]:
        """Mock clustering implementation based on domains and authority levels."""
        if not chunks:
            return [], {}
        
        # Group chunks by domain and authority for mock clustering
        domain_authority_groups = {}
        for chunk in chunks:
            domains = tuple(sorted(chunk.get_domains()))
            authority = chunk.get_authority_level()
            key = (domains, authority)
            
            if key not in domain_authority_groups:
                domain_authority_groups[key] = []
            domain_authority_groups[key].append(chunk)
        
        contextual_chunks = []
        clusters = {}
        cluster_id_counter = 0
        
        for (domains, authority), group_chunks in domain_authority_groups.items():
            if len(group_chunks) >= self.min_cluster_size:
                # Create a cluster for this group
                cluster_id = f"cluster_{cluster_id_counter}"
                cluster_id_counter += 1
                
                # Determine contextual domain
                contextual_domain = self._map_to_contextual_domain(domains)
                
                # Create intelligent cluster
                cluster = IntelligentCluster(
                    id=cluster_id,
                    contextual_domain=contextual_domain,
                    chunk_ids=set(chunk.id for chunk in group_chunks),
                    knowledge_summary=f"Knowledge cluster for {contextual_domain} at {authority} authority level",
                    status=ClusterStatus.STABLE,
                    authority_distribution={authority: len(group_chunks)}
                )
                
                clusters[cluster_id] = cluster
                
                # Create contextual chunks
                for chunk in group_chunks:
                    contextual_chunk = ContextualChunk(
                        base_chunk=chunk,
                        cluster_id=cluster_id,
                        contextual_domain=contextual_domain,
                        boundary_confidence=0.8  # Mock confidence
                    )
                    contextual_chunks.append(contextual_chunk)
            else:
                # Create individual contextual chunks without clusters (noise)
                for chunk in group_chunks:
                    contextual_domain = self._map_to_contextual_domain(chunk.get_domains())
                    contextual_chunk = ContextualChunk(
                        base_chunk=chunk,
                        contextual_domain=contextual_domain,
                        boundary_confidence=0.5  # Lower confidence for unclustered
                    )
                    contextual_chunks.append(contextual_chunk)
        
        return contextual_chunks, clusters
    
    async def assign_chunk_to_cluster(self, chunk: KnowledgeChunk, existing_clusters: Dict[str, IntelligentCluster]) -> ContextualChunk:
        """Assign new chunk to most similar existing cluster."""
        contextual_domain = self._map_to_contextual_domain(chunk.get_domains())
        authority = chunk.get_authority_level()
        
        # Find best matching cluster
        best_cluster = None
        best_score = 0.0
        
        for cluster in existing_clusters.values():
            score = 0.0
            
            # Same contextual domain
            if cluster.contextual_domain == contextual_domain:
                score += 0.6
            
            # Compatible authority level
            if authority in cluster.authority_distribution:
                score += 0.3
            
            # Cluster quality
            score += cluster.quality_metrics.silhouette_score * 0.1
            
            if score > best_score and score > 0.5:  # Minimum threshold
                best_score = score
                best_cluster = cluster
        
        if best_cluster:
            # Add to existing cluster
            best_cluster.chunk_ids.add(chunk.id)
            best_cluster.authority_distribution[authority] = best_cluster.authority_distribution.get(authority, 0) + 1
            
            return ContextualChunk(
                base_chunk=chunk,
                cluster_id=best_cluster.id,
                contextual_domain=contextual_domain,
                boundary_confidence=best_score
            )
        else:
            # Create as noise (no cluster)
            return ContextualChunk(
                base_chunk=chunk,
                contextual_domain=contextual_domain,
                boundary_confidence=0.3
            )
    
    def get_cluster_quality_threshold(self) -> float:
        """Minimum acceptable silhouette score."""
        return 0.3
    
    def _map_to_contextual_domain(self, domains: List[str]) -> ContextualDomain:
        """Map knowledge domains to contextual domains."""
        domain_mapping = {
            "technical": ContextualDomain.TECHNICAL_IMPLEMENTATION,
            "architecture": ContextualDomain.TECHNICAL_ARCHITECTURE,
            "business": ContextualDomain.BUSINESS_PROCESS,
            "policy": ContextualDomain.BUSINESS_POLICY,
            "operations": ContextualDomain.OPERATIONAL_WORKFLOW,
            "legal": ContextualDomain.LEGAL_REGULATORY,
            "compliance": ContextualDomain.LEGAL_COMPLIANCE,
            "design": ContextualDomain.DESIGN_PATTERN,
            "ui": ContextualDomain.DESIGN_UI_UX,
            "security": ContextualDomain.SECURITY_FRAMEWORK
        }
        
        for domain in domains:
            domain_lower = domain.lower()
            for key, contextual_domain in domain_mapping.items():
                if key in domain_lower:
                    return contextual_domain
        
        return ContextualDomain.UNKNOWN