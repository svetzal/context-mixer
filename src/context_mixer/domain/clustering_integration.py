"""
Integration of HDBSCAN clustering with the existing conflict detection system.

This module provides the integration points to use the hierarchical clustering
system in the existing ingest pipeline to optimize conflict detection.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .clustering import (
    KnowledgeCluster, ClusteringResult, ClusterType, ClusterMetadata,
    ConflictDetectionCandidate
)
from .knowledge import KnowledgeChunk
from .conflict import ConflictList
from ..commands.operations.merge import detect_conflicts_batch, detect_conflicts
from ..gateways.llm import LLMGateway

logger = logging.getLogger(__name__)


@dataclass
class ClusteringStatistics:
    """Statistics for clustering optimization performance."""
    total_chunks_clustered: int = 0
    clusters_created: int = 0
    traditional_comparisons_avoided: int = 0
    clustering_time: float = 0.0
    conflict_detection_time: float = 0.0
    optimization_percentage: float = 0.0
    fallback_used: bool = False


class ClusteringConfig:
    """Configuration for the clustering system."""
    
    def __init__(self,
                 enabled: bool = True,
                 min_cluster_size: int = 3,
                 min_samples: int = 1,
                 quality_threshold: float = 0.3,
                 batch_size: int = 5,
                 fallback_to_traditional: bool = True):
        self.enabled = enabled
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.fallback_to_traditional = fallback_to_traditional


class ClusterOptimizedConflictDetector:
    """
    Optimized conflict detector using hierarchical clustering.
    
    This class provides the main interface for using clustering to optimize
    conflict detection in the ingestion pipeline.
    """
    
    def __init__(self, 
                 clustering_service,  # Will be ClusteringService when available
                 llm_gateway: LLMGateway,
                 config: Optional[ClusteringConfig] = None):
        """
        Initialize the cluster-optimized conflict detector.
        
        Args:
            clustering_service: Service for HDBSCAN clustering
            llm_gateway: Gateway for LLM-based conflict detection
            config: Optional clustering configuration
        """
        self.clustering_service = clustering_service
        self.llm_gateway = llm_gateway
        self.config = config or ClusteringConfig()
        self._cluster_cache: Dict[str, ClusteringResult] = {}
        self._statistics = ClusteringStatistics()
    
    async def detect_conflicts_optimized(self,
                                       target_chunk: KnowledgeChunk,
                                       existing_chunks: List[KnowledgeChunk],
                                       use_cache: bool = True,
                                       max_candidates: int = 50) -> Tuple[List[KnowledgeChunk], ClusteringStatistics]:
        """
        Detect conflicts using clustering optimization.
        
        This method implements the clustering optimization strategy to reduce
        conflict detection from O(n²) to O(k*log(k)).
        
        Args:
            target_chunk: Chunk to check for conflicts
            existing_chunks: List of existing chunks to check against
            use_cache: Whether to use cluster caching
            max_candidates: Maximum number of conflict candidates to check
            
        Returns:
            Tuple of (conflicting chunks, clustering statistics)
        """
        start_time = time.time()
        self._statistics = ClusteringStatistics()
        
        if not self.config.enabled or not existing_chunks:
            # Fall back to traditional detection
            conflicts = await self._fallback_detection(target_chunk, existing_chunks)
            self._statistics.fallback_used = True
            return conflicts, self._statistics
        
        try:
            # Step 1: Cluster all chunks (target + existing)
            clustering_start = time.time()
            all_chunks = [target_chunk] + existing_chunks
            clustering_result = await self._cluster_chunks(all_chunks, use_cache)
            self._statistics.clustering_time = time.time() - clustering_start
            
            if not clustering_result or not clustering_result.clusters:
                # Clustering failed, fall back to traditional detection
                logger.warning("Clustering failed, falling back to traditional conflict detection")
                conflicts = await self._fallback_detection(target_chunk, existing_chunks)
                self._statistics.fallback_used = True
                return conflicts, self._statistics
            
            # Step 2: Find target chunk's cluster
            target_cluster = self._find_chunk_cluster(target_chunk, clustering_result)
            
            # Step 3: Select conflict candidates based on clustering
            candidates = self._select_conflict_candidates(
                target_chunk, target_cluster, clustering_result, existing_chunks, max_candidates
            )
            
            # Step 4: Perform conflict detection on candidates only
            detection_start = time.time()
            conflicts = await self._detect_conflicts_in_candidates(target_chunk, candidates)
            self._statistics.conflict_detection_time = time.time() - detection_start
            
            # Update statistics
            self._statistics.total_chunks_clustered = len(all_chunks)
            self._statistics.clusters_created = len(clustering_result.clusters)
            self._statistics.traditional_comparisons_avoided = len(existing_chunks) - len(candidates)
            total_possible_comparisons = len(existing_chunks)
            if total_possible_comparisons > 0:
                self._statistics.optimization_percentage = (
                    self._statistics.traditional_comparisons_avoided / total_possible_comparisons
                ) * 100.0
            
            logger.info(f"Clustering optimization: {self._statistics.optimization_percentage:.1f}% "
                       f"reduction in comparisons ({len(candidates)} vs {len(existing_chunks)})")
            
            return conflicts, self._statistics
            
        except Exception as e:
            logger.error(f"Clustering optimization failed: {e}")
            if self.config.fallback_to_traditional:
                conflicts = await self._fallback_detection(target_chunk, existing_chunks)
                self._statistics.fallback_used = True
                return conflicts, self._statistics
            else:
                raise
    
    async def _cluster_chunks(self, 
                            chunks: List[KnowledgeChunk],
                            use_cache: bool = True) -> Optional[ClusteringResult]:
        """Cluster the given chunks using the clustering service."""
        try:
            # Generate cache key based on chunk IDs
            chunk_ids = sorted([chunk.id for chunk in chunks])
            cache_key = "_".join(chunk_ids)
            
            if use_cache and cache_key in self._cluster_cache:
                return self._cluster_cache[cache_key]
            
            # Perform clustering
            result = await self.clustering_service.cluster_knowledge_chunks(
                chunks,
                clustering_params={
                    'min_cluster_size': self.config.min_cluster_size,
                    'min_samples': self.config.min_samples
                }
            )
            
            if use_cache and result:
                self._cluster_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return None
    
    def _find_chunk_cluster(self, 
                          chunk: KnowledgeChunk,
                          clustering_result: ClusteringResult) -> Optional[KnowledgeCluster]:
        """Find which cluster contains the given chunk."""
        for cluster in clustering_result.clusters:
            if chunk.id in cluster.chunk_ids:
                return cluster
        return None
    
    def _select_conflict_candidates(self,
                                  target_chunk: KnowledgeChunk,
                                  target_cluster: Optional[KnowledgeCluster],
                                  clustering_result: ClusteringResult,
                                  existing_chunks: List[KnowledgeChunk],
                                  max_candidates: int) -> List[KnowledgeChunk]:
        """
        Select conflict candidates based on clustering relationships.
        
        Strategy:
        1. All chunks in the same cluster as target (high probability of conflict)
        2. Representative chunks from related clusters
        3. High-authority chunks from all clusters
        """
        candidates = []
        
        if target_cluster:
            # Add all chunks from the same cluster
            for chunk_id in target_cluster.chunk_ids:
                if chunk_id != target_chunk.id:
                    # Find the actual chunk object from existing_chunks
                    chunk = next((c for c in existing_chunks if c.id == chunk_id), None)
                    if chunk:
                        candidates.append(chunk)
        
        # Add representatives from other clusters, prioritizing by authority and relevance
        for cluster in clustering_result.clusters:
            if target_cluster and cluster.id == target_cluster.id:
                continue  # Skip target cluster, already added
            
            # Add high-authority chunks from other clusters
            cluster_chunk_objects = [
                c for c in existing_chunks 
                if c.id in cluster.chunk_ids
            ]
            cluster_candidates = sorted(
                cluster_chunk_objects,
                key=lambda c: c.get_authority_level().value,
                reverse=True
            )
            
            # Add up to 2 representatives per cluster
            for chunk in cluster_candidates[:2]:
                if len(candidates) < max_candidates:
                    candidates.append(chunk)
        
        # Ensure we don't exceed max_candidates
        return candidates[:max_candidates]
    
    async def _detect_conflicts_in_candidates(self,
                                            target_chunk: KnowledgeChunk,
                                            candidates: List[KnowledgeChunk]) -> List[KnowledgeChunk]:
        """Perform actual conflict detection on the selected candidates."""
        if not candidates:
            return []
        
        # Create pairs for batch processing
        chunk_pairs = [(target_chunk, candidate) for candidate in candidates]
        
        # Use existing batch conflict detection for efficiency
        try:
            conflict_results = await detect_conflicts_batch(
                chunk_pairs, 
                self.llm_gateway,
                batch_size=self.config.batch_size
            )
            
            # Extract the conflicting chunks from the results
            conflicts = []
            for target, candidate, conflict_list in conflict_results:
                if conflict_list and len(conflict_list.list) > 0:
                    conflicts.append(candidate)
            return conflicts
            
        except Exception as e:
            logger.error(f"Batch conflict detection failed: {e}")
            # Fall back to individual conflict detection
            conflicts = []
            for candidate in candidates:
                try:
                    if await detect_conflicts(target_chunk, candidate, self.llm_gateway):
                        conflicts.append(candidate)
                except Exception as inner_e:
                    logger.error(f"Individual conflict detection failed: {inner_e}")
                    continue
            return conflicts
    
    async def _fallback_detection(self,
                                target_chunk: KnowledgeChunk,
                                existing_chunks: List[KnowledgeChunk]) -> List[KnowledgeChunk]:
        """Fall back to traditional O(n²) conflict detection."""
        logger.info("Using traditional conflict detection (no clustering optimization)")
        
        # Create pairs for batch processing
        chunk_pairs = [(target_chunk, candidate) for candidate in existing_chunks]
        
        try:
            conflict_results = await detect_conflicts_batch(
                chunk_pairs,
                self.llm_gateway,
                batch_size=self.config.batch_size
            )
            
            # Extract the conflicting chunks from the results
            conflicts = []
            for target, candidate, conflict_list in conflict_results:
                if conflict_list and len(conflict_list.list) > 0:
                    conflicts.append(candidate)
            return conflicts
            
        except Exception as e:
            logger.error(f"Fallback conflict detection failed: {e}")
            return []
    
    def get_statistics(self) -> ClusteringStatistics:
        """Get the latest clustering statistics."""
        return self._statistics
    
    def clear_cache(self):
        """Clear the cluster cache."""
        self._cluster_cache.clear()
        logger.info("Cluster cache cleared")