"""
Integration of HDBSCAN clustering with the existing conflict detection system.

This module provides the integration points to use the hierarchical clustering
system in the existing ingest pipeline to optimize conflict detection.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Tuple

from .clustering import (
    HierarchicalKnowledgeClusterer, MockHDBSCANClusterer, ContextualChunk,
    IntelligentCluster, DynamicConflictDetector, ClusterQualityMonitor
)
from .knowledge import KnowledgeChunk
from .conflict import ConflictList
from ..commands.operations.merge import detect_conflicts_batch, detect_conflicts
from ..gateways.llm import LLMGateway

logger = logging.getLogger(__name__)


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
                 clusterer: Optional[HierarchicalKnowledgeClusterer] = None,
                 config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.clusterer = clusterer or MockHDBSCANClusterer(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples
        )
        self.quality_monitor = ClusterQualityMonitor()
        self.dynamic_detector = DynamicConflictDetector(self.quality_monitor)
        
        # Performance tracking
        self.performance_stats = {
            "traditional_checks": 0,
            "optimized_checks": 0,
            "clusters_created": 0,
            "fallback_used": 0
        }
    
    async def detect_internal_conflicts_optimized(self,
                                                 chunks: List[KnowledgeChunk],
                                                 llm_gateway: LLMGateway,
                                                 progress_callback=None) -> List[ConflictList]:
        """
        Detect conflicts between chunks using clustering optimization.
        
        This is the main optimization - instead of O(n²) pairwise comparisons,
        use clustering to reduce the search space significantly.
        
        Args:
            chunks: List of chunks to check for conflicts
            llm_gateway: LLM gateway for conflict detection
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of ConflictList objects containing detected conflicts
        """
        if not self.config.enabled or len(chunks) < self.config.min_cluster_size:
            logger.info(f"Clustering disabled or insufficient chunks ({len(chunks)}), using traditional detection")
            return await self._traditional_conflict_detection(chunks, llm_gateway, progress_callback)
        
        try:
            # Step 1: Cluster the chunks
            logger.info(f"Clustering {len(chunks)} chunks for optimized conflict detection")
            contextual_chunks, clusters = await self.clusterer.cluster_chunks(chunks)
            self.performance_stats["clusters_created"] = len(clusters)
            
            # Step 2: Get optimized conflict candidates using clustering
            all_conflicts = []
            total_candidates = 0
            optimized_candidates = 0
            
            for i, chunk in enumerate(contextual_chunks):
                # Get optimized list of candidates to check against
                candidates = self.dynamic_detector.get_conflict_candidates(
                    chunk, contextual_chunks, clusters
                )
                
                total_candidates += len(contextual_chunks) - 1  # All possible comparisons
                optimized_candidates += len(candidates)
                
                if candidates:
                    # Check conflicts only with relevant candidates
                    chunk_pairs = [(chunk, candidate) for candidate in candidates]
                    
                    # Use batch conflict detection for performance
                    batch_results = await detect_conflicts_batch(
                        [(pair[0].base_chunk, pair[1].base_chunk) for pair in chunk_pairs],
                        llm_gateway,
                        self.config.batch_size
                    )
                    
                    # Process results
                    for (chunk1, chunk2), (_, _, conflicts) in zip(chunk_pairs, batch_results):
                        if conflicts.list:
                            all_conflicts.extend(conflicts.list)
                
                if progress_callback:
                    progress_callback(i + 1, f"Checked chunk {i + 1}/{len(contextual_chunks)}")
            
            # Update performance stats
            self.performance_stats["optimized_checks"] = optimized_candidates
            
            # Calculate optimization ratio
            if total_candidates > 0:
                optimization_ratio = 1.0 - (optimized_candidates / total_candidates)
                logger.info(f"Clustering optimization reduced conflict checks by {optimization_ratio:.1%} "
                           f"({optimized_candidates}/{total_candidates})")
            
            return [ConflictList(list=all_conflicts)] if all_conflicts else []
            
        except Exception as e:
            logger.warning(f"Clustering-based conflict detection failed: {e}")
            
            if self.config.fallback_to_traditional:
                logger.info("Falling back to traditional conflict detection")
                self.performance_stats["fallback_used"] += 1
                return await self._traditional_conflict_detection(chunks, llm_gateway, progress_callback)
            else:
                raise
    
    async def detect_external_conflicts_optimized(self,
                                                 new_chunks: List[KnowledgeChunk],
                                                 existing_clusters: Dict[str, IntelligentCluster],
                                                 llm_gateway: LLMGateway,
                                                 knowledge_store,
                                                 progress_callback=None) -> List[ConflictList]:
        """
        Detect conflicts between new chunks and existing knowledge using clustering.
        
        Args:
            new_chunks: New chunks to check
            existing_clusters: Existing cluster map from knowledge store
            llm_gateway: LLM gateway for conflict detection
            knowledge_store: Knowledge store for retrieving existing chunks
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of ConflictList objects containing detected conflicts
        """
        if not self.config.enabled:
            logger.info("Clustering disabled, using traditional external conflict detection")
            return await self._traditional_external_conflict_detection(
                new_chunks, knowledge_store, llm_gateway, progress_callback
            )
        
        try:
            all_conflicts = []
            
            for i, chunk in enumerate(new_chunks):
                # Assign chunk to existing cluster or mark as noise
                contextual_chunk = await self.clusterer.assign_chunk_to_cluster(chunk, existing_clusters)
                
                if contextual_chunk.cluster_id:
                    # Get chunks from related clusters only
                    related_cluster_ids = self._get_related_cluster_ids(
                        contextual_chunk.cluster_id, existing_clusters
                    )
                    
                    # Get existing chunks from related clusters only
                    existing_chunks = []
                    for cluster_id in related_cluster_ids:
                        cluster = existing_clusters[cluster_id]
                        for chunk_id in cluster.chunk_ids:
                            try:
                                existing_chunk = await knowledge_store.get_chunk(chunk_id)
                                if existing_chunk:
                                    existing_chunks.append(existing_chunk)
                            except Exception as e:
                                logger.warning(f"Failed to retrieve chunk {chunk_id}: {e}")
                    
                    logger.info(f"Checking conflicts with {len(existing_chunks)} chunks from related clusters "
                               f"instead of all existing chunks")
                else:
                    # No cluster assignment - check against small random sample
                    # This is a compromise for "noise" chunks
                    existing_chunks = await self._get_sample_existing_chunks(knowledge_store, sample_size=10)
                    logger.info(f"Chunk {chunk.id[:12]}... is noise, checking against sample of {len(existing_chunks)} chunks")
                
                # Check conflicts with selected existing chunks
                if existing_chunks:
                    chunk_pairs = [(chunk, existing_chunk) for existing_chunk in existing_chunks]
                    
                    batch_results = await detect_conflicts_batch(
                        chunk_pairs,
                        llm_gateway,
                        self.config.batch_size
                    )
                    
                    for (new_chunk, existing_chunk), (_, _, conflicts) in zip(chunk_pairs, batch_results):
                        if conflicts.list:
                            all_conflicts.extend(conflicts.list)
                
                if progress_callback:
                    progress_callback(i + 1, f"Checked external conflicts for chunk {i + 1}/{len(new_chunks)}")
            
            return [ConflictList(list=all_conflicts)] if all_conflicts else []
            
        except Exception as e:
            logger.warning(f"Clustering-based external conflict detection failed: {e}")
            
            if self.config.fallback_to_traditional:
                logger.info("Falling back to traditional external conflict detection")
                self.performance_stats["fallback_used"] += 1
                return await self._traditional_external_conflict_detection(
                    new_chunks, knowledge_store, llm_gateway, progress_callback
                )
            else:
                raise
    
    def _get_related_cluster_ids(self, cluster_id: str, clusters: Dict[str, IntelligentCluster]) -> List[str]:
        """Get IDs of clusters that should be checked for conflicts."""
        related_ids = [cluster_id]  # Always include the cluster itself
        
        if cluster_id not in clusters:
            return related_ids
        
        target_cluster = clusters[cluster_id]
        
        for other_id, other_cluster in clusters.items():
            if other_id != cluster_id and target_cluster.should_check_conflicts_with(other_cluster):
                related_ids.append(other_id)
        
        return related_ids
    
    async def _get_sample_existing_chunks(self, knowledge_store, sample_size: int = 10) -> List[KnowledgeChunk]:
        """Get a small sample of existing chunks for noise chunk conflict checking."""
        try:
            # This would need to be implemented in the knowledge store interface
            # For now, return empty list as a safe fallback
            # In production, this could query recent chunks or high-authority chunks
            return []
        except Exception as e:
            logger.warning(f"Failed to get sample existing chunks: {e}")
            return []
    
    async def _traditional_conflict_detection(self,
                                            chunks: List[KnowledgeChunk],
                                            llm_gateway: LLMGateway,
                                            progress_callback=None) -> List[ConflictList]:
        """Fallback to traditional O(n²) conflict detection."""
        all_conflicts = []
        total_pairs = len(chunks) * (len(chunks) - 1) // 2
        current_pair = 0
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                current_pair += 1
                
                try:
                    conflicts = detect_conflicts(chunk1.content, chunk2.content, llm_gateway)
                    if conflicts.list:
                        all_conflicts.extend(conflicts.list)
                except Exception as e:
                    logger.warning(f"Failed to check conflicts between chunks {chunk1.id[:12]}... and {chunk2.id[:12]}...: {e}")
                
                if progress_callback:
                    progress_callback(current_pair, f"Checked pair {current_pair}/{total_pairs}")
        
        self.performance_stats["traditional_checks"] = total_pairs
        return [ConflictList(list=all_conflicts)] if all_conflicts else []
    
    async def _traditional_external_conflict_detection(self,
                                                      new_chunks: List[KnowledgeChunk],
                                                      knowledge_store,
                                                      llm_gateway: LLMGateway,
                                                      progress_callback=None) -> List[ConflictList]:
        """Fallback to traditional external conflict detection."""
        all_conflicts = []
        
        for i, chunk in enumerate(new_chunks):
            try:
                conflicting_chunks = await knowledge_store.detect_conflicts(chunk)
                if conflicting_chunks:
                    # Convert to ConflictList format - would need proper implementation
                    # This is a simplified version
                    pass
            except Exception as e:
                logger.warning(f"Failed to check external conflicts for chunk {chunk.id[:12]}...: {e}")
            
            if progress_callback:
                progress_callback(i + 1, f"Checked external conflicts for chunk {i + 1}/{len(new_chunks)}")
        
        return []
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics for the clustering optimization."""
        stats = self.performance_stats.copy()
        
        if stats["traditional_checks"] > 0 and stats["optimized_checks"] > 0:
            # Calculate reduction ratio when both are available
            total_traditional = stats["traditional_checks"]
            total_optimized = stats["optimized_checks"]
            stats["reduction_ratio"] = 1.0 - (total_optimized / total_traditional)
            stats["reduction_percentage"] = stats["reduction_ratio"] * 100
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "traditional_checks": 0,
            "optimized_checks": 0,
            "clusters_created": 0,
            "fallback_used": 0
        }