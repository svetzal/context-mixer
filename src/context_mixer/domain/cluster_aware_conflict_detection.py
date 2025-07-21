"""
Cluster-aware conflict detection service.

This service integrates HDBSCAN clustering with conflict detection to optimize performance
by reducing expensive pairwise LLM comparisons from O(n²) to O(k*log(k)) where k << n².
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
import asyncio

from .clustering import ClusteringResult, ConflictDetectionCandidate
from .clustering_service import ClusteringService
from .knowledge import KnowledgeChunk
from ..commands.operations.merge import detect_conflicts_batch
from ..gateways.llm import LLMGateway


logger = logging.getLogger(__name__)


class ClusterAwareConflictDetector:
    """
    Optimized conflict detection using HDBSCAN clustering.

    This implementation reduces conflict detection complexity from O(n²) to O(k*log(k))
    by using clustering to identify semantically related chunks and only performing
    expensive LLM-based conflict detection on relevant pairs.
    """

    def __init__(self, clustering_service: ClusteringService, llm_gateway: LLMGateway):
        """
        Initialize the cluster-aware conflict detector.

        Args:
            clustering_service: Service for HDBSCAN clustering
            llm_gateway: Gateway for LLM-based conflict detection
        """
        self.clustering_service = clustering_service
        self.llm_gateway = llm_gateway
        self._cluster_cache: Dict[str, ClusteringResult] = {}

    async def detect_conflicts_optimized(
        self,
        target_chunk: KnowledgeChunk,
        existing_chunks: List[KnowledgeChunk],
        use_cache: bool = True,
        max_candidates: int = 50
    ) -> List[KnowledgeChunk]:
        """
        Detect conflicts using cluster-aware optimization.

        This method implements the dynamic conflict detection strategy from PLAN.md:
        1. Cluster all chunks using HDBSCAN
        2. Generate prioritized conflict detection candidates based on cluster relationships
        3. Perform LLM-based conflict detection only on high-priority candidates

        Args:
            target_chunk: Chunk to check for conflicts
            existing_chunks: List of existing chunks to check against
            use_cache: Whether to use cached clustering results
            max_candidates: Maximum number of candidates to check (performance limit)

        Returns:
            List of conflicting chunks
        """
        if not existing_chunks:
            return []

        logger.info(f"Starting cluster-aware conflict detection for chunk {target_chunk.id} "
                   f"against {len(existing_chunks)} existing chunks")

        # Step 1: Get or create clustering for existing chunks
        clustering_result = await self._get_or_create_clustering(
            existing_chunks, use_cache
        )

        # Step 2: Generate prioritized conflict detection candidates
        candidates = self.clustering_service.generate_conflict_detection_candidates(
            clustering_result, target_chunk
        )

        # Limit candidates for performance
        candidates = candidates[:max_candidates]

        logger.info(f"Generated {len(candidates)} conflict detection candidates "
                   f"(reduced from {len(existing_chunks)} potential comparisons)")

        # Step 3: Perform LLM-based conflict detection on candidates
        conflicts = await self._detect_conflicts_from_candidates(
            target_chunk, existing_chunks, candidates
        )

        logger.info(f"Found {len(conflicts)} conflicts using cluster-aware detection")
        return conflicts

    async def batch_detect_conflicts_optimized(
        self,
        target_chunks: List[KnowledgeChunk],
        existing_chunks: List[KnowledgeChunk],
        batch_size: int = 5,
        progress_callback=None
    ) -> Dict[str, List[KnowledgeChunk]]:
        """
        Batch conflict detection with clustering optimization.

        Args:
            target_chunks: Chunks to check for conflicts
            existing_chunks: Existing chunks to check against
            batch_size: Number of concurrent conflict detections
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping chunk IDs to their conflicts
        """
        if not target_chunks or not existing_chunks:
            return {}

        logger.info(f"Starting batch cluster-aware conflict detection for "
                   f"{len(target_chunks)} chunks against {len(existing_chunks)} existing chunks")

        # Create clustering once for all chunks
        clustering_result = await self._get_or_create_clustering(existing_chunks, use_cache=True)

        # Process chunks in batches
        results = {}
        for i in range(0, len(target_chunks), batch_size):
            batch = target_chunks[i:i + batch_size]

            # Process batch concurrently
            batch_tasks = [
                self._detect_conflicts_for_chunk_with_clustering(
                    chunk, existing_chunks, clustering_result
                )
                for chunk in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect results
            for j, result in enumerate(batch_results):
                chunk = batch[j]
                if isinstance(result, Exception):
                    logger.error(f"Error detecting conflicts for chunk {chunk.id}: {result}")
                    results[chunk.id] = []
                else:
                    results[chunk.id] = result

            # Progress callback
            if progress_callback:
                progress_callback(i + len(batch), len(target_chunks))

        total_conflicts = sum(len(conflicts) for conflicts in results.values())
        logger.info(f"Batch conflict detection completed: {total_conflicts} total conflicts found")

        return results

    async def _get_or_create_clustering(
        self,
        chunks: List[KnowledgeChunk],
        use_cache: bool = True
    ) -> ClusteringResult:
        """Get cached clustering result or create new one."""
        # Create cache key based on chunk IDs
        cache_key = self._create_cache_key(chunks)

        if use_cache and cache_key in self._cluster_cache:
            logger.debug(f"Using cached clustering result for {len(chunks)} chunks")
            return self._cluster_cache[cache_key]

        # Create new clustering
        logger.debug(f"Creating new clustering for {len(chunks)} chunks")
        clustering_result = await self.clustering_service.cluster_knowledge_chunks(chunks)

        # Cache result
        if use_cache:
            self._cluster_cache[cache_key] = clustering_result

            # Limit cache size to prevent memory issues
            if len(self._cluster_cache) > 10:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cluster_cache))
                del self._cluster_cache[oldest_key]

        return clustering_result

    async def _detect_conflicts_for_chunk_with_clustering(
        self,
        target_chunk: KnowledgeChunk,
        existing_chunks: List[KnowledgeChunk],
        clustering_result: ClusteringResult
    ) -> List[KnowledgeChunk]:
        """Detect conflicts for a single chunk using existing clustering."""
        candidates = self.clustering_service.generate_conflict_detection_candidates(
            clustering_result, target_chunk
        )

        # Limit candidates for performance
        candidates = candidates[:50]

        return await self._detect_conflicts_from_candidates(
            target_chunk, existing_chunks, candidates
        )

    async def _detect_conflicts_from_candidates(
        self,
        target_chunk: KnowledgeChunk,
        existing_chunks: List[KnowledgeChunk],
        candidates: List[ConflictDetectionCandidate]
    ) -> List[KnowledgeChunk]:
        """Perform LLM-based conflict detection on prioritized candidates."""
        if not candidates:
            return []

        # Create chunk lookup for efficient access
        chunk_lookup = {chunk.id: chunk for chunk in existing_chunks}

        # Prepare chunk pairs for batch processing
        chunk_pairs = []
        candidate_map = {}  # Map pair index to candidate info

        for i, candidate in enumerate(candidates):
            if candidate.chunk2_id in chunk_lookup:
                chunk2 = chunk_lookup[candidate.chunk2_id]
                chunk_pairs.append((target_chunk, chunk2))
                candidate_map[i] = candidate

        if not chunk_pairs:
            return []

        # Use existing batch conflict detection with optimized pairs
        logger.debug(f"Performing LLM-based conflict detection on {len(chunk_pairs)} candidate pairs")

        try:
            # Use the existing batch conflict detection function
            conflict_results = await detect_conflicts_batch(
                chunk_pairs, 
                self.llm_gateway, 
                batch_size=5
            )

            # Extract conflicting chunks
            conflicts = []
            for i, (chunk1, chunk2, conflict_list) in enumerate(conflict_results):
                if conflict_list and len(conflict_list.list) > 0:
                    conflicts.append(chunk2)

            return conflicts

        except Exception as e:
            logger.error(f"Error in batch conflict detection: {e}")
            # Fallback to individual checks
            return await self._fallback_individual_conflict_detection(
                target_chunk, chunk_pairs
            )

    async def _fallback_individual_conflict_detection(
        self,
        target_chunk: KnowledgeChunk,
        chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]]
    ) -> List[KnowledgeChunk]:
        """Fallback to individual conflict detection if batch processing fails."""
        conflicts = []

        for chunk1, chunk2 in chunk_pairs:
            try:
                from ..commands.operations.merge import detect_conflicts
                conflict_result = detect_conflicts(
                    chunk1.content, 
                    chunk2.content, 
                    self.llm_gateway
                )

                if conflict_result and len(conflict_result.list) > 0:
                    conflicts.append(chunk2)

            except Exception as e:
                logger.warning(f"Error detecting conflict between {chunk1.id} and {chunk2.id}: {e}")
                continue

        return conflicts

    def _create_cache_key(self, chunks: List[KnowledgeChunk]) -> str:
        """Create a cache key based on chunk IDs."""
        chunk_ids = sorted([chunk.id for chunk in chunks])
        return f"clustering_{hash(tuple(chunk_ids))}"

    def clear_cache(self):
        """Clear the clustering cache."""
        self._cluster_cache.clear()
        logger.debug("Clustering cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cluster_cache),
            'cache_keys': list(self._cluster_cache.keys())
        }

    async def analyze_clustering_performance(
        self,
        chunks: List[KnowledgeChunk]
    ) -> Dict[str, any]:
        """
        Analyze clustering performance and provide metrics.

        Args:
            chunks: Chunks to analyze

        Returns:
            Performance analysis results
        """
        if not chunks:
            return {'error': 'No chunks provided'}

        # Perform clustering
        clustering_result = await self.clustering_service.cluster_knowledge_chunks(chunks)

        # Calculate performance metrics
        total_chunks = len(chunks)
        num_clusters = len(clustering_result.clusters)
        noise_chunks = len(clustering_result.noise_chunk_ids)

        # Calculate potential conflict detection reduction
        # Traditional approach: O(n²) comparisons
        traditional_comparisons = total_chunks * (total_chunks - 1) // 2

        # Cluster-aware approach: sum of intra-cluster + limited inter-cluster comparisons
        cluster_comparisons = 0
        for cluster in clustering_result.clusters:
            cluster_size = len(cluster.chunk_ids)
            # Intra-cluster comparisons
            cluster_comparisons += cluster_size * (cluster_size - 1) // 2
            # Inter-cluster comparisons (limited to 3 representatives per cluster)
            inter_cluster_reps = min(3, cluster_size)
            cluster_comparisons += inter_cluster_reps * (num_clusters - 1) * 3

        # Add noise chunk comparisons (against cluster representatives)
        cluster_comparisons += noise_chunks * num_clusters * 3

        reduction_ratio = 1 - (cluster_comparisons / max(traditional_comparisons, 1))

        analysis = {
            'total_chunks': total_chunks,
            'num_clusters': num_clusters,
            'noise_chunks': noise_chunks,
            'clustering_metrics': clustering_result.performance_metrics,
            'conflict_detection_optimization': {
                'traditional_comparisons': traditional_comparisons,
                'cluster_aware_comparisons': cluster_comparisons,
                'reduction_ratio': reduction_ratio,
                'estimated_llm_call_reduction': f"{reduction_ratio * 100:.1f}%"
            },
            'cluster_distribution': [
                {
                    'cluster_id': cluster.id,
                    'chunk_count': len(cluster.chunk_ids),
                    'cluster_type': cluster.metadata.cluster_type,
                    'domains': cluster.metadata.domains
                }
                for cluster in clustering_result.clusters
            ]
        }

        return analysis
