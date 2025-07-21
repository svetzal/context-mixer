"""
HDBSCAN-based clustering service for knowledge optimization.

This service implements hierarchical domain-aware clustering to optimize conflict detection
by reducing expensive pairwise comparisons through intelligent grouping of related chunks.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:
    hdbscan = None

from .clustering import (
    KnowledgeCluster, ClusteringResult, ClusterType, ClusterMetadata,
    ConflictDetectionCandidate
)
from .knowledge import KnowledgeChunk
from ..gateways.llm import LLMGateway

logger = logging.getLogger(__name__)


class ClusteringService:
    """
    Service for HDBSCAN-based knowledge clustering and conflict detection optimization.

    Implements the hierarchical clustering strategy from PLAN.md to reduce conflict detection
    from O(n²) to O(k*log(k)) where k << n².
    """

    def __init__(self, llm_gateway: Optional[LLMGateway] = None):
        """
        Initialize the clustering service.

        Args:
            llm_gateway: Optional LLM gateway for generating cluster summaries
        """
        if hdbscan is None:
            raise ImportError("hdbscan package is required. Install with: pip install hdbscan")

        self.llm_gateway = llm_gateway
        self.scaler = StandardScaler()

        # Default HDBSCAN parameters optimized for knowledge clustering
        self.default_params = {
            'min_cluster_size': 3,  # Minimum chunks per cluster
            'min_samples': 2,  # Minimum samples for core points
            'cluster_selection_epsilon': 0.1,  # Distance threshold
            'metric': 'euclidean',  # Distance metric
            'cluster_selection_method': 'eom'  # Excess of Mass method
        }

    async def cluster_knowledge_chunks(
            self,
            chunks: List[KnowledgeChunk],
            clustering_params: Optional[Dict] = None
    ) -> ClusteringResult:
        """
        Perform hierarchical HDBSCAN clustering on knowledge chunks.

        Args:
            chunks: List of knowledge chunks to cluster
            clustering_params: Optional HDBSCAN parameters override

        Returns:
            ClusteringResult with hierarchical clusters
        """
        if not chunks:
            return ClusteringResult(
                clusters=[],
                noise_chunk_ids=[],
                clustering_params={},
                performance_metrics={}
            )

        logger.info(f"Starting HDBSCAN clustering for {len(chunks)} knowledge chunks")

        # Extract embeddings and prepare data
        embeddings, chunk_id_map = self._prepare_embeddings(chunks)
        if embeddings.size == 0:
            logger.warning("No embeddings available for clustering")
            return self._create_fallback_clusters(chunks)

        # Apply HDBSCAN clustering
        params = {**self.default_params, **(clustering_params or {})}
        clusterer = hdbscan.HDBSCAN(**params)

        # Normalize embeddings for better clustering
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        cluster_labels = clusterer.fit_predict(embeddings_scaled)

        # Create hierarchical clusters
        clusters = await self._create_hierarchical_clusters(
            chunks, cluster_labels, clusterer, chunk_id_map
        )

        # Identify noise (unclustered chunks)
        noise_chunk_ids = [
            chunks[i].id for i, label in enumerate(cluster_labels) if label == -1
        ]

        # Calculate performance metrics
        metrics = self._calculate_clustering_metrics(cluster_labels, clusterer)

        result = ClusteringResult(
            clusters=clusters,
            noise_chunk_ids=noise_chunk_ids,
            clustering_params=params,
            performance_metrics=metrics
        )

        logger.info(
            f"Clustering completed: {len(clusters)} clusters, {len(noise_chunk_ids)} noise chunks")
        return result

    def generate_conflict_detection_candidates(
            self,
            clustering_result: ClusteringResult,
            target_chunk: KnowledgeChunk
    ) -> List[ConflictDetectionCandidate]:
        """
        Generate prioritized conflict detection candidates using cluster relationships.

        This implements the dynamic conflict detection strategy from PLAN.md,
        significantly reducing the number of expensive LLM calls.

        Args:
            clustering_result: Result from clustering operation
            target_chunk: Chunk to find conflict candidates for

        Returns:
            Prioritized list of conflict detection candidates
        """
        candidates = []
        target_cluster = self._find_chunk_cluster(clustering_result, target_chunk.id)

        if target_cluster is None:
            # Handle noise chunks - check against cluster representatives
            candidates.extend(self._generate_noise_candidates(clustering_result, target_chunk))
        else:
            # Generate candidates based on cluster relationships
            candidates.extend(self._generate_cluster_based_candidates(
                clustering_result, target_chunk, target_cluster
            ))

        # Sort by priority score (higher = more important)
        candidates.sort(key=lambda c: c.priority_score, reverse=True)

        logger.debug(
            f"Generated {len(candidates)} conflict detection candidates for chunk "
            f"{target_chunk.id}")
        return candidates

    def _prepare_embeddings(self, chunks: List[KnowledgeChunk]) -> Tuple[
        np.ndarray, Dict[int, str]]:
        """Prepare embeddings matrix and create chunk ID mapping."""
        embeddings = []
        chunk_id_map = {}

        for i, chunk in enumerate(chunks):
            if chunk.embedding:
                embeddings.append(chunk.embedding)
                chunk_id_map[i] = chunk.id

        if not embeddings:
            return np.array([]), {}

        return np.array(embeddings), chunk_id_map

    async def _create_hierarchical_clusters(
            self,
            chunks: List[KnowledgeChunk],
            cluster_labels: np.ndarray,
            clusterer: 'hdbscan.HDBSCAN',
            chunk_id_map: Dict[int, str]
    ) -> List[KnowledgeCluster]:
        """Create hierarchical clusters from HDBSCAN results."""
        clusters = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        # Group chunks by cluster
        cluster_chunks = {}
        for i, label in enumerate(cluster_labels):
            if label != -1 and i in chunk_id_map:
                chunk_id = chunk_id_map[i]
                chunk = next(c for c in chunks if c.id == chunk_id)
                if label not in cluster_chunks:
                    cluster_chunks[label] = []
                cluster_chunks[label].append(chunk)

        # Create clusters with hierarchical organization
        for label, chunk_list in cluster_chunks.items():
            cluster = await self._create_cluster_from_chunks(
                label, chunk_list, clusterer
            )
            clusters.append(cluster)

        # Organize into hierarchy based on domains and contexts
        return self._organize_cluster_hierarchy(clusters)

    async def _create_cluster_from_chunks(
            self,
            hdbscan_label: int,
            chunks: List[KnowledgeChunk],
            clusterer: 'hdbscan.HDBSCAN'
    ) -> KnowledgeCluster:
        """Create a KnowledgeCluster from a group of chunks."""
        cluster_id = str(uuid.uuid4())

        # Analyze chunk metadata to determine cluster properties
        domains = set()
        authority_levels = set()
        scopes = set()

        for chunk in chunks:
            domains.update(chunk.metadata.domains)
            authority_levels.add(chunk.metadata.authority)
            scopes.update(chunk.metadata.scope)

        # Calculate centroid from chunk embeddings
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        centroid = None
        if embeddings:
            centroid = np.mean(embeddings, axis=0).tolist()

        # Determine cluster type based on content analysis
        cluster_type = self._determine_cluster_type(chunks)

        # Generate cluster summary if LLM is available
        summary = await self._generate_cluster_summary(chunks) if self.llm_gateway else None

        # Get stability score from HDBSCAN
        stability_score = None
        if hasattr(clusterer, 'cluster_persistence_'):
            stability_scores = clusterer.cluster_persistence_
            if hdbscan_label < len(stability_scores):
                stability_score = float(stability_scores[hdbscan_label])

        metadata = ClusterMetadata(
            cluster_type=cluster_type,
            domains=list(domains),
            authority_levels=authority_levels,
            scopes=list(set(scopes)),
            created_at=datetime.utcnow().isoformat(),
            chunk_count=len(chunks)
        )

        return KnowledgeCluster(
            id=cluster_id,
            chunk_ids=[chunk.id for chunk in chunks],
            centroid=centroid,
            summary=summary,
            metadata=metadata,
            hdbscan_cluster_id=hdbscan_label,
            stability_score=stability_score
        )

    def _determine_cluster_type(self, chunks: List[KnowledgeChunk]) -> ClusterType:
        """Determine the appropriate cluster type based on chunk analysis."""
        # Analyze domains to determine hierarchy level
        all_domains = set()
        for chunk in chunks:
            all_domains.update(chunk.metadata.domains)

        # Simple heuristic: if chunks span multiple high-level domains, it's a domain cluster
        high_level_domains = {'technical', 'business', 'operational', 'legal', 'design'}
        domain_overlap = len(all_domains.intersection(high_level_domains))

        if domain_overlap > 1:
            return ClusterType.KNOWLEDGE_DOMAIN
        elif len(all_domains) > 3:
            return ClusterType.CONTEXTUAL_SUBDOMAIN
        else:
            return ClusterType.SEMANTIC_CLUSTER

    async def _generate_cluster_summary(self, chunks: List[KnowledgeChunk]) -> Optional[str]:
        """Generate a concise summary of the knowledge in a cluster using LLM."""
        if not self.llm_gateway or not chunks:
            return None

        try:
            # Combine chunk contents for analysis
            combined_content = "\n\n".join([
                f"Chunk {i + 1}: {chunk.content[:200]}..."
                for i, chunk in enumerate(chunks[:5])  # Limit to first 5 chunks
            ])

            prompt = f"""Analyze the following knowledge chunks and provide a concise summary (
            2-3 sentences) of the common themes and knowledge domain they represent:

{combined_content}

Summary:"""

            response = await self.llm_gateway.generate_text(prompt, max_tokens=100)
            return response.strip()

        except Exception as e:
            logger.warning(f"Failed to generate cluster summary: {e}")
            return None

    def _organize_cluster_hierarchy(self, clusters: List[KnowledgeCluster]) -> List[
        KnowledgeCluster]:
        """Organize clusters into a hierarchy based on domain relationships."""
        # For now, return flat structure - hierarchy organization can be enhanced later
        # This would involve analyzing domain relationships and creating parent-child links
        return clusters

    def _find_chunk_cluster(self, clustering_result: ClusteringResult, chunk_id: str) -> Optional[
        KnowledgeCluster]:
        """Find the cluster containing a specific chunk."""
        for cluster in clustering_result.clusters:
            if chunk_id in cluster.chunk_ids:
                return cluster
        return None

    def _generate_cluster_based_candidates(
            self,
            clustering_result: ClusteringResult,
            target_chunk: KnowledgeChunk,
            target_cluster: KnowledgeCluster
    ) -> List[ConflictDetectionCandidate]:
        """Generate conflict detection candidates based on cluster relationships."""
        candidates = []

        # 1. Same cluster candidates (highest priority, lowest threshold)
        for chunk_id in target_cluster.chunk_ids:
            if chunk_id != target_chunk.id:
                candidates.append(ConflictDetectionCandidate(
                    chunk1_id=target_chunk.id,
                    chunk2_id=chunk_id,
                    cluster1_id=target_cluster.id,
                    cluster2_id=target_cluster.id,
                    similarity_threshold=0.7,  # Same cluster threshold
                    priority_score=1.0,
                    relationship_context="same-cluster"
                ))

        # 2. Cross-cluster candidates (lower priority, higher threshold)
        for cluster in clustering_result.clusters:
            if cluster.id != target_cluster.id:
                # Check if clusters are related (same domain, etc.)
                relationship_score = self._calculate_cluster_relationship_score(target_cluster,
                                                                                cluster)
                if relationship_score > 0.3:  # Only consider related clusters

                    # Select representative chunks from the other cluster
                    representative_chunks = cluster.chunk_ids[:3]  # Limit to 3 representatives

                    for chunk_id in representative_chunks:
                        threshold = 0.8 if relationship_score > 0.7 else 0.9
                        priority = relationship_score * 0.5  # Lower priority than same-cluster

                        candidates.append(ConflictDetectionCandidate(
                            chunk1_id=target_chunk.id,
                            chunk2_id=chunk_id,
                            cluster1_id=target_cluster.id,
                            cluster2_id=cluster.id,
                            similarity_threshold=threshold,
                            priority_score=priority,
                            relationship_context=f"cross-cluster-related-{relationship_score:.2f}"
                        ))

        return candidates

    def _generate_noise_candidates(
            self,
            clustering_result: ClusteringResult,
            target_chunk: KnowledgeChunk
    ) -> List[ConflictDetectionCandidate]:
        """Generate candidates for noise chunks (not assigned to any cluster)."""
        candidates = []

        # For noise chunks, check against cluster centroids/representatives
        for cluster in clustering_result.clusters:
            if cluster.chunk_ids:  # Only consider non-empty clusters
                # Use first chunk as representative (could be enhanced to use actual centroid)
                representative_chunk_id = cluster.chunk_ids[0]

                candidates.append(ConflictDetectionCandidate(
                    chunk1_id=target_chunk.id,
                    chunk2_id=representative_chunk_id,
                    cluster1_id=None,  # Noise chunk
                    cluster2_id=cluster.id,
                    similarity_threshold=0.9,  # High threshold for noise
                    priority_score=0.3,  # Lower priority
                    relationship_context="noise-to-cluster"
                ))

        return candidates

    def _calculate_cluster_relationship_score(
            self,
            cluster1: KnowledgeCluster,
            cluster2: KnowledgeCluster
    ) -> float:
        """Calculate relationship score between two clusters."""
        # Domain overlap
        domains1 = set(cluster1.metadata.domains)
        domains2 = set(cluster2.metadata.domains)
        domain_overlap = len(domains1.intersection(domains2)) / max(len(domains1.union(domains2)),
                                                                    1)

        # Authority level compatibility
        auth1 = cluster1.metadata.authority_levels
        auth2 = cluster2.metadata.authority_levels
        auth_overlap = len(auth1.intersection(auth2)) / max(len(auth1.union(auth2)), 1)

        # Scope overlap
        scopes1 = set(cluster1.metadata.scopes)
        scopes2 = set(cluster2.metadata.scopes)
        scope_overlap = len(scopes1.intersection(scopes2)) / max(len(scopes1.union(scopes2)), 1)

        # Weighted combination
        return (domain_overlap * 0.5) + (auth_overlap * 0.3) + (scope_overlap * 0.2)

    def _create_fallback_clusters(self, chunks: List[KnowledgeChunk]) -> ClusteringResult:
        """Create fallback clusters when embeddings are not available."""
        # Group by domain as fallback
        domain_groups = {}
        for chunk in chunks:
            for domain in chunk.metadata.domains:
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(chunk)

        clusters = []
        for domain, chunk_list in domain_groups.items():
            cluster_id = str(uuid.uuid4())
            metadata = ClusterMetadata(
                cluster_type=ClusterType.KNOWLEDGE_DOMAIN,
                domains=[domain],
                authority_levels={chunk.metadata.authority for chunk in chunk_list},
                scopes=[],
                created_at=datetime.utcnow().isoformat(),
                chunk_count=len(chunk_list)
            )

            cluster = KnowledgeCluster(
                id=cluster_id,
                chunk_ids=[chunk.id for chunk in chunk_list],
                metadata=metadata
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            noise_chunk_ids=[],
            clustering_params={'fallback': True},
            performance_metrics={'fallback_mode': True}
        )

    def _calculate_clustering_metrics(
            self,
            cluster_labels: np.ndarray,
            clusterer: 'hdbscan.HDBSCAN'
    ) -> Dict[str, float]:
        """Calculate clustering performance metrics."""
        metrics = {}

        # Basic metrics
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        metrics['num_clusters'] = len(unique_labels)
        metrics['num_noise_points'] = np.sum(cluster_labels == -1)
        metrics['noise_ratio'] = metrics['num_noise_points'] / len(cluster_labels)

        # HDBSCAN-specific metrics
        if hasattr(clusterer, 'cluster_persistence_'):
            metrics['avg_cluster_stability'] = float(np.mean(clusterer.cluster_persistence_))

        return metrics
