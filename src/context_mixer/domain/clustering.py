"""
HDBSCAN-based clustering for knowledge optimization.

This module implements HDBSCAN clustering to optimize conflict detection
by grouping semantically similar chunks and reducing the number of
expensive pairwise comparisons needed during conflict analysis.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field

try:
    import hdbscan
except ImportError:
    hdbscan = None


@dataclass
class ClusterMetadata:
    """Metadata about a knowledge cluster."""
    cluster_id: int
    size: int
    persistence: float
    representative_chunk_id: Optional[str] = None
    chunk_ids: Optional[Set[str]] = None


class ClusteringConfig(BaseModel):
    """Configuration for HDBSCAN clustering."""
    min_cluster_size: int = Field(default=5, ge=2, description="Minimum cluster size")
    min_samples: int = Field(default=3, ge=1, description="Minimum samples for core points")
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0, description="Distance threshold for cluster merging")
    metric: str = Field(default='euclidean', description="Distance metric for clustering")
    alpha: float = Field(default=1.0, gt=0.0, description="Distance scaling parameter")
    prediction_data: bool = Field(default=True, description="Generate prediction data for new points")


class KnowledgeClusterer:
    """HDBSCAN-based clusterer for knowledge chunks."""
    
    def __init__(self, config: ClusteringConfig = None):
        """
        Initialize the knowledge clusterer.
        
        Args:
            config: Configuration for HDBSCAN clustering
        """
        if hdbscan is None:
            raise ImportError("HDBSCAN is required for clustering. Install with: pip install hdbscan")
        
        self.config = config or ClusteringConfig()
        self._clusterer: Optional[hdbscan.HDBSCAN] = None
        self._cluster_metadata: Dict[int, ClusterMetadata] = {}
        self._chunk_to_cluster: Dict[str, int] = {}
        self._fitted = False
        
        logging.info(f"Initialized KnowledgeClusterer with config: {self.config}")

    def fit(self, embeddings: np.ndarray, chunk_ids: List[str]) -> Dict[int, ClusterMetadata]:
        """
        Fit HDBSCAN clusterer on knowledge embeddings.
        
        Args:
            embeddings: Array of embeddings with shape (n_chunks, embedding_dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
            
        Returns:
            Dictionary mapping cluster IDs to cluster metadata
            
        Raises:
            ValueError: If embeddings and chunk_ids have different lengths
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Embeddings and chunk_ids must have the same length")
        
        if len(embeddings) < self.config.min_cluster_size:
            logging.warning(f"Not enough data points ({len(embeddings)}) for clustering "
                          f"(min_cluster_size={self.config.min_cluster_size}). "
                          f"All chunks will be treated as noise.")
            # Create a single cluster with all chunks as noise
            self._cluster_metadata = {-1: ClusterMetadata(
                cluster_id=-1,
                size=len(chunk_ids),
                persistence=0.0,
                chunk_ids=set(chunk_ids)
            )}
            self._chunk_to_cluster = {chunk_id: -1 for chunk_id in chunk_ids}
            self._fitted = True
            return self._cluster_metadata
        
        # Initialize HDBSCAN clusterer
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric=self.config.metric,
            alpha=self.config.alpha,
            prediction_data=self.config.prediction_data
        )
        
        # Fit the clusterer
        logging.info(f"Fitting HDBSCAN on {len(embeddings)} embeddings")
        cluster_labels = self._clusterer.fit_predict(embeddings)
        
        # Build cluster metadata
        self._build_cluster_metadata(cluster_labels, chunk_ids)
        
        self._fitted = True
        logging.info(f"Clustering complete. Found {len(self._cluster_metadata)} clusters "
                    f"(including noise cluster if present)")
        
        return self._cluster_metadata

    def predict_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Predict cluster for a new embedding.
        
        Args:
            embedding: Embedding vector for new chunk
            
        Returns:
            Tuple of (cluster_id, confidence) where cluster_id is -1 for noise
            
        Raises:
            RuntimeError: If clusterer hasn't been fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Clusterer must be fitted before prediction")
        
        if self._clusterer is None:
            # No clusters were formed, everything is noise
            return -1, 0.0
        
        try:
            # Use HDBSCAN's approximate prediction
            cluster_label, strength = hdbscan.approximate_predict(
                self._clusterer, [embedding]
            )
            return int(cluster_label[0]), float(strength[0])
        except Exception as e:
            logging.warning(f"Cluster prediction failed: {e}. Treating as noise.")
            return -1, 0.0

    def get_cluster_for_chunk(self, chunk_id: str) -> int:
        """
        Get cluster ID for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Cluster ID (-1 for noise)
        """
        return self._chunk_to_cluster.get(chunk_id, -1)

    def get_chunks_in_cluster(self, cluster_id: int) -> Set[str]:
        """
        Get all chunk IDs in a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Set of chunk IDs in the cluster
        """
        if cluster_id in self._cluster_metadata:
            return self._cluster_metadata[cluster_id].chunk_ids or set()
        return set()

    def get_nearby_clusters(self, cluster_id: int, max_distance: float = 0.5) -> List[int]:
        """
        Get clusters that are nearby to the given cluster.
        
        This is a simplified implementation that could be enhanced with
        actual distance calculations between cluster centroids.
        
        Args:
            cluster_id: ID of the source cluster
            max_distance: Maximum distance for considering clusters nearby
            
        Returns:
            List of nearby cluster IDs
        """
        if not self._fitted or self._clusterer is None:
            return []
        
        # For now, return all clusters as we don't have centroid distances computed
        # This could be optimized by computing cluster centroids and distances
        nearby = [cid for cid in self._cluster_metadata.keys() if cid != cluster_id and cid != -1]
        return nearby

    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        if not self._fitted:
            return {"fitted": False}
        
        cluster_sizes = [meta.size for meta in self._cluster_metadata.values() if meta.cluster_id != -1]
        noise_size = self._cluster_metadata.get(-1, ClusterMetadata(-1, 0, 0.0)).size
        
        stats = {
            "fitted": True,
            "total_clusters": len([c for c in self._cluster_metadata if c != -1]),
            "total_chunks": sum(meta.size for meta in self._cluster_metadata.values()),
            "noise_chunks": noise_size,
            "config": self.config.model_dump()
        }
        
        if cluster_sizes:
            stats.update({
                "avg_cluster_size": np.mean(cluster_sizes),
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "cluster_size_std": np.std(cluster_sizes)
            })
        
        if self._clusterer and hasattr(self._clusterer, 'relative_validity_'):
            stats["relative_validity"] = float(self._clusterer.relative_validity_)
        
        return stats

    def _build_cluster_metadata(self, cluster_labels: np.ndarray, chunk_ids: List[str]) -> None:
        """Build cluster metadata from HDBSCAN results."""
        self._cluster_metadata.clear()
        self._chunk_to_cluster.clear()
        
        # Group chunks by cluster
        cluster_chunks: Dict[int, List[str]] = {}
        for chunk_id, cluster_id in zip(chunk_ids, cluster_labels):
            cluster_id = int(cluster_id)
            if cluster_id not in cluster_chunks:
                cluster_chunks[cluster_id] = []
            cluster_chunks[cluster_id].append(chunk_id)
            self._chunk_to_cluster[chunk_id] = cluster_id
        
        # Build metadata for each cluster
        for cluster_id, chunk_list in cluster_chunks.items():
            # Get persistence from HDBSCAN if available
            persistence = 0.0
            if (self._clusterer and hasattr(self._clusterer, 'cluster_persistence_') 
                and cluster_id != -1 and cluster_id < len(self._clusterer.cluster_persistence_)):
                persistence = float(self._clusterer.cluster_persistence_[cluster_id])
            
            # Select representative chunk (first one for simplicity)
            representative = chunk_list[0] if chunk_list else None
            
            self._cluster_metadata[cluster_id] = ClusterMetadata(
                cluster_id=cluster_id,
                size=len(chunk_list),
                persistence=persistence,
                representative_chunk_id=representative,
                chunk_ids=set(chunk_list)
            )