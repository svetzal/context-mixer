"""
Gateway for HDBSCAN clustering functionality.

This module provides a mockable interface to HDBSCAN clustering, allowing
for dependency injection and testing without requiring the actual HDBSCAN library.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClusteringParameters:
    """Parameters for HDBSCAN clustering."""
    min_cluster_size: int = 3
    min_samples: int = 1
    alpha: float = 1.0
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int = 0
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


@dataclass
class ClusteringResult:
    """Result of HDBSCAN clustering operation."""
    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    cluster_persistence: Optional[np.ndarray] = None
    condensed_tree: Optional[Any] = None
    single_linkage_tree: Optional[Any] = None
    cluster_hierarchy: Optional[Dict[int, List[int]]] = None
    noise_points: Optional[List[int]] = None


class HDBSCANGateway(ABC):
    """
    Abstract gateway interface for HDBSCAN clustering operations.
    
    This interface allows for dependency injection and mocking during tests.
    """
    
    @abstractmethod
    async def cluster(self, 
                     embeddings: np.ndarray,
                     parameters: ClusteringParameters) -> ClusteringResult:
        """
        Perform HDBSCAN clustering on the given embeddings.
        
        Args:
            embeddings: N-dimensional array of embeddings to cluster
            parameters: Clustering parameters
            
        Returns:
            ClusteringResult containing cluster labels and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if HDBSCAN is available for clustering.
        
        Returns:
            True if HDBSCAN can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def get_version(self) -> Optional[str]:
        """
        Get the version of the HDBSCAN library.
        
        Returns:
            Version string if available, None otherwise
        """
        pass


class RealHDBSCANGateway(HDBSCANGateway):
    """
    Real implementation of HDBSCAN gateway using the actual HDBSCAN library.
    """
    
    def __init__(self):
        """Initialize the real HDBSCAN gateway."""
        self._hdbscan = None
        self._available = False
        self._version = None
        
        try:
            import hdbscan
            self._hdbscan = hdbscan
            self._available = True
            self._version = getattr(hdbscan, '__version__', 'unknown')
            logger.info(f"HDBSCAN library loaded successfully (version: {self._version})")
        except ImportError as e:
            logger.warning(f"HDBSCAN library not available: {e}")
            self._available = False
    
    async def cluster(self, 
                     embeddings: np.ndarray,
                     parameters: ClusteringParameters) -> ClusteringResult:
        """Perform actual HDBSCAN clustering."""
        if not self._available:
            raise RuntimeError("HDBSCAN library is not available")
        
        if embeddings.size == 0:
            raise ValueError("Cannot cluster empty embeddings array")
        
        # Create HDBSCAN clusterer with parameters
        clusterer = self._hdbscan.HDBSCAN(
            min_cluster_size=parameters.min_cluster_size,
            min_samples=parameters.min_samples,
            alpha=parameters.alpha,
            cluster_selection_epsilon=parameters.cluster_selection_epsilon,
            max_cluster_size=parameters.max_cluster_size or None,
            metric=parameters.metric,
            cluster_selection_method=parameters.cluster_selection_method
        )
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Extract additional information
        probabilities = getattr(clusterer, 'probabilities_', None)
        cluster_persistence = getattr(clusterer, 'cluster_persistence_', None)
        condensed_tree = getattr(clusterer, 'condensed_tree_', None)
        single_linkage_tree = getattr(clusterer, 'single_linkage_tree_', None)
        
        # Find noise points (labeled as -1)
        noise_points = np.where(cluster_labels == -1)[0].tolist() if len(cluster_labels) > 0 else []
        
        # Build cluster hierarchy
        cluster_hierarchy = {}
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label != -1:  # Skip noise
                cluster_hierarchy[int(label)] = np.where(cluster_labels == label)[0].tolist()
        
        return ClusteringResult(
            labels=cluster_labels,
            probabilities=probabilities,
            cluster_persistence=cluster_persistence,
            condensed_tree=condensed_tree,
            single_linkage_tree=single_linkage_tree,
            cluster_hierarchy=cluster_hierarchy,
            noise_points=noise_points
        )
    
    def is_available(self) -> bool:
        """Check if HDBSCAN is available."""
        return self._available
    
    def get_version(self) -> Optional[str]:
        """Get HDBSCAN version."""
        return self._version


class MockHDBSCANGateway(HDBSCANGateway):
    """
    Mock implementation of HDBSCAN gateway for testing and development.
    
    This implementation provides deterministic clustering behavior based on
    domain knowledge and authority levels, allowing testing without the actual
    HDBSCAN dependency.
    """
    
    def __init__(self, deterministic: bool = True):
        """
        Initialize the mock HDBSCAN gateway.
        
        Args:
            deterministic: Whether to use deterministic clustering (for testing)
        """
        self.deterministic = deterministic
        self._version = "mock-1.0.0"
    
    async def cluster(self, 
                     embeddings: np.ndarray,
                     parameters: ClusteringParameters) -> ClusteringResult:
        """
        Perform mock clustering based on embedding patterns.
        
        This implementation creates clusters by analyzing embedding similarities
        and applying domain-aware grouping rules.
        """
        if embeddings.size == 0:
            return ClusteringResult(labels=np.array([]))
        
        n_samples = embeddings.shape[0]
        
        if n_samples < parameters.min_cluster_size:
            # All points are noise if we don't have enough for a cluster
            labels = np.full(n_samples, -1, dtype=int)
            return ClusteringResult(
                labels=labels,
                noise_points=list(range(n_samples))
            )
        
        # Simple clustering based on embedding similarity
        labels = self._perform_mock_clustering(embeddings, parameters)
        
        # Generate mock probabilities (higher for non-noise points)
        probabilities = np.where(labels == -1, 0.1, 0.9)
        
        # Build cluster hierarchy
        cluster_hierarchy = {}
        noise_points = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            indices = np.where(labels == label)[0].tolist()
            if label == -1:
                noise_points = indices
            else:
                cluster_hierarchy[int(label)] = indices
        
        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            cluster_hierarchy=cluster_hierarchy,
            noise_points=noise_points
        )
    
    def _perform_mock_clustering(self, 
                               embeddings: np.ndarray,
                               parameters: ClusteringParameters) -> np.ndarray:
        """
        Perform the actual mock clustering logic.
        
        This uses a simple distance-based clustering approach that mimics
        HDBSCAN behavior for testing purposes.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.cluster import DBSCAN
        
        # Use DBSCAN as a simple approximation of HDBSCAN behavior
        # This provides predictable clustering for testing
        dbscan = DBSCAN(eps=0.5, min_samples=parameters.min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        # Filter out clusters that are too small
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != -1 and count < parameters.min_cluster_size:
                labels[labels == label] = -1
        
        # Relabel clusters to be consecutive starting from 0
        unique_labels = np.unique(labels)
        relabel_map = {}
        next_label = 0
        
        for label in unique_labels:
            if label == -1:
                relabel_map[label] = -1
            else:
                relabel_map[label] = next_label
                next_label += 1
        
        # Apply relabeling
        relabeled = np.array([relabel_map[label] for label in labels])
        
        return relabeled
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True
    
    def get_version(self) -> Optional[str]:
        """Get mock version."""
        return self._version


def create_hdbscan_gateway(prefer_real: bool = True) -> HDBSCANGateway:
    """
    Factory function to create an appropriate HDBSCAN gateway.
    
    Args:
        prefer_real: Whether to prefer the real HDBSCAN implementation
        
    Returns:
        HDBSCANGateway instance (real if available, mock otherwise)
    """
    if prefer_real:
        real_gateway = RealHDBSCANGateway()
        if real_gateway.is_available():
            return real_gateway
        else:
            logger.info("HDBSCAN not available, using mock implementation")
            return MockHDBSCANGateway()
    else:
        return MockHDBSCANGateway()