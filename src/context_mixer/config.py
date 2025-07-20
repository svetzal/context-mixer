"""
Configuration module for Context Mixer.

This module provides centralized configuration settings for the application.
"""

from pathlib import Path

DEFAULT_ROOT_CONTEXT_FILENAME = "context.md"

class Config:
    """Configuration settings for Context Mixer."""

    def __init__(self, 
                 library_path: Path = None, 
                 conflict_detection_batch_size: int = 5,
                 clustering_enabled: bool = True,
                 min_cluster_size: int = 3,
                 clustering_fallback: bool = True):
        """
        Initialize a Config object with library path, batch processing, and clustering settings.

        Args:
            library_path: Path to the context library. If None, uses the default path.
            conflict_detection_batch_size: Number of conflict detections to process concurrently (default: 5).
            clustering_enabled: Enable HDBSCAN clustering for conflict detection optimization (default: True).
            min_cluster_size: Minimum size for clusters in HDBSCAN (default: 3).
            clustering_fallback: Fall back to traditional O(n²) detection if clustering fails (default: True).
        """
        self._library_path = library_path or Path.home() / ".context-mixer"
        self._conflict_detection_batch_size = conflict_detection_batch_size
        self._clustering_enabled = clustering_enabled
        self._min_cluster_size = min_cluster_size
        self._clustering_fallback = clustering_fallback

    @property
    def library_path(self) -> Path:
        """
        Get the library path.

        Returns:
            Path: The library path.
        """
        return self._library_path

    @property
    def conflict_detection_batch_size(self) -> int:
        """
        Get the conflict detection batch size.

        Returns:
            int: The number of conflict detections to process concurrently.
        """
        return self._conflict_detection_batch_size

    @property
    def clustering_enabled(self) -> bool:
        """
        Get whether HDBSCAN clustering is enabled for conflict detection optimization.

        Returns:
            bool: True if clustering is enabled, False otherwise.
        """
        return self._clustering_enabled

    @property
    def min_cluster_size(self) -> int:
        """
        Get the minimum cluster size for HDBSCAN clustering.

        Returns:
            int: The minimum number of chunks required to form a cluster.
        """
        return self._min_cluster_size

    @property
    def clustering_fallback(self) -> bool:
        """
        Get whether to fall back to traditional conflict detection if clustering fails.

        Returns:
            bool: True if fallback is enabled, False otherwise.
        """
        return self._clustering_fallback
