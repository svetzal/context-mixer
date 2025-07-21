"""
CLI parameter handling for Context Mixer.

This module provides utilities for handling and validating CLI parameters
without dependencies on the main CLI module. This allows the parameter
logic to be tested independently.
"""

from pathlib import Path
from typing import Optional
from .config import Config


class CLIParameterHandler:
    """Handles conversion of CLI parameters to Config objects."""
    
    @staticmethod
    def create_config_from_ingest_params(
        library_path: Optional[Path] = None,
        clustering: bool = True,
        min_cluster_size: int = 3,
        clustering_fallback: bool = True,
        batch_size: int = 5
    ) -> Config:
        """
        Create a Config object from ingest command parameters.
        
        Args:
            library_path: Path to the context library
            clustering: Enable HDBSCAN clustering for optimized conflict detection
            min_cluster_size: Minimum number of chunks required to form a cluster
            clustering_fallback: Fall back to traditional O(n²) conflict detection if clustering fails
            batch_size: Number of conflict detections to process concurrently
            
        Returns:
            Config: Configured Config object
        """
        return Config(
            library_path=library_path,
            conflict_detection_batch_size=batch_size,
            clustering_enabled=clustering,
            min_cluster_size=min_cluster_size,
            clustering_fallback=clustering_fallback
        )
    
    @staticmethod
    def validate_cluster_size(min_cluster_size: int) -> bool:
        """
        Validate that minimum cluster size is positive.
        
        Args:
            min_cluster_size: The minimum cluster size to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return min_cluster_size > 0
    
    @staticmethod
    def validate_batch_size(batch_size: int) -> bool:
        """
        Validate that batch size is positive.
        
        Args:
            batch_size: The batch size to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return batch_size > 0