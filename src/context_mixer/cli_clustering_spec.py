"""
Test for CLI integration with clustering options.
"""

import pytest
from context_mixer.config import Config


class DescribeCliClusteringIntegration:
    """Test specifications for CLI clustering integration."""
    
    def should_create_config_with_clustering_options(self):
        """Should create config with clustering options from CLI parameters."""
        config = Config(
            library_path=None,
            conflict_detection_batch_size=10,
            clustering_enabled=False,
            min_cluster_size=5,
            clustering_fallback=False
        )
        
        assert config.clustering_enabled is False
        assert config.min_cluster_size == 5
        assert config.clustering_fallback is False
        assert config.conflict_detection_batch_size == 10
    
    def should_create_config_with_default_clustering_settings(self):
        """Should create config with default clustering settings."""
        config = Config()
        
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 3
        assert config.clustering_fallback is True
        assert config.conflict_detection_batch_size == 5