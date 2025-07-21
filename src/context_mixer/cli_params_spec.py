"""
Tests for CLI parameter handling.

This module tests the CLI parameter handling utilities to ensure
proper configuration and parameter validation.
"""

import pytest
from pathlib import Path

from .cli_params import CLIParameterHandler
from .config import Config


class DescribeClusteringCLIParameters:
    """Test cases for CLI clustering parameter handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_path = Path("/tmp/test-context-mixer")

    def should_use_default_clustering_parameters(self):
        """Test that default clustering parameters are used when not specified."""
        config = CLIParameterHandler.create_config_from_ingest_params()
        
        # Verify default clustering settings
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 3
        assert config.clustering_fallback is True
        assert config.conflict_detection_batch_size == 5

    def should_use_custom_clustering_parameters(self):
        """Test that custom clustering parameters are properly passed."""
        config = CLIParameterHandler.create_config_from_ingest_params(
            library_path=self.temp_path,
            clustering=False,
            min_cluster_size=5,
            clustering_fallback=False,
            batch_size=10
        )
        
        # Verify custom clustering settings
        assert config.clustering_enabled is False
        assert config.min_cluster_size == 5
        assert config.clustering_fallback is False
        assert config.conflict_detection_batch_size == 10

    def should_handle_clustering_enabled_explicitly(self):
        """Test explicitly enabling clustering."""
        config = CLIParameterHandler.create_config_from_ingest_params(
            library_path=self.temp_path,
            clustering=True
        )
        
        assert config.clustering_enabled is True

    def should_validate_min_cluster_size_positive(self):
        """Test that min-cluster-size must be positive."""
        assert CLIParameterHandler.validate_cluster_size(1) is True
        assert CLIParameterHandler.validate_cluster_size(3) is True
        assert CLIParameterHandler.validate_cluster_size(0) is False
        assert CLIParameterHandler.validate_cluster_size(-1) is False

    def should_validate_batch_size_positive(self):
        """Test that batch-size must be positive."""
        assert CLIParameterHandler.validate_batch_size(1) is True
        assert CLIParameterHandler.validate_batch_size(5) is True
        assert CLIParameterHandler.validate_batch_size(0) is False
        assert CLIParameterHandler.validate_batch_size(-1) is False


class DescribeConfigWithClusteringParameters:
    """Test cases for Config class with clustering parameters."""

    def should_create_config_with_default_clustering_values(self):
        """Test Config creation with default clustering values."""
        config = Config()
        
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 3
        assert config.clustering_fallback is True

    def should_create_config_with_custom_clustering_values(self):
        """Test Config creation with custom clustering values."""
        config = Config(
            clustering_enabled=False,
            min_cluster_size=5,
            clustering_fallback=False
        )
        
        assert config.clustering_enabled is False
        assert config.min_cluster_size == 5
        assert config.clustering_fallback is False

    def should_access_clustering_properties(self):
        """Test that clustering properties are accessible."""
        config = Config(
            clustering_enabled=True,
            min_cluster_size=7,
            clustering_fallback=True
        )
        
        # Test property access
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 7
        assert config.clustering_fallback is True

    def should_preserve_existing_config_functionality(self):
        """Test that existing config functionality still works."""
        custom_path = Path("/custom/path")
        config = Config(
            library_path=custom_path,
            conflict_detection_batch_size=10
        )
        
        # Test existing properties still work
        assert config.library_path == custom_path
        assert config.conflict_detection_batch_size == 10
        
        # Test new clustering properties have defaults
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 3
        assert config.clustering_fallback is True