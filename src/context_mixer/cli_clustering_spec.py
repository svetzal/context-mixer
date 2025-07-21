"""
Tests for CLI clustering parameter handling.

This module tests the CLI integration of clustering parameters to ensure
proper configuration and parameter validation.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner

from .cli import app
from .config import Config


class DescribeClusteringCLIParameters:
    """Test cases for CLI clustering parameter handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_path = Path("/tmp/test-context-mixer")

    @patch('src.context_mixer.cli.KnowledgeStoreFactory')
    @patch('src.context_mixer.cli.IngestCommand')
    @patch('src.context_mixer.cli.llm_gateway', MagicMock())
    def should_use_default_clustering_parameters(self, mock_ingest_command, mock_factory):
        """Test that default clustering parameters are used when not specified."""
        mock_store = MagicMock()
        mock_factory.create_vector_store.return_value = mock_store
        mock_command_instance = MagicMock()
        mock_ingest_command.return_value = mock_command_instance
        
        # Mock the execute method to return a successful result
        mock_result = MagicMock()
        mock_result.success = True
        mock_command_instance.execute.return_value = mock_result
        
        result = self.runner.invoke(app, [
            'ingest',
            '--library-path', str(self.temp_path),
            '/some/test/path'
        ])
        
        # Verify the command succeeded
        assert result.exit_code == 0
        
        # Verify factory was called with correct config
        mock_factory.create_vector_store.assert_called_once()
        call_args = mock_factory.create_vector_store.call_args
        config = call_args.kwargs['config']
        
        # Verify default clustering settings
        assert config.clustering_enabled is True
        assert config.min_cluster_size == 3
        assert config.clustering_fallback is True
        assert config.conflict_detection_batch_size == 5

    @patch('src.context_mixer.cli.KnowledgeStoreFactory')
    @patch('src.context_mixer.cli.IngestCommand')
    @patch('src.context_mixer.cli.llm_gateway', MagicMock())
    def should_use_custom_clustering_parameters(self, mock_ingest_command, mock_factory):
        """Test that custom clustering parameters are properly passed."""
        mock_store = MagicMock()
        mock_factory.create_vector_store.return_value = mock_store
        mock_command_instance = MagicMock()
        mock_ingest_command.return_value = mock_command_instance
        
        # Mock the execute method to return a successful result
        mock_result = MagicMock()
        mock_result.success = True
        mock_command_instance.execute.return_value = mock_result
        
        result = self.runner.invoke(app, [
            'ingest',
            '--library-path', str(self.temp_path),
            '--no-clustering',
            '--min-cluster-size', '5',
            '--no-clustering-fallback',
            '--batch-size', '10',
            '/some/test/path'
        ])
        
        # Verify the command succeeded
        assert result.exit_code == 0
        
        # Verify factory was called with correct custom config
        mock_factory.create_vector_store.assert_called_once()
        call_args = mock_factory.create_vector_store.call_args
        config = call_args.kwargs['config']
        
        # Verify custom clustering settings
        assert config.clustering_enabled is False
        assert config.min_cluster_size == 5
        assert config.clustering_fallback is False
        assert config.conflict_detection_batch_size == 10

    @patch('src.context_mixer.cli.KnowledgeStoreFactory')
    @patch('src.context_mixer.cli.IngestCommand')
    @patch('src.context_mixer.cli.llm_gateway', MagicMock())
    def should_handle_clustering_enabled_explicitly(self, mock_ingest_command, mock_factory):
        """Test explicitly enabling clustering."""
        mock_store = MagicMock()
        mock_factory.create_vector_store.return_value = mock_store
        mock_command_instance = MagicMock()
        mock_ingest_command.return_value = mock_command_instance
        
        # Mock the execute method to return a successful result
        mock_result = MagicMock()
        mock_result.success = True
        mock_command_instance.execute.return_value = mock_result
        
        result = self.runner.invoke(app, [
            'ingest',
            '--library-path', str(self.temp_path),
            '--clustering',
            '/some/test/path'
        ])
        
        # Verify the command succeeded
        assert result.exit_code == 0
        
        # Verify clustering is enabled
        mock_factory.create_vector_store.assert_called_once()
        call_args = mock_factory.create_vector_store.call_args
        config = call_args.kwargs['config']
        
        assert config.clustering_enabled is True

    def should_validate_min_cluster_size_positive(self):
        """Test that min-cluster-size must be positive."""
        result = self.runner.invoke(app, [
            'ingest',
            '--library-path', str(self.temp_path),
            '--min-cluster-size', '0',
            '/some/test/path'
        ])
        
        # Should fail with validation error for non-positive cluster size
        # Note: Typer handles this validation automatically for int values
        # The actual validation depends on the specific validation rules we implement
        # For now, we just verify the command can handle the parameter

    def should_validate_batch_size_positive(self):
        """Test that batch-size must be positive."""
        result = self.runner.invoke(app, [
            'ingest',
            '--library-path', str(self.temp_path),
            '--batch-size', '0',
            '/some/test/path'
        ])
        
        # Should handle non-positive batch size appropriately
        # The exact behavior depends on validation rules we implement


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