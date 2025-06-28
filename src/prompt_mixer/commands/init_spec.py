"""
Tests for the init command.
"""

import pytest
from unittest.mock import MagicMock

from prompt_mixer.commands.init import do_init
from prompt_mixer.gateways.git import GitGateway


@pytest.fixture
def mock_console():
    """Create a mock console."""
    console = MagicMock()
    return console


@pytest.fixture
def mock_path(tmp_path):
    """Create a temporary path for testing."""
    return tmp_path / "prompt-mixer"


@pytest.fixture
def mock_git_gateway():
    """Create a mock GitGateway."""
    git_gateway = MagicMock(spec=GitGateway)
    return git_gateway


class DescribeDoInit:
    """Tests for the do_init function."""

    def should_clone_repository_when_remote_is_provided(self, mock_git_gateway, mock_console, mock_path):
        """Should clone a repository when remote is provided."""
        # Arrange
        remote = "git@github.com:user/repo.git"
        mock_git_gateway.clone.return_value = (True, "Successfully cloned")

        # Act
        do_init(mock_console, mock_path, remote, None, None, mock_git_gateway)

        # Assert
        mock_git_gateway.clone.assert_called_once_with(remote, mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_initialize_new_repository_when_remote_not_provided(self, mock_git_gateway, mock_console, mock_path):
        """Should initialize a new repository when remote is not provided."""
        # Arrange
        mock_git_gateway.init.return_value = (True, "Successfully initialized")

        # Act
        do_init(mock_console, mock_path, None, None, None, mock_git_gateway)

        # Assert
        mock_git_gateway.init.assert_called_once_with(mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_create_directory_if_it_doesnt_exist(self, mock_git_gateway, mock_console, mock_path):
        """Should create the directory if it doesn't exist."""
        # Arrange
        mock_git_gateway.init.return_value = (True, "Successfully initialized")

        # Act
        do_init(mock_console, mock_path, None, None, None, mock_git_gateway)

        # Assert
        assert mock_path.exists()
        assert mock_path.is_dir()

    def should_handle_provider_and_model_parameters(self, mock_git_gateway, mock_console, mock_path):
        """Should handle provider and model parameters."""
        # Arrange
        provider = "ollama"
        model = "phi3"
        mock_git_gateway.init.return_value = (True, "Successfully initialized")

        # Act
        do_init(mock_console, mock_path, None, provider, model, mock_git_gateway)

        # Assert
        mock_git_gateway.init.assert_called_once_with(mock_path)
        # Check that the provider and model message was printed
        assert any("Provider and model configuration" in str(args) 
                  for args, _ in mock_console.print.call_args_list)
