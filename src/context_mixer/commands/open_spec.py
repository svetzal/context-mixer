"""
Tests for the OpenCommand class.

Tests the command boundary as specified in the issue - the Command pattern
implementation provides a perfect testing point for command behavior.
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from context_mixer.commands.open import OpenCommand
from context_mixer.commands.base import CommandContext, CommandResult
from context_mixer.config import Config
from rich.console import Console


@pytest.fixture
def mock_console():
    return MagicMock(spec=Console)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.library_path = Path("/test/library")
    return config


@pytest.fixture
def open_command():
    return OpenCommand()


@pytest.fixture
def command_context(mock_console, mock_config):
    return CommandContext(
        console=mock_console,
        config=mock_config
    )


class DescribeOpenCommand:
    """Test the OpenCommand class at the command boundary."""

    async def should_execute_successfully_with_valid_context(self, open_command, command_context, mocker):
        """Test that OpenCommand executes successfully with valid context."""
        # Mock the do_open function to avoid actual subprocess calls
        mock_do_open = mocker.patch('context_mixer.commands.open.do_open')
        
        result = await open_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Library opened successfully"
        assert result.error is None
        
        # Verify do_open was called with correct parameters
        mock_do_open.assert_called_once_with(
            console=command_context.console,
            config=command_context.config
        )

    async def should_handle_exceptions_gracefully(self, open_command, command_context, mocker):
        """Test that OpenCommand handles exceptions and returns error result."""
        # Mock do_open to raise an exception
        mock_do_open = mocker.patch('context_mixer.commands.open.do_open')
        test_error = Exception("Test error")
        mock_do_open.side_effect = test_error
        
        result = await open_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to open library: Test error"
        assert result.error == test_error

    async def should_pass_context_dependencies_correctly(self, open_command, command_context, mocker):
        """Test that OpenCommand passes context dependencies correctly."""
        mock_do_open = mocker.patch('context_mixer.commands.open.do_open')
        
        await open_command.execute(command_context)
        
        # Verify the exact arguments passed to do_open
        call_args = mock_do_open.call_args
        assert call_args[1]['console'] == command_context.console
        assert call_args[1]['config'] == command_context.config

    async def should_return_standardized_command_result(self, open_command, command_context, mocker):
        """Test that OpenCommand returns a standardized CommandResult."""
        mocker.patch('context_mixer.commands.open.do_open')
        
        result = await open_command.execute(command_context)
        
        # Verify result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        assert hasattr(result, 'data')
        assert hasattr(result, 'error')
        assert isinstance(result.data, dict)