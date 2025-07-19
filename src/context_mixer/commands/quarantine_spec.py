"""
Tests for the Quarantine Command classes.

Tests the command boundary as specified in the issue - the Command pattern
implementation provides a perfect testing point for command behavior.
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from context_mixer.commands.quarantine import (
    QuarantineListCommand,
    QuarantineReviewCommand,
    QuarantineResolveCommand,
    QuarantineStatsCommand,
    QuarantineClearCommand
)
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
def command_context(mock_console, mock_config):
    return CommandContext(
        console=mock_console,
        config=mock_config
    )


class DescribeQuarantineListCommand:
    """Test the QuarantineListCommand class at the command boundary."""

    @pytest.fixture
    def list_command(self):
        return QuarantineListCommand()

    @pytest.fixture
    def list_context(self, mock_console, mock_config):
        return CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={
                'reason_filter': 'conflict',
                'resolved_filter': False,
                'priority_filter': 1,
                'project_filter': 'test-project'
            }
        )

    async def should_execute_successfully_with_valid_context(self, list_command, list_context, mocker):
        """Test that QuarantineListCommand executes successfully with valid context."""
        mock_do_quarantine_list = mocker.patch('context_mixer.commands.quarantine.do_quarantine_list')
        
        result = await list_command.execute(list_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Quarantine list displayed successfully"
        assert result.error is None
        
        # Verify do_quarantine_list was called with correct parameters
        mock_do_quarantine_list.assert_called_once_with(
            console=list_context.console,
            config=list_context.config,
            reason_filter=list_context.parameters['reason_filter'],
            resolved_filter=list_context.parameters['resolved_filter'],
            priority_filter=list_context.parameters['priority_filter'],
            project_filter=list_context.parameters['project_filter']
        )

    async def should_handle_exceptions_gracefully(self, list_command, list_context, mocker):
        """Test that QuarantineListCommand handles exceptions and returns error result."""
        mock_do_quarantine_list = mocker.patch('context_mixer.commands.quarantine.do_quarantine_list')
        test_error = Exception("Test list error")
        mock_do_quarantine_list.side_effect = test_error
        
        result = await list_command.execute(list_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to list quarantined chunks: Test list error"
        assert result.error == test_error

    async def should_use_default_parameters_when_not_provided(self, list_command, command_context, mocker):
        """Test that QuarantineListCommand uses default parameters when not provided."""
        mock_do_quarantine_list = mocker.patch('context_mixer.commands.quarantine.do_quarantine_list')
        
        result = await list_command.execute(command_context)
        
        assert result.success is True
        
        # Verify do_quarantine_list was called with None values for optional parameters
        call_args = mock_do_quarantine_list.call_args
        assert call_args[1]['reason_filter'] is None
        assert call_args[1]['resolved_filter'] is None
        assert call_args[1]['priority_filter'] is None
        assert call_args[1]['project_filter'] is None


class DescribeQuarantineReviewCommand:
    """Test the QuarantineReviewCommand class at the command boundary."""

    @pytest.fixture
    def review_command(self):
        return QuarantineReviewCommand()

    @pytest.fixture
    def review_context(self, mock_console, mock_config):
        return CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'chunk_id': 'test-chunk-123'}
        )

    async def should_execute_successfully_with_valid_context(self, review_command, review_context, mocker):
        """Test that QuarantineReviewCommand executes successfully with valid context."""
        mock_do_quarantine_review = mocker.patch('context_mixer.commands.quarantine.do_quarantine_review')
        
        result = await review_command.execute(review_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Quarantine review completed successfully"
        assert result.data['chunk_id'] == 'test-chunk-123'
        assert result.error is None
        
        # Verify do_quarantine_review was called with correct parameters
        mock_do_quarantine_review.assert_called_once_with(
            console=review_context.console,
            config=review_context.config,
            chunk_id='test-chunk-123'
        )

    async def should_handle_exceptions_gracefully(self, review_command, review_context, mocker):
        """Test that QuarantineReviewCommand handles exceptions and returns error result."""
        mock_do_quarantine_review = mocker.patch('context_mixer.commands.quarantine.do_quarantine_review')
        test_error = Exception("Test review error")
        mock_do_quarantine_review.side_effect = test_error
        
        result = await review_command.execute(review_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to review quarantined chunk: Test review error"
        assert result.error == test_error

    async def should_require_chunk_id_parameter(self, review_command, command_context):
        """Test that QuarantineReviewCommand requires chunk_id parameter."""
        result = await review_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "chunk_id parameter is required"


class DescribeQuarantineResolveCommand:
    """Test the QuarantineResolveCommand class at the command boundary."""

    @pytest.fixture
    def resolve_command(self):
        return QuarantineResolveCommand()

    @pytest.fixture
    def resolve_context(self, mock_console, mock_config):
        return CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={
                'chunk_id': 'test-chunk-456',
                'action': 'accept',
                'reason': 'Reviewed and approved',
                'resolved_by': 'test-user',
                'notes': 'Additional notes'
            }
        )

    async def should_execute_successfully_with_valid_context(self, resolve_command, resolve_context, mocker):
        """Test that QuarantineResolveCommand executes successfully with valid context."""
        mock_do_quarantine_resolve = mocker.patch('context_mixer.commands.quarantine.do_quarantine_resolve')
        
        result = await resolve_command.execute(resolve_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Quarantine resolved successfully"
        assert result.data['chunk_id'] == 'test-chunk-456'
        assert result.data['action'] == 'accept'
        assert result.data['reason'] == 'Reviewed and approved'
        assert result.error is None
        
        # Verify do_quarantine_resolve was called with correct parameters
        mock_do_quarantine_resolve.assert_called_once_with(
            console=resolve_context.console,
            config=resolve_context.config,
            chunk_id='test-chunk-456',
            action='accept',
            reason='Reviewed and approved',
            resolved_by='test-user',
            notes='Additional notes'
        )

    async def should_handle_exceptions_gracefully(self, resolve_command, resolve_context, mocker):
        """Test that QuarantineResolveCommand handles exceptions and returns error result."""
        mock_do_quarantine_resolve = mocker.patch('context_mixer.commands.quarantine.do_quarantine_resolve')
        test_error = Exception("Test resolve error")
        mock_do_quarantine_resolve.side_effect = test_error
        
        result = await resolve_command.execute(resolve_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to resolve quarantined chunk: Test resolve error"
        assert result.error == test_error

    async def should_require_mandatory_parameters(self, resolve_command, mock_console, mock_config):
        """Test that QuarantineResolveCommand requires mandatory parameters."""
        # Test missing chunk_id
        context_missing_chunk_id = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'action': 'accept', 'reason': 'test'}
        )
        
        result = await resolve_command.execute(context_missing_chunk_id)
        assert result.success is False
        assert "chunk_id, action, and reason parameters are required" in result.message
        
        # Test missing action
        context_missing_action = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'chunk_id': 'test', 'reason': 'test'}
        )
        
        result = await resolve_command.execute(context_missing_action)
        assert result.success is False
        assert "chunk_id, action, and reason parameters are required" in result.message
        
        # Test missing reason
        context_missing_reason = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'chunk_id': 'test', 'action': 'accept'}
        )
        
        result = await resolve_command.execute(context_missing_reason)
        assert result.success is False
        assert "chunk_id, action, and reason parameters are required" in result.message


class DescribeQuarantineStatsCommand:
    """Test the QuarantineStatsCommand class at the command boundary."""

    @pytest.fixture
    def stats_command(self):
        return QuarantineStatsCommand()

    async def should_execute_successfully_with_valid_context(self, stats_command, command_context, mocker):
        """Test that QuarantineStatsCommand executes successfully with valid context."""
        mock_do_quarantine_stats = mocker.patch('context_mixer.commands.quarantine.do_quarantine_stats')
        
        result = await stats_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Quarantine statistics displayed successfully"
        assert result.error is None
        
        # Verify do_quarantine_stats was called with correct parameters
        mock_do_quarantine_stats.assert_called_once_with(
            console=command_context.console,
            config=command_context.config
        )

    async def should_handle_exceptions_gracefully(self, stats_command, command_context, mocker):
        """Test that QuarantineStatsCommand handles exceptions and returns error result."""
        mock_do_quarantine_stats = mocker.patch('context_mixer.commands.quarantine.do_quarantine_stats')
        test_error = Exception("Test stats error")
        mock_do_quarantine_stats.side_effect = test_error
        
        result = await stats_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to display quarantine statistics: Test stats error"
        assert result.error == test_error


class DescribeQuarantineClearCommand:
    """Test the QuarantineClearCommand class at the command boundary."""

    @pytest.fixture
    def clear_command(self):
        return QuarantineClearCommand()

    async def should_execute_successfully_with_valid_context(self, clear_command, command_context, mocker):
        """Test that QuarantineClearCommand executes successfully with valid context."""
        mock_do_quarantine_clear = mocker.patch('context_mixer.commands.quarantine.do_quarantine_clear')
        
        result = await clear_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Quarantine cleared successfully"
        assert result.error is None
        
        # Verify do_quarantine_clear was called with correct parameters
        mock_do_quarantine_clear.assert_called_once_with(
            console=command_context.console,
            config=command_context.config
        )

    async def should_handle_exceptions_gracefully(self, clear_command, command_context, mocker):
        """Test that QuarantineClearCommand handles exceptions and returns error result."""
        mock_do_quarantine_clear = mocker.patch('context_mixer.commands.quarantine.do_quarantine_clear')
        test_error = Exception("Test clear error")
        mock_do_quarantine_clear.side_effect = test_error
        
        result = await clear_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to clear quarantine: Test clear error"
        assert result.error == test_error