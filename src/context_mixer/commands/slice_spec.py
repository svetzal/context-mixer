"""
Tests for the SliceCommand class.

Tests the command boundary as specified in the issue - the Command pattern
implementation provides a perfect testing point for command behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from context_mixer.commands.base import CommandContext, CommandResult
from context_mixer.commands.slice import SliceCommand
from context_mixer.config import Config
from context_mixer.gateways.llm import LLMGateway


@pytest.fixture
def mock_console():
    return MagicMock(spec=Console)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.library_path = Path("/test/library")
    return config


@pytest.fixture
def mock_llm_gateway():
    return MagicMock(spec=LLMGateway)


@pytest.fixture
def slice_command():
    return SliceCommand()


@pytest.fixture
def command_context(mock_console, mock_config, mock_llm_gateway):
    return CommandContext(
        console=mock_console,
        config=mock_config,
        llm_gateway=mock_llm_gateway,
        parameters={
            'output_path': Path('/test/output'),
            'granularity': 'detailed',
            'domains': ['technical', 'business'],
            'project_ids': ['proj1', 'proj2'],
            'exclude_projects': ['proj3'],
            'authority_level': 'high'
        }
    )


class DescribeSliceCommand:
    """Test the SliceCommand class at the command boundary."""

    async def should_execute_successfully_with_valid_context(self, slice_command, command_context, mocker):
        """Test that SliceCommand executes successfully with valid context."""
        # Mock the do_slice function
        mock_do_slice = mocker.patch('context_mixer.commands.slice.do_slice')
        
        result = await slice_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Context sliced successfully"
        assert result.error is None
        
        # Verify do_slice was called with correct parameters
        mock_do_slice.assert_called_once_with(
            console=command_context.console,
            config=command_context.config,
            llm_gateway=command_context.llm_gateway,
            output_path=command_context.parameters['output_path'],
            granularity=command_context.parameters['granularity'],
            domains=command_context.parameters['domains'],
            project_ids=command_context.parameters['project_ids'],
            exclude_projects=command_context.parameters['exclude_projects'],
            authority_level=command_context.parameters['authority_level']
        )

    async def should_handle_exceptions_gracefully(self, slice_command, command_context, mocker):
        """Test that SliceCommand handles exceptions and returns error result."""
        # Mock do_slice to raise an exception
        mock_do_slice = mocker.patch('context_mixer.commands.slice.do_slice')
        test_error = Exception("Test slice error")
        mock_do_slice.side_effect = test_error
        
        result = await slice_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to slice context: Test slice error"
        assert result.error == test_error

    async def should_use_default_parameters_when_not_provided(self, slice_command, mock_console, mock_config, mock_llm_gateway, mocker):
        """Test that SliceCommand uses default parameters when not provided."""
        mock_do_slice = mocker.patch('context_mixer.commands.slice.do_slice')
        
        # Create context with minimal parameters
        minimal_context = CommandContext(
            console=mock_console,
            config=mock_config,
            llm_gateway=mock_llm_gateway,
            parameters={}
        )
        
        result = await slice_command.execute(minimal_context)
        
        assert result.success is True
        
        # Verify do_slice was called with default values
        call_args = mock_do_slice.call_args
        assert call_args[1]['granularity'] == 'basic'  # Default value
        assert call_args[1]['output_path'] is None
        assert call_args[1]['domains'] is None
        assert call_args[1]['project_ids'] is None
        assert call_args[1]['exclude_projects'] is None
        assert call_args[1]['authority_level'] is None

    async def should_return_command_result_with_parameter_data(self, slice_command, command_context, mocker):
        """Test that SliceCommand returns result with parameter data."""
        mocker.patch('context_mixer.commands.slice.do_slice')
        
        result = await slice_command.execute(command_context)
        
        assert result.success is True
        assert result.data['output_path'] == str(command_context.parameters['output_path'])
        assert result.data['granularity'] == command_context.parameters['granularity']
        assert result.data['domains'] == command_context.parameters['domains']
        assert result.data['project_ids'] == command_context.parameters['project_ids']
        assert result.data['exclude_projects'] == command_context.parameters['exclude_projects']
        assert result.data['authority_level'] == command_context.parameters['authority_level']

    async def should_pass_all_context_dependencies(self, slice_command, command_context, mocker):
        """Test that SliceCommand passes all required context dependencies."""
        mock_do_slice = mocker.patch('context_mixer.commands.slice.do_slice')
        
        await slice_command.execute(command_context)
        
        # Verify all required dependencies were passed
        call_args = mock_do_slice.call_args
        assert call_args[1]['console'] == command_context.console
        assert call_args[1]['config'] == command_context.config
        assert call_args[1]['llm_gateway'] == command_context.llm_gateway

    async def should_handle_path_parameter_conversion(self, slice_command, mock_console, mock_config, mock_llm_gateway, mocker):
        """Test that SliceCommand handles Path parameter conversion correctly."""
        mock_do_slice = mocker.patch('context_mixer.commands.slice.do_slice')
        
        # Create context with string path that should be converted
        context_with_string_path = CommandContext(
            console=mock_console,
            config=mock_config,
            llm_gateway=mock_llm_gateway,
            parameters={'output_path': '/test/string/path'}
        )
        
        result = await slice_command.execute(context_with_string_path)
        
        assert result.success is True
        # The result data should contain the string representation
        assert result.data['output_path'] == '/test/string/path'