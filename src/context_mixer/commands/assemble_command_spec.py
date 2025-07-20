"""
Tests for the AssembleCommand class.

Tests the command boundary as specified in the issue - the Command pattern
implementation provides a perfect testing point for command behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from context_mixer.commands.assemble import AssembleCommand
from context_mixer.commands.base import CommandContext, CommandResult
from context_mixer.config import Config


@pytest.fixture
def mock_console():
    return MagicMock(spec=Console)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.library_path = Path("/test/library")
    return config


@pytest.fixture
def assemble_command():
    return AssembleCommand()


@pytest.fixture
def command_context(mock_console, mock_config):
    return CommandContext(
        console=mock_console,
        config=mock_config,
        parameters={
            'target': 'copilot',
            'output': Path('/test/output.md'),
            'profile': 'python-dev',
            'filter_tags': 'lang:python,layer:testing',
            'project_ids': ['proj1', 'proj2'],
            'exclude_projects': ['proj3'],
            'token_budget': 4096,
            'quality_threshold': 0.9,
            'verbose': True
        }
    )


class DescribeAssembleCommand:
    """Test the AssembleCommand class at the command boundary."""

    async def should_execute_successfully_with_valid_context(self, assemble_command, command_context, mocker):
        """Test that AssembleCommand executes successfully with valid context."""
        # Mock the do_assemble function
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        result = await assemble_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Context assembled successfully"
        assert result.error is None
        
        # Verify do_assemble was called with correct parameters
        mock_do_assemble.assert_called_once_with(
            console=command_context.console,
            config=command_context.config,
            target=command_context.parameters['target'],
            output=command_context.parameters['output'],
            profile=command_context.parameters['profile'],
            filter_tags=command_context.parameters['filter_tags'],
            project_ids=command_context.parameters['project_ids'],
            exclude_projects=command_context.parameters['exclude_projects'],
            token_budget=command_context.parameters['token_budget'],
            quality_threshold=command_context.parameters['quality_threshold'],
            verbose=command_context.parameters['verbose']
        )

    async def should_handle_exceptions_gracefully(self, assemble_command, command_context, mocker):
        """Test that AssembleCommand handles exceptions and returns error result."""
        # Mock do_assemble to raise an exception
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        test_error = Exception("Test assemble error")
        mock_do_assemble.side_effect = test_error
        
        result = await assemble_command.execute(command_context)
        
        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to assemble context: Test assemble error"
        assert result.error == test_error

    async def should_use_default_parameters_when_not_provided(self, assemble_command, mock_console, mock_config, mocker):
        """Test that AssembleCommand uses default parameters when not provided."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        # Create context with minimal parameters (only target is required)
        minimal_context = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'target': 'claude'}
        )
        
        result = await assemble_command.execute(minimal_context)
        
        assert result.success is True
        
        # Verify do_assemble was called with default values
        call_args = mock_do_assemble.call_args
        assert call_args[1]['target'] == 'claude'
        assert call_args[1]['output'] is None
        assert call_args[1]['profile'] is None
        assert call_args[1]['filter_tags'] is None
        assert call_args[1]['project_ids'] is None
        assert call_args[1]['exclude_projects'] is None
        assert call_args[1]['token_budget'] == 8192  # Default value
        assert call_args[1]['quality_threshold'] == 0.8  # Default value
        assert call_args[1]['verbose'] is False  # Default value

    async def should_return_command_result_with_parameter_data(self, assemble_command, command_context, mocker):
        """Test that AssembleCommand returns result with parameter data."""
        mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        result = await assemble_command.execute(command_context)
        
        assert result.success is True
        assert result.data['target'] == command_context.parameters['target']
        assert result.data['output'] == str(command_context.parameters['output'])
        assert result.data['token_budget'] == command_context.parameters['token_budget']
        assert result.data['quality_threshold'] == command_context.parameters['quality_threshold']

    async def should_pass_all_context_dependencies(self, assemble_command, command_context, mocker):
        """Test that AssembleCommand passes all required context dependencies."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        await assemble_command.execute(command_context)
        
        # Verify all required dependencies were passed
        call_args = mock_do_assemble.call_args
        assert call_args[1]['console'] == command_context.console
        assert call_args[1]['config'] == command_context.config

    async def should_handle_none_output_path_correctly(self, assemble_command, mock_console, mock_config, mocker):
        """Test that AssembleCommand handles None output path correctly."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        context_with_none_output = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'target': 'cursor', 'output': None}
        )
        
        result = await assemble_command.execute(context_with_none_output)
        
        assert result.success is True
        assert result.data['output'] is None

    async def should_validate_required_target_parameter(self, assemble_command, mock_console, mock_config, mocker):
        """Test that AssembleCommand handles missing target parameter."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        # Create context without target parameter
        context_without_target = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={}
        )
        
        result = await assemble_command.execute(context_without_target)
        
        # The command should still execute (target defaults to None)
        # but do_assemble will be called with None target
        assert result.success is True
        call_args = mock_do_assemble.call_args
        assert call_args[1]['target'] is None

    async def should_handle_numeric_parameters_correctly(self, assemble_command, mock_console, mock_config, mocker):
        """Test that AssembleCommand handles numeric parameters correctly."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        context_with_numbers = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={
                'target': 'copilot',
                'token_budget': 16384,
                'quality_threshold': 0.95
            }
        )
        
        result = await assemble_command.execute(context_with_numbers)
        
        assert result.success is True
        call_args = mock_do_assemble.call_args
        assert call_args[1]['token_budget'] == 16384
        assert call_args[1]['quality_threshold'] == 0.95

    async def should_handle_list_parameters_correctly(self, assemble_command, mock_console, mock_config, mocker):
        """Test that AssembleCommand handles list parameters correctly."""
        mock_do_assemble = mocker.patch('context_mixer.commands.assemble.do_assemble')
        
        context_with_lists = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={
                'target': 'claude',
                'project_ids': ['web-app', 'api-service'],
                'exclude_projects': ['legacy-system']
            }
        )
        
        result = await assemble_command.execute(context_with_lists)
        
        assert result.success is True
        call_args = mock_do_assemble.call_args
        assert call_args[1]['project_ids'] == ['web-app', 'api-service']
        assert call_args[1]['exclude_projects'] == ['legacy-system']