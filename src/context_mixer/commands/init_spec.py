"""
Tests for the init command and InitCommand class.

Tests both the legacy do_init function and the new InitCommand class
that implements the Command pattern boundary.
"""

import pytest

from context_mixer.commands.base import CommandContext, CommandResult
from context_mixer.commands.init import do_init, InitCommand
from context_mixer.config import Config
from context_mixer.gateways.git import GitGateway


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path / "prompt-mixer"

@pytest.fixture
def mock_config(mock_path):
    return Config(mock_path)


@pytest.fixture
def mock_git_gateway(mocker):
    return mocker.MagicMock(spec=GitGateway)


class DescribeDoInit:

    def should_clone_repository_when_remote_is_provided(self, mock_git_gateway, mock_console, mock_path, mock_config):
        remote = "git@github.com:user/repo.git"
        mock_git_gateway.clone.return_value = (True, "Success")

        do_init(mock_console, mock_config, remote, None, None, mock_git_gateway)

        mock_git_gateway.clone.assert_called_once_with(remote, mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_initialize_new_repository_when_remote_not_provided(self, mock_git_gateway, mock_console, mock_path, mock_config):
        mock_git_gateway.init.return_value = (True, "Success")

        do_init(mock_console, mock_config, None, None, None, mock_git_gateway)

        mock_git_gateway.init.assert_called_once_with(mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_create_directory_if_it_doesnt_exist(self, mock_git_gateway, mock_console, mock_path, mock_config):
        mock_git_gateway.init.return_value = (True, "Success")

        do_init(mock_console, mock_config, None, None, None, mock_git_gateway)

        assert mock_path.exists()
        assert mock_path.is_dir()

    def should_handle_provider_and_model_parameters(self, mock_git_gateway, mock_console, mock_path, mock_config):
        provider = "ollama"
        model = "phi3"
        mock_git_gateway.init.return_value = (True, "Success")

        do_init(mock_console, mock_config, None, provider, model, mock_git_gateway)

        mock_git_gateway.init.assert_called_once_with(mock_path)
        message_found = False
        for args, _ in mock_console.print.call_args_list:
            if "Provider and model configuration" in str(args):
                message_found = True
                break
        assert message_found


@pytest.fixture
def init_command():
    return InitCommand()


@pytest.fixture
def command_context(mock_console, mock_config, mock_git_gateway):
    return CommandContext(
        console=mock_console,
        config=mock_config,
        git_gateway=mock_git_gateway,
        parameters={
            'remote': 'git@github.com:user/repo.git',
            'provider': 'ollama',
            'model': 'phi3'
        }
    )


class DescribeInitCommand:
    """Test the InitCommand class at the command boundary."""

    async def should_execute_successfully_with_valid_context(self, init_command, command_context, mocker):
        """Test that InitCommand executes successfully with valid context."""
        # Mock the do_init function
        mock_do_init = mocker.patch('context_mixer.commands.init.do_init')

        result = await init_command.execute(command_context)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Library initialized successfully"
        assert result.error is None

        # Verify do_init was called with correct parameters
        mock_do_init.assert_called_once_with(
            console=command_context.console,
            config=command_context.config,
            remote=command_context.parameters['remote'],
            provider=command_context.parameters['provider'],
            model=command_context.parameters['model'],
            git_gateway=command_context.git_gateway
        )

    async def should_handle_exceptions_gracefully(self, init_command, command_context, mocker):
        """Test that InitCommand handles exceptions and returns error result."""
        # Mock do_init to raise an exception
        mock_do_init = mocker.patch('context_mixer.commands.init.do_init')
        test_error = Exception("Test init error")
        mock_do_init.side_effect = test_error

        result = await init_command.execute(command_context)

        assert isinstance(result, CommandResult)
        assert result.success is False
        assert result.message == "Failed to initialize library: Test init error"
        assert result.error == test_error

    async def should_use_default_parameters_when_not_provided(self, init_command, mock_console, mock_config, mock_git_gateway, mocker):
        """Test that InitCommand uses default parameters when not provided."""
        mock_do_init = mocker.patch('context_mixer.commands.init.do_init')

        # Create context with minimal parameters
        minimal_context = CommandContext(
            console=mock_console,
            config=mock_config,
            git_gateway=mock_git_gateway,
            parameters={}
        )

        result = await init_command.execute(minimal_context)

        assert result.success is True

        # Verify do_init was called with None values for optional parameters
        call_args = mock_do_init.call_args
        assert call_args[1]['remote'] is None
        assert call_args[1]['provider'] is None
        assert call_args[1]['model'] is None
        assert call_args[1]['git_gateway'] == mock_git_gateway

    async def should_pass_all_context_dependencies(self, init_command, command_context, mocker):
        """Test that InitCommand passes all required context dependencies."""
        mock_do_init = mocker.patch('context_mixer.commands.init.do_init')

        await init_command.execute(command_context)

        # Verify all required dependencies were passed
        call_args = mock_do_init.call_args
        assert call_args[1]['console'] == command_context.console
        assert call_args[1]['config'] == command_context.config
        assert call_args[1]['git_gateway'] == command_context.git_gateway

    async def should_return_standardized_command_result(self, init_command, command_context, mocker):
        """Test that InitCommand returns a standardized CommandResult."""
        mocker.patch('context_mixer.commands.init.do_init')

        result = await init_command.execute(command_context)

        # Verify result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        assert hasattr(result, 'data')
        assert hasattr(result, 'error')
        assert isinstance(result.data, dict)
