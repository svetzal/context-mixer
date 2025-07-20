from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from context_mixer.commands.base import Command, CommandContext, CommandResult
from context_mixer.config import Config
from context_mixer.domain.events import EventBus


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
        config=mock_config,
        parameters={'test_param': 'test_value'}
    )


class DescribeCommandContext:
    def should_initialize_with_required_parameters(self, mock_console, mock_config):
        context = CommandContext(console=mock_console, config=mock_config)

        assert context.console == mock_console
        assert context.config == mock_config
        assert context.llm_gateway is None
        assert context.git_gateway is None
        assert context.knowledge_store is None
        assert context.event_bus is not None  # Should be auto-initialized
        assert isinstance(context.event_bus, EventBus)
        assert context.parameters == {}

    def should_initialize_parameters_dict_when_none(self, mock_console, mock_config):
        context = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters=None
        )

        assert context.parameters == {}

    def should_preserve_provided_parameters(self, mock_console, mock_config):
        params = {'key1': 'value1', 'key2': 'value2'}
        context = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters=params
        )

        assert context.parameters == params


class DescribeCommandResult:
    def should_initialize_with_required_success_parameter(self):
        result = CommandResult(success=True)

        assert result.success is True
        assert result.message is None
        assert result.data == {}
        assert result.error is None

    def should_initialize_data_dict_when_none(self):
        result = CommandResult(success=True, data=None)

        assert result.data == {}

    def should_preserve_provided_data(self):
        data = {'key1': 'value1', 'key2': 'value2'}
        result = CommandResult(success=True, data=data)

        assert result.data == data

    def should_handle_error_result(self):
        error = Exception("Test error")
        result = CommandResult(
            success=False,
            message="Operation failed",
            error=error
        )

        assert result.success is False
        assert result.message == "Operation failed"
        assert result.error == error


class TestCommand(Command):
    """Test implementation of Command for testing purposes."""

    async def execute(self, context: CommandContext) -> CommandResult:
        return CommandResult(
            success=True,
            message="Test command executed",
            data={'test_param': context.parameters.get('test_param')}
        )


class DescribeCommand:
    def should_be_abstract_base_class(self):
        # Should not be able to instantiate Command directly
        with pytest.raises(TypeError):
            Command()

    def should_require_execute_method_implementation(self):
        class IncompleteCommand(Command):
            pass

        # Should not be able to instantiate without implementing execute
        with pytest.raises(TypeError):
            IncompleteCommand()

    async def should_execute_successfully_with_context(self, command_context):
        command = TestCommand()

        result = await command.execute(command_context)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.message == "Test command executed"
        assert result.data['test_param'] == 'test_value'


class DescribeCommandPatternIntegration:
    async def should_support_command_composition(self, command_context):
        """Test that commands can be composed and chained."""
        command1 = TestCommand()

        # Execute first command
        result1 = await command1.execute(command_context)
        assert result1.success is True

        # Use result from first command in second command context
        command_context.parameters['previous_result'] = result1.data
        command2 = TestCommand()

        result2 = await command2.execute(command_context)
        assert result2.success is True
        assert 'previous_result' in command_context.parameters

    async def should_handle_command_errors_gracefully(self, command_context):
        """Test that command errors are handled properly."""

        class FailingCommand(Command):
            async def execute(self, context: CommandContext) -> CommandResult:
                raise Exception("Simulated command failure")

        command = FailingCommand()

        # The command should raise the exception, but in practice,
        # commands should catch exceptions and return CommandResult with error
        with pytest.raises(Exception, match="Simulated command failure"):
            await command.execute(command_context)

    async def should_support_testable_isolation(self, mock_console, mock_config):
        """Test that commands are easily testable in isolation."""
        # Create minimal context for testing
        test_context = CommandContext(
            console=mock_console,
            config=mock_config,
            parameters={'isolated_test': True}
        )

        command = TestCommand()
        result = await command.execute(test_context)

        assert result.success is True
        assert result.data['test_param'] is None  # Not provided in test context

        # Verify no external dependencies were called
        mock_console.print.assert_not_called()
