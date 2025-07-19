"""
Tests for Conflict Resolution Strategies

This module contains comprehensive tests for the Strategy pattern implementation
of conflict resolution.
"""

import pytest
from unittest.mock import MagicMock
from rich.console import Console

from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from context_mixer.commands.interactions.conflict_resolution_strategies import (
    ConflictResolutionStrategy,
    UserInteractiveResolutionStrategy,
    AutomaticResolutionStrategy,
    LLMBasedResolutionStrategy,
    ConflictResolutionContext,
    ConflictResolutionStrategyFactory
)


@pytest.fixture
def sample_conflict():
    """Create a sample conflict for testing."""
    return Conflict(
        description="Test conflict description",
        conflicting_guidance=[
            ConflictingGuidance(content="Use tabs for indentation", source="existing"),
            ConflictingGuidance(content="Use spaces for indentation", source="new")
        ]
    )


@pytest.fixture
def sample_conflicts(sample_conflict):
    """Create a list of sample conflicts for testing."""
    conflict2 = Conflict(
        description="Another test conflict",
        conflicting_guidance=[
            ConflictingGuidance(content="Use camelCase", source="style_guide"),
            ConflictingGuidance(content="Use snake_case", source="existing")
        ]
    )
    return [sample_conflict, conflict2]


@pytest.fixture
def mock_console(mocker):
    """Create a mock console for testing."""
    return mocker.MagicMock(spec=Console)


@pytest.fixture
def mock_llm_gateway(mocker):
    """Create a mock LLM gateway for testing."""
    return mocker.MagicMock()


class DescribeUserInteractiveResolutionStrategy:

    @pytest.fixture
    def strategy(self):
        return UserInteractiveResolutionStrategy()

    def should_return_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "UserInteractive"

    def should_return_empty_list_for_no_conflicts(self, strategy, mock_console):
        result = strategy.resolve_conflicts([], mock_console)
        assert result == []

    def should_resolve_conflict_with_user_choice(self, strategy, sample_conflict, mock_console):
        # Mock user choosing option 1 (first guidance)
        mock_console.input.return_value = "1"

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use tabs for indentation"
        assert result[0].description == sample_conflict.description
        assert result[0].conflicting_guidance == sample_conflict.conflicting_guidance

    def should_resolve_conflict_with_custom_resolution(self, strategy, sample_conflict, mock_console):
        # Mock user choosing custom option (option 3 for 2 guidance items)
        # Then entering a single line followed by termination dot
        mock_console.input.side_effect = ["3", "Use 4 spaces for indentation", "."]

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use 4 spaces for indentation"

    def should_resolve_conflict_with_multiline_custom_resolution(self, strategy, sample_conflict, mock_console):
        # Mock user choosing custom option (option 3 for 2 guidance items)
        # Then entering multiple lines followed by termination dot
        mock_console.input.side_effect = [
            "3",  # Choose custom resolution option
            "Use 4 spaces for indentation.",
            "This applies to all Python files.",
            "Follow PEP 8 guidelines.",
            "."  # Termination dot
        ]

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        expected_resolution = "Use 4 spaces for indentation.\nThis applies to all Python files.\nFollow PEP 8 guidelines."
        assert len(result) == 1
        assert result[0].resolution == expected_resolution

    def should_handle_invalid_input_then_valid_choice(self, strategy, sample_conflict, mock_console):
        # Mock invalid input followed by valid choice
        mock_console.input.side_effect = ["invalid", "0", "999", "2"]

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use spaces for indentation"
        # Should have printed error messages
        assert mock_console.print.call_count >= 2

    def should_create_default_console_when_none_provided(self, strategy, sample_conflict, mocker):
        # Mock the Console constructor
        mock_console_class = mocker.patch('context_mixer.commands.interactions.conflict_resolution_strategies.Console')
        mock_console_instance = MagicMock()
        mock_console_class.return_value = mock_console_instance
        mock_console_instance.input.return_value = "1"

        result = strategy.resolve_conflicts([sample_conflict])

        assert len(result) == 1
        mock_console_class.assert_called_once()

    def should_resolve_conflict_with_not_a_conflict_option(self, strategy, sample_conflict, mock_console):
        # Mock user choosing "This is not a conflict" option (option 4 for 2 guidance items)
        mock_console.input.return_value = "4"

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution is None
        assert result[0].description == sample_conflict.description
        assert result[0].conflicting_guidance == sample_conflict.conflicting_guidance


class DescribeAutomaticResolutionStrategy:

    @pytest.fixture
    def strategy(self):
        return AutomaticResolutionStrategy()

    def should_return_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "Automatic"

    def should_return_empty_list_for_no_conflicts(self, strategy, mock_console):
        result = strategy.resolve_conflicts([], mock_console)
        assert result == []

    def should_prefer_existing_source_by_default(self, strategy, sample_conflict, mock_console):
        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use tabs for indentation"  # from "existing" source

    def should_fallback_to_first_when_no_existing_source(self, strategy, mock_console):
        conflict = Conflict(
            description="Test conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Option A", source="source1"),
                ConflictingGuidance(content="Option B", source="source2")
            ]
        )

        result = strategy.resolve_conflicts([conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Option A"  # first guidance

    def should_handle_empty_guidance_list(self, strategy, mock_console):
        conflict = Conflict(
            description="Empty conflict",
            conflicting_guidance=[]
        )

        result = strategy.resolve_conflicts([conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == ""

    def should_respect_prefer_existing_false(self, sample_conflict, mock_console):
        strategy = AutomaticResolutionStrategy(prefer_existing=False)

        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use tabs for indentation"  # first guidance

    def should_respect_fallback_to_first_false(self, mock_console):
        strategy = AutomaticResolutionStrategy(prefer_existing=False, fallback_to_first=False)
        conflict = Conflict(
            description="Test conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Option A", source="source1"),
                ConflictingGuidance(content="Option B", source="source2")
            ]
        )

        result = strategy.resolve_conflicts([conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == ""

    def should_log_resolution_when_console_provided(self, strategy, sample_conflict, mock_console):
        result = strategy.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        mock_console.print.assert_called()


class DescribeLLMBasedResolutionStrategy:

    @pytest.fixture
    def strategy_with_gateway(self, mock_llm_gateway):
        return LLMBasedResolutionStrategy(mock_llm_gateway)

    @pytest.fixture
    def strategy_without_gateway(self):
        return LLMBasedResolutionStrategy()

    def should_return_strategy_name(self, strategy_without_gateway):
        assert strategy_without_gateway.get_strategy_name() == "LLMBased"

    def should_return_empty_list_for_no_conflicts(self, strategy_with_gateway, mock_console):
        result = strategy_with_gateway.resolve_conflicts([], mock_console)
        assert result == []

    def should_fallback_to_automatic_when_no_gateway(self, strategy_without_gateway, sample_conflict, mock_console):
        result = strategy_without_gateway.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use tabs for indentation"  # automatic strategy result
        mock_console.print.assert_called_with("[yellow]No LLM gateway available, falling back to automatic resolution[/yellow]")

    def should_use_automatic_fallback_on_llm_error(self, strategy_with_gateway, sample_conflict, mock_console):
        # For now, since LLM integration is placeholder, it should fall back to automatic
        result = strategy_with_gateway.resolve_conflicts([sample_conflict], mock_console)

        assert len(result) == 1
        assert result[0].resolution == "Use tabs for indentation"  # automatic strategy result

    def should_build_proper_resolution_prompt(self, strategy_with_gateway, sample_conflict):
        prompt = strategy_with_gateway._build_resolution_prompt(sample_conflict)

        assert "Test conflict description" in prompt
        assert "Use tabs for indentation" in prompt
        assert "Use spaces for indentation" in prompt
        assert "existing" in prompt
        assert "new" in prompt
        assert "Resolution:" in prompt


class DescribeConflictResolutionContext:

    @pytest.fixture
    def mock_strategy(self, mocker):
        return mocker.MagicMock(spec=ConflictResolutionStrategy)

    @pytest.fixture
    def context(self, mock_strategy):
        return ConflictResolutionContext(mock_strategy)

    def should_initialize_with_strategy(self, context, mock_strategy):
        assert context.get_strategy() == mock_strategy

    def should_allow_strategy_change(self, context, mock_strategy):
        new_strategy = MagicMock(spec=ConflictResolutionStrategy)
        context.set_strategy(new_strategy)

        assert context.get_strategy() == new_strategy
        assert context.get_strategy() != mock_strategy

    def should_delegate_resolve_conflicts_to_strategy(self, context, mock_strategy, sample_conflicts, mock_console):
        expected_result = [sample_conflicts[0]]  # mock return value
        mock_strategy.resolve_conflicts.return_value = expected_result

        result = context.resolve_conflicts(sample_conflicts, mock_console)

        assert result == expected_result
        mock_strategy.resolve_conflicts.assert_called_once_with(sample_conflicts, mock_console)


class DescribeConflictResolutionStrategyFactory:

    def should_create_interactive_strategy(self):
        strategy = ConflictResolutionStrategyFactory.create_strategy("interactive")
        assert isinstance(strategy, UserInteractiveResolutionStrategy)

    def should_create_interactive_strategy_with_aliases(self):
        for alias in ["user", "manual"]:
            strategy = ConflictResolutionStrategyFactory.create_strategy(alias)
            assert isinstance(strategy, UserInteractiveResolutionStrategy)

    def should_create_automatic_strategy(self):
        strategy = ConflictResolutionStrategyFactory.create_strategy("automatic")
        assert isinstance(strategy, AutomaticResolutionStrategy)

    def should_create_automatic_strategy_with_aliases(self):
        for alias in ["auto", "rule-based"]:
            strategy = ConflictResolutionStrategyFactory.create_strategy(alias)
            assert isinstance(strategy, AutomaticResolutionStrategy)

    def should_create_llm_strategy(self):
        strategy = ConflictResolutionStrategyFactory.create_strategy("llm")
        assert isinstance(strategy, LLMBasedResolutionStrategy)

    def should_create_llm_strategy_with_aliases(self):
        for alias in ["ai", "llm-based"]:
            strategy = ConflictResolutionStrategyFactory.create_strategy(alias)
            assert isinstance(strategy, LLMBasedResolutionStrategy)

    def should_pass_kwargs_to_automatic_strategy(self):
        strategy = ConflictResolutionStrategyFactory.create_strategy(
            "automatic", 
            prefer_existing=False, 
            fallback_to_first=False
        )
        assert isinstance(strategy, AutomaticResolutionStrategy)
        assert strategy.prefer_existing is False
        assert strategy.fallback_to_first is False

    def should_pass_kwargs_to_llm_strategy(self, mock_llm_gateway):
        strategy = ConflictResolutionStrategyFactory.create_strategy(
            "llm", 
            llm_gateway=mock_llm_gateway
        )
        assert isinstance(strategy, LLMBasedResolutionStrategy)
        assert strategy.llm_gateway == mock_llm_gateway

    def should_handle_case_insensitive_strategy_types(self):
        strategy = ConflictResolutionStrategyFactory.create_strategy("INTERACTIVE")
        assert isinstance(strategy, UserInteractiveResolutionStrategy)

    def should_raise_error_for_unknown_strategy_type(self):
        with pytest.raises(ValueError, match="Unknown strategy type: unknown"):
            ConflictResolutionStrategyFactory.create_strategy("unknown")

    def should_return_available_strategies(self):
        strategies = ConflictResolutionStrategyFactory.get_available_strategies()
        assert strategies == ["interactive", "automatic", "llm"]


class DescribeStrategyIntegration:
    """Integration tests for the strategy pattern."""

    def should_maintain_backward_compatibility_with_protocol(self, sample_conflicts, mock_console):
        # Test that all strategies implement the ConflictResolver protocol
        strategies = [
            UserInteractiveResolutionStrategy(),
            AutomaticResolutionStrategy(),
            LLMBasedResolutionStrategy()
        ]

        for strategy in strategies:
            # Should have resolve_conflicts method
            assert hasattr(strategy, 'resolve_conflicts')
            assert callable(getattr(strategy, 'resolve_conflicts'))

            # Should be able to resolve conflicts (though interactive will need mocking)
            if isinstance(strategy, UserInteractiveResolutionStrategy):
                mock_console.input.return_value = "1"
                result = strategy.resolve_conflicts(sample_conflicts, mock_console)
            else:
                result = strategy.resolve_conflicts(sample_conflicts, mock_console)

            assert isinstance(result, list)
            assert len(result) == len(sample_conflicts)
            for conflict in result:
                assert isinstance(conflict, Conflict)
                assert conflict.resolution is not None

    def should_allow_runtime_strategy_switching(self, sample_conflicts, mock_console):
        # Start with automatic strategy
        context = ConflictResolutionContext(AutomaticResolutionStrategy())

        # Resolve with automatic strategy
        result1 = context.resolve_conflicts(sample_conflicts, mock_console)
        assert result1[0].resolution == "Use tabs for indentation"  # existing preference

        # Switch to LLM strategy (will fallback to automatic)
        context.set_strategy(LLMBasedResolutionStrategy())
        result2 = context.resolve_conflicts(sample_conflicts, mock_console)
        assert result2[0].resolution == "Use tabs for indentation"  # same result due to fallback

        # Verify strategy was actually changed
        assert isinstance(context.get_strategy(), LLMBasedResolutionStrategy)
