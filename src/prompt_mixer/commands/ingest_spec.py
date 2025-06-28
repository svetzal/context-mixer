"""
Tests for the ingest command.
"""

import pytest
from pathlib import Path

from rich.panel import Panel

from prompt_mixer.commands.ingest import do_ingest
from prompt_mixer.commands.operations.merge import detect_conflicts, resolve_conflict
from prompt_mixer.config import DEFAULT_ROOT_CONTEXT_FILENAME
from prompt_mixer.domain.conflict import Conflict, ConflictingGuidance
from prompt_mixer.gateways.llm import LLMGateway
from prompt_mixer.spec_helpers import MessageMatcher

MERGED_CONTENT_FROM_LLM = "Merged content from LLM"


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()

@pytest.fixture
def mock_llm_gateway(mocker):
    mock = mocker.MagicMock(spec=LLMGateway)
    # Set up the generate method to return the merged content
    mock.generate.return_value = MERGED_CONTENT_FROM_LLM
    return mock

@pytest.fixture
def mock_config(tmp_path, mocker):
    config = mocker.MagicMock()
    library_path = tmp_path / "library"
    library_path.mkdir(exist_ok=True)
    config.library_path = library_path
    return config

@pytest.fixture
def mock_path(tmp_path):
    test_file = tmp_path / "test-file.md"
    test_file.write_text("Test content")
    return test_file


class DescribeDoIngest:

    def should_print_messages_when_ingesting_to_empty_library(self, mock_console, mock_config, mock_llm_gateway, mock_path):
        do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, filename=mock_path)

        assert mock_console.print.call_count >= 2  # At least 2 print calls
        panel_call = mock_console.print.call_args_list[0]
        assert isinstance(panel_call[0][0], Panel)
        assert "Successfully imported prompt as context.md" in mock_console.print.call_args_list[1][0][0]

    def should_create_context_file_with_correct_content(self, mock_console, mock_config, mock_llm_gateway, mock_path):
        test_content = mock_path.read_text()

        do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, filename=mock_path)

        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        assert output_file.exists()
        assert output_file.read_text() == test_content

    def should_merge_with_existing_context_file(self, mock_console, mock_config, mock_llm_gateway, mock_path, mocker):
        # Mock detect_conflicts to return an empty ConflictList (no conflicts)
        from prompt_mixer.domain.conflict import ConflictList
        mocker.patch('prompt_mixer.commands.operations.merge.detect_conflicts', return_value=ConflictList(list=[]))

        # Create an existing context.md file
        existing_content = "Existing line 1\nExisting line 2\nShared line"
        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        output_file.write_text(existing_content)

        # Set up new content with some overlap
        mock_path.write_text("New line 1\nNew line 2\nShared line")

        do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, filename=mock_path)

        # Check that the merged content is the response from the LLM broker
        merged_content = output_file.read_text()
        assert merged_content == MERGED_CONTENT_FROM_LLM

        # Verify that generate was called with a message containing both contents
        expected_content = ["Existing line 1", "New line 1"]
        mock_llm_gateway.generate.assert_called_once_with(messages=MessageMatcher(expected_content))

        # Check that the correct message was printed
        assert "Successfully merged prompt with existing context.md" in mock_console.print.call_args_list[1][0][0]

    def should_detect_and_resolve_conflicts(self, mock_console, mock_config, mock_llm_gateway, mock_path, mocker):
        # Create a mock Conflict object
        indent_4_string = "Use 4 spaces for indentation"
        indent_2_string = "Use 2 spaces for indentation"
        conflict = Conflict(
            description="There is a conflict in the indentation guidance",
            conflicting_guidance=[
                ConflictingGuidance(content=indent_4_string, source="existing"),
                ConflictingGuidance(content=indent_2_string, source="new")
            ],
            resolution=None
        )

        # Mock detect_conflicts to return a ConflictList containing the mock Conflict
        from prompt_mixer.domain.conflict import ConflictList
        mock_detect_conflicts = mocker.patch('prompt_mixer.commands.operations.merge.detect_conflicts', return_value=ConflictList(list=[conflict]))

        # Mock resolve_conflict to return the list of conflicts with resolutions set
        # Create a copy of the conflict with resolution set
        resolved_conflict = Conflict(
            description="There is a conflict in the indentation guidance",
            conflicting_guidance=[
                ConflictingGuidance(content=indent_4_string, source="existing"),
                ConflictingGuidance(content=indent_2_string, source="new")
            ],
            resolution=indent_4_string
        )
        mock_resolve_conflict = mocker.patch('prompt_mixer.commands.operations.merge.resolve_conflict', return_value=[resolved_conflict])

        # Create an existing context.md file
        existing_content = "# Project Guidelines\n\n## Code Style\n\n- Use 4 spaces for indentation"
        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        output_file.write_text(existing_content)

        # Set up new content with conflicting guidance
        new_content = "# Coding Standards\n\n## Style Guide\n\n- Use 2 spaces for indentation"
        mock_path.write_text(new_content)

        # Set up mock_console.input to return "1" (choosing the first option)
        mock_console.input.return_value = "1"

        do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, filename=mock_path)

        # Check that detect_conflicts was called with the correct arguments
        mock_detect_conflicts.assert_called_once_with(existing_content, new_content, mock_llm_gateway)

        # Check that resolve_conflict was called with the list of conflicts and console
        mock_resolve_conflict.assert_called_once_with([conflict], mock_console)

        # Check that the merged content is the response from the LLM broker
        merged_content = output_file.read_text()
        assert merged_content == MERGED_CONTENT_FROM_LLM

        # Verify that generate was called with a message containing the resolved conflict
        expected_content = ["Resolution: Use 4 spaces for indentation"]
        mock_llm_gateway.generate.assert_called_once_with(messages=MessageMatcher(expected_content))

        # Check that the correct message was printed
        assert "Successfully merged prompt with existing context.md" in mock_console.print.call_args_list[-1][0][0]
