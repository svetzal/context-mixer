"""
Tests for the ingest command.
"""

import pytest
from pathlib import Path

from rich.panel import Panel

from prompt_mixer.commands.ingest import do_ingest
from prompt_mixer.gateways.llm import LLMGateway, Message
from prompt_mixer.spec_helpers import MessageMatcher


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()

@pytest.fixture
def mock_llm_gateway(mocker):
    mock = mocker.MagicMock(spec=LLMGateway)
    # Set up the generate method to return the merged content
    mock.generate.return_value = "Merged content from LLM"
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

        output_file = mock_config.library_path / "context.md"
        assert output_file.exists()
        assert output_file.read_text() == test_content

    def should_merge_with_existing_context_file(self, mock_console, mock_config, mock_llm_gateway, mock_path):
        # Create an existing context.md file
        existing_content = "Existing line 1\nExisting line 2\nShared line"
        output_file = mock_config.library_path / "context.md"
        output_file.write_text(existing_content)

        # Set up new content with some overlap
        mock_path.write_text("New line 1\nNew line 2\nShared line")

        do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, filename=mock_path)

        # Check that the merged content is the response from the LLM broker
        merged_content = output_file.read_text()
        assert merged_content == "Merged content from LLM"

        # Verify that generate was called with a message containing both contents
        expected_content = ["Existing line 1", "New line 1"]
        mock_llm_gateway.generate.assert_called_once_with(messages=MessageMatcher(expected_content))

        # Check that the correct message was printed
        assert "Successfully merged prompt with existing context.md" in mock_console.print.call_args_list[1][0][0]
