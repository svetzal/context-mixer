"""
Tests for the ingest command.
"""

import pytest
from pathlib import Path

from mojentic.llm import LLMBroker
from rich.panel import Panel

from prompt_mixer.commands.ingest import do_ingest


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()

@pytest.fixture
def mock_broker(mocker):
    return mocker.MagicMock(spec=LLMBroker)

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

    def should_print_messages_when_ingesting(self, mock_console, mock_config, mock_broker, mock_path):
        do_ingest(console=mock_console, config=mock_config, llm_broker=mock_broker, filename=mock_path)

        assert mock_console.print.call_count >= 2  # At least 2 print calls
        panel_call = mock_console.print.call_args_list[0]
        assert isinstance(panel_call[0][0], Panel)
        assert "Successfully imported prompt as context.md" in mock_console.print.call_args_list[1][0][0]

    def should_create_context_file_with_correct_content(self, mock_console, mock_config, mock_broker, mock_path):
        test_content = mock_path.read_text()

        do_ingest(console=mock_console, config=mock_config, llm_broker=mock_broker, filename=mock_path)

        output_file = mock_config.library_path / "context.md"
        assert output_file.exists()
        assert output_file.read_text() == test_content
