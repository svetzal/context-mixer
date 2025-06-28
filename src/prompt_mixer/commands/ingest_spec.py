"""
Tests for the ingest command.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from mojentic.llm import LLMBroker
from rich.panel import Panel

from prompt_mixer.commands.ingest import do_ingest


@pytest.fixture
def mock_console():
    """Create a mock console."""
    return MagicMock()

@pytest.fixture
def mock_broker():
    """Create a mock LLMBroker."""
    return MagicMock(LLMBroker)

@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config."""
    config = MagicMock()
    library_path = tmp_path / "library"
    library_path.mkdir(exist_ok=True)
    config.library_path = library_path
    return config


@pytest.fixture
def mock_path(tmp_path):
    """Create a temporary path for testing."""
    test_file = tmp_path / "test-file.md"
    test_file.write_text("Test content")
    return test_file


class DescribeDoIngest:
    """Tests for the do_ingest function."""

    def should_print_messages_when_ingesting(self, mock_console, mock_config, mock_broker, mock_path):
        """Should print messages when ingesting."""
        # Arrange
        # No additional arrangement needed

        # Act
        do_ingest(console=mock_console, config=mock_config, llm_broker=mock_broker, filename=mock_path)

        # Assert
        assert mock_console.print.call_count >= 2  # At least 2 print calls

        # Check for the Panel call (first call)
        panel_call = mock_console.print.call_args_list[0]
        assert isinstance(panel_call[0][0], Panel)

        # Check for the success message (second call)
        assert "Successfully imported prompt as instructions.md" in mock_console.print.call_args_list[1][0][0]
