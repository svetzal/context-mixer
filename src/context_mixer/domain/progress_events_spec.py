import pytest
from unittest.mock import MagicMock
from datetime import datetime

from context_mixer.domain.progress_events import (
    ProgressStartedEvent, ProgressUpdatedEvent, ProgressCompletedEvent, ProgressFailedEvent
)


@pytest.fixture
def sample_progress_started_event():
    """Create a sample ProgressStartedEvent for testing."""
    return ProgressStartedEvent(
        event_id="progress-start-123",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        event_type="progress_started",
        operation_id="file_reading",
        operation_name="Reading files",
        total=10,
        project_id="test-project"
    )


@pytest.fixture
def sample_progress_updated_event():
    """Create a sample ProgressUpdatedEvent for testing."""
    return ProgressUpdatedEvent(
        event_id="progress-update-456",
        timestamp=datetime(2024, 1, 1, 12, 1, 0),
        event_type="progress_updated",
        operation_id="file_reading",
        operation_name="Reading files",
        current=5,
        total=10,
        message="Reading file.txt",
        metadata={"file_name": "file.txt"},
        project_id="test-project"
    )


@pytest.fixture
def sample_progress_completed_event():
    """Create a sample ProgressCompletedEvent for testing."""
    return ProgressCompletedEvent(
        event_id="progress-complete-789",
        timestamp=datetime(2024, 1, 1, 12, 2, 0),
        event_type="progress_completed",
        operation_id="file_reading",
        operation_name="Reading files",
        project_id="test-project"
    )


@pytest.fixture
def sample_progress_failed_event():
    """Create a sample ProgressFailedEvent for testing."""
    return ProgressFailedEvent(
        event_id="progress-failed-101",
        timestamp=datetime(2024, 1, 1, 12, 3, 0),
        event_type="progress_failed",
        operation_id="file_reading",
        operation_name="Reading files",
        error="File not found",
        project_id="test-project"
    )


class DescribeProgressStartedEvent:
    def should_set_correct_event_type(self):
        event = ProgressStartedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            total=100
        )

        assert event.event_type == "progress_started"

    def should_contain_all_required_data(self, sample_progress_started_event):
        event = sample_progress_started_event

        assert event.operation_id == "file_reading"
        assert event.operation_name == "Reading files"
        assert event.total == 10
        assert event.project_id == "test-project"


class DescribeProgressUpdatedEvent:
    def should_set_correct_event_type(self):
        event = ProgressUpdatedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            current=50,
            total=100
        )

        assert event.event_type == "progress_updated"

    def should_contain_all_required_data(self, sample_progress_updated_event):
        event = sample_progress_updated_event

        assert event.operation_id == "file_reading"
        assert event.operation_name == "Reading files"
        assert event.current == 5
        assert event.total == 10
        assert event.message == "Reading file.txt"
        assert event.metadata == {"file_name": "file.txt"}
        assert event.project_id == "test-project"

    def should_calculate_percentage_correctly(self, sample_progress_updated_event):
        event = sample_progress_updated_event

        assert event.percentage == 50.0

    def should_handle_zero_total_in_percentage(self):
        event = ProgressUpdatedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            current=5,
            total=0
        )

        assert event.percentage == 0.0

    def should_cap_percentage_at_100(self):
        event = ProgressUpdatedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            current=150,
            total=100
        )

        assert event.percentage == 100.0


class DescribeProgressCompletedEvent:
    def should_set_correct_event_type(self):
        event = ProgressCompletedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation"
        )

        assert event.event_type == "progress_completed"

    def should_contain_all_required_data(self, sample_progress_completed_event):
        event = sample_progress_completed_event

        assert event.operation_id == "file_reading"
        assert event.operation_name == "Reading files"
        assert event.project_id == "test-project"


class DescribeProgressFailedEvent:
    def should_set_correct_event_type(self):
        event = ProgressFailedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            error="Test error"
        )

        assert event.event_type == "progress_failed"

    def should_contain_all_required_data(self, sample_progress_failed_event):
        event = sample_progress_failed_event

        assert event.operation_id == "file_reading"
        assert event.operation_name == "Reading files"
        assert event.error == "File not found"
        assert event.project_id == "test-project"