from unittest.mock import MagicMock

import pytest
from rich.console import Console

from context_mixer.domain.events import EventBus
from context_mixer.domain.progress_events import (
    ProgressStartedEvent, ProgressUpdatedEvent, ProgressCompletedEvent, ProgressFailedEvent
)
from context_mixer.utils.event_driven_progress import (
    EventDrivenProgressTracker, EventPublishingProgressTracker, create_cli_progress_tracker
)
from context_mixer.utils.progress import ProgressObserver, ProgressUpdate, ProgressStatus


@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for each test."""
    return EventBus()


@pytest.fixture
def mock_observer():
    """Create a mock progress observer."""
    return MagicMock(spec=ProgressObserver)


@pytest.fixture
def event_driven_tracker(event_bus, mock_observer):
    """Create an EventDrivenProgressTracker with mock observer."""
    return EventDrivenProgressTracker(observer=mock_observer, event_bus=event_bus)


@pytest.fixture
def event_publishing_tracker(event_bus):
    """Create an EventPublishingProgressTracker."""
    return EventPublishingProgressTracker(event_bus=event_bus, project_id="test-project")


class DescribeEventDrivenProgressTracker:
    def should_subscribe_to_progress_events_on_initialization(self, event_bus, mock_observer):
        tracker = EventDrivenProgressTracker(observer=mock_observer, event_bus=event_bus)

        assert event_bus.get_handler_count("progress_started") == 1
        assert event_bus.get_handler_count("progress_updated") == 1
        assert event_bus.get_handler_count("progress_completed") == 1
        assert event_bus.get_handler_count("progress_failed") == 1

    def should_handle_progress_started_events(self, event_driven_tracker, event_bus, mock_observer):
        event = ProgressStartedEvent(
            event_id="test",
            timestamp=None,
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            total=100,
            project_id="test-project"
        )

        event_bus.publish(event)

        mock_observer.on_operation_start.assert_called_once_with("test_op", "Test Operation", 100)

    def should_handle_progress_updated_events(self, event_driven_tracker, event_bus, mock_observer):
        event = ProgressUpdatedEvent(
            event_id="test",
            timestamp=None,
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            current=50,
            total=100,
            message="Processing...",
            metadata={"key": "value"}
        )

        event_bus.publish(event)

        mock_observer.on_progress_update.assert_called_once()
        call_args = mock_observer.on_progress_update.call_args[0][0]
        assert isinstance(call_args, ProgressUpdate)
        assert call_args.operation_id == "test_op"
        assert call_args.operation_name == "Test Operation"
        assert call_args.current == 50
        assert call_args.total == 100
        assert call_args.status == ProgressStatus.IN_PROGRESS
        assert call_args.message == "Processing..."
        assert call_args.metadata == {"key": "value"}

    def should_handle_progress_completed_events(self, event_driven_tracker, event_bus, mock_observer):
        event = ProgressCompletedEvent(
            event_id="test",
            timestamp=None,
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            project_id="test-project"
        )

        event_bus.publish(event)

        mock_observer.on_operation_complete.assert_called_once_with("test_op", "Test Operation")

    def should_handle_progress_failed_events(self, event_driven_tracker, event_bus, mock_observer):
        event = ProgressFailedEvent(
            event_id="test",
            timestamp=None,
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            error="Test error",
            project_id="test-project"
        )

        event_bus.publish(event)

        mock_observer.on_operation_failed.assert_called_once_with("test_op", "Test Operation", "Test error")

    def should_work_without_observer(self, event_bus):
        tracker = EventDrivenProgressTracker(observer=None, event_bus=event_bus)
        
        # Should not raise exception when handling events without observer
        event = ProgressStartedEvent(
            event_id="test",
            timestamp=None,
            event_type="",
            operation_id="test_op",
            operation_name="Test Operation",
            total=100
        )
        
        event_bus.publish(event)  # Should not raise exception

    def should_unsubscribe_from_events(self, event_driven_tracker, event_bus):
        # Verify subscribed
        assert event_bus.get_handler_count("progress_started") == 1
        
        event_driven_tracker.unsubscribe()
        
        # Verify unsubscribed
        assert event_bus.get_handler_count("progress_started") == 0
        assert event_bus.get_handler_count("progress_updated") == 0
        assert event_bus.get_handler_count("progress_completed") == 0
        assert event_bus.get_handler_count("progress_failed") == 0


class DescribeEventPublishingProgressTracker:
    def should_publish_progress_started_event_on_start_operation(self, event_publishing_tracker, event_bus):
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("progress_started", event_handler)
        
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        
        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, ProgressStartedEvent)
        assert event.operation_id == "test_op"
        assert event.operation_name == "Test Operation"
        assert event.total == 100
        assert event.project_id == "test-project"

    def should_publish_progress_updated_event_on_update_progress(self, event_publishing_tracker, event_bus):
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("progress_updated", event_handler)
        
        # Start operation first
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        
        # Update progress
        event_publishing_tracker.update_progress("test_op", 50, "Processing...", {"key": "value"})
        
        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, ProgressUpdatedEvent)
        assert event.operation_id == "test_op"
        assert event.operation_name == "Test Operation"
        assert event.current == 50
        assert event.total == 100
        assert event.message == "Processing..."
        assert event.metadata == {"key": "value"}
        assert event.project_id == "test-project"

    def should_publish_progress_completed_event_on_complete_operation(self, event_publishing_tracker, event_bus):
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("progress_completed", event_handler)
        
        # Start operation first
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        
        # Complete operation
        event_publishing_tracker.complete_operation("test_op")
        
        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, ProgressCompletedEvent)
        assert event.operation_id == "test_op"
        assert event.operation_name == "Test Operation"
        assert event.project_id == "test-project"

    def should_publish_progress_failed_event_on_fail_operation(self, event_publishing_tracker, event_bus):
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("progress_failed", event_handler)
        
        # Start operation first
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        
        # Fail operation
        event_publishing_tracker.fail_operation("test_op", "Test error")
        
        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, ProgressFailedEvent)
        assert event.operation_id == "test_op"
        assert event.operation_name == "Test Operation"
        assert event.error == "Test error"
        assert event.project_id == "test-project"

    def should_ignore_updates_for_unknown_operations(self, event_publishing_tracker, event_bus):
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("progress_updated", event_handler)
        
        # Try to update progress for non-existent operation
        event_publishing_tracker.update_progress("unknown_op", 50)
        
        assert len(events_received) == 0

    def should_track_operation_state_internally(self, event_publishing_tracker):
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        
        assert "test_op" in event_publishing_tracker._operations
        assert event_publishing_tracker._operations["test_op"]["name"] == "Test Operation"
        assert event_publishing_tracker._operations["test_op"]["total"] == 100
        assert event_publishing_tracker._operations["test_op"]["current"] == 0
        assert event_publishing_tracker._operations["test_op"]["status"] == ProgressStatus.STARTED

    def should_remove_operation_on_completion(self, event_publishing_tracker):
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        assert "test_op" in event_publishing_tracker._operations
        
        event_publishing_tracker.complete_operation("test_op")
        assert "test_op" not in event_publishing_tracker._operations

    def should_remove_operation_on_failure(self, event_publishing_tracker):
        event_publishing_tracker.start_operation("test_op", "Test Operation", 100)
        assert "test_op" in event_publishing_tracker._operations
        
        event_publishing_tracker.fail_operation("test_op", "Test error")
        assert "test_op" not in event_publishing_tracker._operations


class DescribeEventDrivenProgressIntegration:
    def should_work_end_to_end(self, event_bus, mock_observer):
        # Create both trackers using the same event bus
        publishing_tracker = EventPublishingProgressTracker(event_bus=event_bus, project_id="integration-test")
        driven_tracker = EventDrivenProgressTracker(observer=mock_observer, event_bus=event_bus)
        
        # Simulate a complete operation lifecycle
        publishing_tracker.start_operation("integration_op", "Integration Test", 3)
        publishing_tracker.update_progress("integration_op", 1, "Step 1")
        publishing_tracker.update_progress("integration_op", 2, "Step 2")
        publishing_tracker.complete_operation("integration_op")
        
        # Verify all observer methods were called
        mock_observer.on_operation_start.assert_called_once_with("integration_op", "Integration Test", 3)
        assert mock_observer.on_progress_update.call_count == 2
        mock_observer.on_operation_complete.assert_called_once_with("integration_op", "Integration Test")


class DescribeCreateCliProgressTracker:
    def should_create_event_driven_tracker_with_cli_observer(self):
        console = MagicMock(spec=Console)
        
        tracker = create_cli_progress_tracker(console)
        
        assert isinstance(tracker, EventDrivenProgressTracker)
        assert tracker.observer is not None
        assert tracker.event_bus is not None

    def should_use_provided_event_bus(self):
        console = MagicMock(spec=Console)
        custom_event_bus = EventBus()
        
        tracker = create_cli_progress_tracker(console, custom_event_bus)
        
        assert tracker.event_bus is custom_event_bus