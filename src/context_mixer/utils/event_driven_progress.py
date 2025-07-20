from typing import Optional

from rich.console import Console

from context_mixer.domain.events import EventBus, get_event_bus
from context_mixer.domain.progress_events import (
    ProgressStartedEvent, ProgressUpdatedEvent, ProgressCompletedEvent, ProgressFailedEvent
)
from .cli_progress import CLIProgressObserver
from .progress import ProgressObserver, ProgressUpdate, ProgressStatus


class EventDrivenProgressTracker:
    """
    Progress tracker that subscribes to progress events instead of requiring direct injection.
    
    This replaces the need for direct ProgressTracker dependency injection by listening
    to progress events published through the event bus.
    """
    
    def __init__(self, observer: Optional[ProgressObserver] = None, event_bus: Optional[EventBus] = None):
        """
        Initialize the event-driven progress tracker.
        
        Args:
            observer: The progress observer to notify (e.g., CLIProgressObserver)
            event_bus: The event bus to subscribe to (defaults to global event bus)
        """
        self.observer = observer
        self.event_bus = event_bus or get_event_bus()
        self._subscribe_to_events()
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to all progress-related events."""
        self.event_bus.subscribe("progress_started", self._handle_progress_started)
        self.event_bus.subscribe("progress_updated", self._handle_progress_updated)
        self.event_bus.subscribe("progress_completed", self._handle_progress_completed)
        self.event_bus.subscribe("progress_failed", self._handle_progress_failed)
    
    def _handle_progress_started(self, event: ProgressStartedEvent) -> None:
        """Handle progress started events."""
        if self.observer:
            self.observer.on_operation_start(event.operation_id, event.operation_name, event.total)
    
    def _handle_progress_updated(self, event: ProgressUpdatedEvent) -> None:
        """Handle progress updated events."""
        if self.observer:
            update = ProgressUpdate(
                operation_id=event.operation_id,
                operation_name=event.operation_name,
                current=event.current,
                total=event.total,
                status=ProgressStatus.IN_PROGRESS,
                message=event.message,
                metadata=event.metadata
            )
            self.observer.on_progress_update(update)
    
    def _handle_progress_completed(self, event: ProgressCompletedEvent) -> None:
        """Handle progress completed events."""
        if self.observer:
            self.observer.on_operation_complete(event.operation_id, event.operation_name)
    
    def _handle_progress_failed(self, event: ProgressFailedEvent) -> None:
        """Handle progress failed events."""
        if self.observer:
            self.observer.on_operation_failed(event.operation_id, event.operation_name, event.error)
    
    def unsubscribe(self) -> None:
        """Unsubscribe from all progress events. Useful for cleanup."""
        self.event_bus.unsubscribe("progress_started", self._handle_progress_started)
        self.event_bus.unsubscribe("progress_updated", self._handle_progress_updated)
        self.event_bus.unsubscribe("progress_completed", self._handle_progress_completed)
        self.event_bus.unsubscribe("progress_failed", self._handle_progress_failed)


class EventPublishingProgressTracker:
    """
    A progress tracker that publishes progress events instead of directly notifying observers.
    
    This can be used as a drop-in replacement for the original ProgressTracker,
    but instead of directly calling observer methods, it publishes events that
    EventDrivenProgressTracker can listen to.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, project_id: Optional[str] = None):
        """
        Initialize the event-publishing progress tracker.
        
        Args:
            event_bus: The event bus to publish to (defaults to global event bus)
            project_id: Optional project ID to include in events
        """
        self.event_bus = event_bus or get_event_bus()
        self.project_id = project_id
        self._operations = {}
    
    def start_operation(self, operation_id: str, operation_name: str, total: int) -> None:
        """Start tracking a new operation by publishing a ProgressStartedEvent."""
        self._operations[operation_id] = {
            'name': operation_name,
            'current': 0,
            'total': total,
            'status': ProgressStatus.STARTED
        }
        
        event = ProgressStartedEvent(
            event_id="",
            timestamp=None,
            event_type="",
            operation_id=operation_id,
            operation_name=operation_name,
            total=total,
            project_id=self.project_id
        )
        self.event_bus.publish(event)
    
    def update_progress(self, operation_id: str, current: int, message: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        """Update progress for an operation by publishing a ProgressUpdatedEvent."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['current'] = current
        op['status'] = ProgressStatus.IN_PROGRESS
        
        event = ProgressUpdatedEvent(
            event_id="",
            timestamp=None,
            event_type="",
            operation_id=operation_id,
            operation_name=op['name'],
            current=current,
            total=op['total'],
            message=message,
            metadata=metadata,
            project_id=self.project_id
        )
        self.event_bus.publish(event)
    
    def complete_operation(self, operation_id: str) -> None:
        """Mark an operation as completed by publishing a ProgressCompletedEvent."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['status'] = ProgressStatus.COMPLETED
        op['current'] = op['total']
        
        event = ProgressCompletedEvent(
            event_id="",
            timestamp=None,
            event_type="",
            operation_id=operation_id,
            operation_name=op['name'],
            project_id=self.project_id
        )
        self.event_bus.publish(event)
        del self._operations[operation_id]
    
    def fail_operation(self, operation_id: str, error: str) -> None:
        """Mark an operation as failed by publishing a ProgressFailedEvent."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['status'] = ProgressStatus.FAILED
        
        event = ProgressFailedEvent(
            event_id="",
            timestamp=None,
            event_type="",
            operation_id=operation_id,
            operation_name=op['name'],
            error=error,
            project_id=self.project_id
        )
        self.event_bus.publish(event)
        del self._operations[operation_id]


def create_cli_progress_tracker(console: Console, event_bus: Optional[EventBus] = None) -> EventDrivenProgressTracker:
    """
    Convenience function to create an event-driven progress tracker with CLI observer.
    
    Args:
        console: Rich console for displaying progress
        event_bus: Optional event bus (defaults to global event bus)
    
    Returns:
        EventDrivenProgressTracker configured with CLIProgressObserver
    """
    cli_observer = CLIProgressObserver(console)
    return EventDrivenProgressTracker(observer=cli_observer, event_bus=event_bus)