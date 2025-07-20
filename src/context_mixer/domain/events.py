import asyncio
import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Event(ABC):
    """
    Base class for all domain events.

    Events represent something that has happened in the domain and are immutable.
    They carry the data necessary for event handlers to process the event.
    """
    event_id: str
    timestamp: datetime
    event_type: str

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(UTC)


@dataclass
class ChunksIngestedEvent(Event):
    """
    Event published when chunks have been successfully ingested into the knowledge store.
    """
    project_id: str
    project_name: str
    chunk_count: int
    file_paths: List[str]
    processing_time_seconds: float

    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "chunks_ingested"
        super().__post_init__()


@dataclass
class ConflictDetectedEvent(Event):
    """
    Event published when conflicts are detected during ingestion or processing.
    """
    project_id: str
    conflict_count: int
    conflict_types: List[str]
    affected_files: List[str]

    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "conflict_detected"
        super().__post_init__()


@dataclass
class ConflictResolvedEvent(Event):
    """
    Event published when conflicts have been resolved.
    """
    project_id: str
    resolved_conflict_count: int
    resolution_strategy: str
    auto_resolved_count: int
    manually_resolved_count: int

    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "conflict_resolved"
        super().__post_init__()


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can return None or awaitable


class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.

    Provides loose coupling between components by allowing publishers to emit events
    without knowing about subscribers, and subscribers to listen for events without
    knowing about publishers.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._async_handlers: Dict[str, List[AsyncEventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._global_async_handlers: List[AsyncEventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a synchronous handler to a specific event type.

        Args:
            event_type: The type of event to listen for
            handler: The handler function to call when the event is published
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")

    def subscribe_async(self, event_type: str, handler: AsyncEventHandler) -> None:
        """
        Subscribe an asynchronous handler to a specific event type.

        Args:
            event_type: The type of event to listen for
            handler: The async handler function to call when the event is published
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)
        logger.debug(f"Subscribed async handler to event type: {event_type}")

    def subscribe_global(self, handler: EventHandler) -> None:
        """
        Subscribe a synchronous handler to all events.

        Args:
            handler: The handler function to call for any published event
        """
        self._global_handlers.append(handler)
        logger.debug("Subscribed global handler")

    def subscribe_global_async(self, handler: AsyncEventHandler) -> None:
        """
        Subscribe an asynchronous handler to all events.

        Args:
            handler: The async handler function to call for any published event
        """
        self._global_async_handlers.append(handler)
        logger.debug("Subscribed global async handler")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from a specific event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from event type: {event_type}")

    def unsubscribe_async(self, event_type: str, handler: AsyncEventHandler) -> None:
        """
        Unsubscribe an async handler from a specific event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The async handler to remove
        """
        if event_type in self._async_handlers and handler in self._async_handlers[event_type]:
            self._async_handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed async handler from event type: {event_type}")

    def publish(self, event: Event) -> None:
        """
        Publish an event synchronously to all registered handlers.

        Args:
            event: The event to publish
        """
        logger.debug(f"Publishing event: {event.event_type} (ID: {event.event_id})")

        # Call global handlers first
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")

        # Call specific event type handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type}: {e}")

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously to all registered handlers.

        Args:
            event: The event to publish
        """
        logger.debug(f"Publishing event async: {event.event_type} (ID: {event.event_id})")

        # Collect all handlers to call
        handlers_to_call = []

        # Add global async handlers
        for handler in self._global_async_handlers:
            handlers_to_call.append(handler(event))

        # Add specific event type async handlers
        if event.event_type in self._async_handlers:
            for handler in self._async_handlers[event.event_type]:
                handlers_to_call.append(handler(event))

        # Execute all handlers concurrently
        if handlers_to_call:
            results = await asyncio.gather(*handlers_to_call, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in async event handler: {result}")

    def clear_handlers(self) -> None:
        """
        Clear all registered handlers. Useful for testing.
        """
        self._handlers.clear()
        self._async_handlers.clear()
        self._global_handlers.clear()
        self._global_async_handlers.clear()
        logger.debug("Cleared all event handlers")

    def get_handler_count(self, event_type: Optional[str] = None) -> int:
        """
        Get the number of handlers registered for a specific event type or all handlers.

        Args:
            event_type: The event type to count handlers for, or None for all handlers

        Returns:
            The number of registered handlers
        """
        if event_type is None:
            total = len(self._global_handlers) + len(self._global_async_handlers)
            for handlers in self._handlers.values():
                total += len(handlers)
            for handlers in self._async_handlers.values():
                total += len(handlers)
            return total
        else:
            count = 0
            if event_type in self._handlers:
                count += len(self._handlers[event_type])
            if event_type in self._async_handlers:
                count += len(self._async_handlers[event_type])
            return count


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.

    Returns:
        The global EventBus instance
    """
    return _event_bus
