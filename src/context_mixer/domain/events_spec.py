import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
import asyncio

from context_mixer.domain.events import (
    Event, EventBus, ChunksIngestedEvent, ConflictDetectedEvent, ConflictResolvedEvent,
    get_event_bus, EventHandler, AsyncEventHandler
)


@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for each test."""
    return EventBus()


@pytest.fixture
def sample_chunks_ingested_event():
    """Create a sample ChunksIngestedEvent for testing."""
    return ChunksIngestedEvent(
        event_id="test-event-123",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        event_type="chunks_ingested",
        project_id="test-project",
        project_name="Test Project",
        chunk_count=5,
        file_paths=["file1.py", "file2.py"],
        processing_time_seconds=1.5
    )


@pytest.fixture
def sample_conflict_detected_event():
    """Create a sample ConflictDetectedEvent for testing."""
    return ConflictDetectedEvent(
        event_id="conflict-event-456",
        timestamp=datetime(2024, 1, 1, 12, 5, 0),
        event_type="conflict_detected",
        project_id="test-project",
        conflict_count=2,
        conflict_types=["duplicate_guidance", "conflicting_rules"],
        affected_files=["config.py", "rules.py"]
    )


@pytest.fixture
def sample_conflict_resolved_event():
    """Create a sample ConflictResolvedEvent for testing."""
    return ConflictResolvedEvent(
        event_id="resolved-event-789",
        timestamp=datetime(2024, 1, 1, 12, 10, 0),
        event_type="conflict_resolved",
        project_id="test-project",
        resolved_conflict_count=2,
        resolution_strategy="llm_based",
        auto_resolved_count=1,
        manually_resolved_count=1
    )


class DescribeEvent:
    def should_auto_generate_event_id_when_not_provided(self):
        event = ChunksIngestedEvent(
            event_id="",
            timestamp=datetime.utcnow(),
            event_type="chunks_ingested",
            project_id="test",
            project_name="Test",
            chunk_count=1,
            file_paths=[],
            processing_time_seconds=1.0
        )

        assert event.event_id != ""
        assert len(event.event_id) > 0

    def should_auto_generate_timestamp_when_not_provided(self):
        before = datetime.utcnow()
        event = ChunksIngestedEvent(
            event_id="test-id",
            timestamp=None,
            event_type="chunks_ingested",
            project_id="test",
            project_name="Test",
            chunk_count=1,
            file_paths=[],
            processing_time_seconds=1.0
        )
        after = datetime.utcnow()

        assert before <= event.timestamp <= after

    def should_preserve_provided_event_id_and_timestamp(self, sample_chunks_ingested_event):
        assert sample_chunks_ingested_event.event_id == "test-event-123"
        assert sample_chunks_ingested_event.timestamp == datetime(2024, 1, 1, 12, 0, 0)


class DescribeChunksIngestedEvent:
    def should_set_correct_event_type(self):
        event = ChunksIngestedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            project_id="test",
            project_name="Test",
            chunk_count=1,
            file_paths=[],
            processing_time_seconds=1.0
        )

        assert event.event_type == "chunks_ingested"

    def should_contain_all_required_data(self, sample_chunks_ingested_event):
        event = sample_chunks_ingested_event

        assert event.project_id == "test-project"
        assert event.project_name == "Test Project"
        assert event.chunk_count == 5
        assert event.file_paths == ["file1.py", "file2.py"]
        assert event.processing_time_seconds == 1.5


class DescribeConflictDetectedEvent:
    def should_set_correct_event_type(self):
        event = ConflictDetectedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            project_id="test",
            conflict_count=1,
            conflict_types=["test"],
            affected_files=["test.py"]
        )

        assert event.event_type == "conflict_detected"

    def should_contain_all_required_data(self, sample_conflict_detected_event):
        event = sample_conflict_detected_event

        assert event.project_id == "test-project"
        assert event.conflict_count == 2
        assert event.conflict_types == ["duplicate_guidance", "conflicting_rules"]
        assert event.affected_files == ["config.py", "rules.py"]


class DescribeConflictResolvedEvent:
    def should_set_correct_event_type(self):
        event = ConflictResolvedEvent(
            event_id="test",
            timestamp=datetime.utcnow(),
            event_type="",
            project_id="test",
            resolved_conflict_count=1,
            resolution_strategy="test",
            auto_resolved_count=0,
            manually_resolved_count=1
        )

        assert event.event_type == "conflict_resolved"

    def should_contain_all_required_data(self, sample_conflict_resolved_event):
        event = sample_conflict_resolved_event

        assert event.project_id == "test-project"
        assert event.resolved_conflict_count == 2
        assert event.resolution_strategy == "llm_based"
        assert event.auto_resolved_count == 1
        assert event.manually_resolved_count == 1


class DescribeEventBus:
    def should_initialize_with_empty_handlers(self, event_bus):
        assert event_bus.get_handler_count() == 0
        assert event_bus.get_handler_count("chunks_ingested") == 0

    def should_subscribe_and_publish_to_specific_event_type(self, event_bus, sample_chunks_ingested_event):
        handler_called = False
        received_event = None

        def test_handler(event):
            nonlocal handler_called, received_event
            handler_called = True
            received_event = event

        event_bus.subscribe("chunks_ingested", test_handler)
        assert event_bus.get_handler_count("chunks_ingested") == 1

        event_bus.publish(sample_chunks_ingested_event)

        assert handler_called
        assert received_event == sample_chunks_ingested_event

    def should_not_call_handler_for_different_event_type(self, event_bus, sample_chunks_ingested_event):
        handler_called = False

        def test_handler(event):
            nonlocal handler_called
            handler_called = True

        event_bus.subscribe("conflict_detected", test_handler)
        event_bus.publish(sample_chunks_ingested_event)

        assert not handler_called

    def should_call_multiple_handlers_for_same_event_type(self, event_bus, sample_chunks_ingested_event):
        handler1_called = False
        handler2_called = False

        def handler1(event):
            nonlocal handler1_called
            handler1_called = True

        def handler2(event):
            nonlocal handler2_called
            handler2_called = True

        event_bus.subscribe("chunks_ingested", handler1)
        event_bus.subscribe("chunks_ingested", handler2)

        event_bus.publish(sample_chunks_ingested_event)

        assert handler1_called
        assert handler2_called

    def should_call_global_handlers_for_any_event(self, event_bus, sample_chunks_ingested_event, sample_conflict_detected_event):
        global_handler_calls = []

        def global_handler(event):
            global_handler_calls.append(event.event_type)

        event_bus.subscribe_global(global_handler)

        event_bus.publish(sample_chunks_ingested_event)
        event_bus.publish(sample_conflict_detected_event)

        assert "chunks_ingested" in global_handler_calls
        assert "conflict_detected" in global_handler_calls
        assert len(global_handler_calls) == 2

    def should_handle_exceptions_in_handlers_gracefully(self, event_bus, sample_chunks_ingested_event):
        good_handler_called = False

        def failing_handler(event):
            raise Exception("Handler failed")

        def good_handler(event):
            nonlocal good_handler_called
            good_handler_called = True

        event_bus.subscribe("chunks_ingested", failing_handler)
        event_bus.subscribe("chunks_ingested", good_handler)

        # Should not raise exception
        event_bus.publish(sample_chunks_ingested_event)

        # Good handler should still be called
        assert good_handler_called

    def should_unsubscribe_handlers(self, event_bus, sample_chunks_ingested_event):
        handler_called = False

        def test_handler(event):
            nonlocal handler_called
            handler_called = True

        event_bus.subscribe("chunks_ingested", test_handler)
        event_bus.unsubscribe("chunks_ingested", test_handler)

        event_bus.publish(sample_chunks_ingested_event)

        assert not handler_called
        assert event_bus.get_handler_count("chunks_ingested") == 0

    def should_clear_all_handlers(self, event_bus):
        def dummy_handler(event):
            pass

        async def dummy_async_handler(event):
            pass

        event_bus.subscribe("test_event", dummy_handler)
        event_bus.subscribe_async("test_event", dummy_async_handler)
        event_bus.subscribe_global(dummy_handler)
        event_bus.subscribe_global_async(dummy_async_handler)

        assert event_bus.get_handler_count() > 0

        event_bus.clear_handlers()

        assert event_bus.get_handler_count() == 0


class DescribeEventBusAsync:
    async def should_subscribe_and_publish_async_handlers(self, event_bus, sample_chunks_ingested_event):
        handler_called = False
        received_event = None

        async def async_handler(event):
            nonlocal handler_called, received_event
            handler_called = True
            received_event = event

        event_bus.subscribe_async("chunks_ingested", async_handler)

        await event_bus.publish_async(sample_chunks_ingested_event)

        assert handler_called
        assert received_event == sample_chunks_ingested_event

    async def should_call_multiple_async_handlers_concurrently(self, event_bus, sample_chunks_ingested_event):
        handler1_called = False
        handler2_called = False
        call_order = []

        async def handler1(event):
            nonlocal handler1_called
            await asyncio.sleep(0.01)  # Small delay
            call_order.append("handler1")
            handler1_called = True

        async def handler2(event):
            nonlocal handler2_called
            await asyncio.sleep(0.005)  # Smaller delay
            call_order.append("handler2")
            handler2_called = True

        event_bus.subscribe_async("chunks_ingested", handler1)
        event_bus.subscribe_async("chunks_ingested", handler2)

        await event_bus.publish_async(sample_chunks_ingested_event)

        assert handler1_called
        assert handler2_called
        # handler2 should complete first due to shorter delay
        assert call_order == ["handler2", "handler1"]

    async def should_call_global_async_handlers(self, event_bus, sample_chunks_ingested_event):
        global_handler_called = False

        async def global_async_handler(event):
            nonlocal global_handler_called
            global_handler_called = True

        event_bus.subscribe_global_async(global_async_handler)

        await event_bus.publish_async(sample_chunks_ingested_event)

        assert global_handler_called

    async def should_handle_exceptions_in_async_handlers_gracefully(self, event_bus, sample_chunks_ingested_event):
        good_handler_called = False

        async def failing_async_handler(event):
            raise Exception("Async handler failed")

        async def good_async_handler(event):
            nonlocal good_handler_called
            good_handler_called = True

        event_bus.subscribe_async("chunks_ingested", failing_async_handler)
        event_bus.subscribe_async("chunks_ingested", good_async_handler)

        # Should not raise exception
        await event_bus.publish_async(sample_chunks_ingested_event)

        # Good handler should still be called
        assert good_handler_called

    async def should_unsubscribe_async_handlers(self, event_bus, sample_chunks_ingested_event):
        handler_called = False

        async def async_handler(event):
            nonlocal handler_called
            handler_called = True

        event_bus.subscribe_async("chunks_ingested", async_handler)
        event_bus.unsubscribe_async("chunks_ingested", async_handler)

        await event_bus.publish_async(sample_chunks_ingested_event)

        assert not handler_called


class DescribeGlobalEventBus:
    def should_return_same_instance(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def should_be_usable_across_modules(self, sample_chunks_ingested_event):
        # This simulates using the global event bus from different modules
        global_bus = get_event_bus()
        handler_called = False

        def test_handler(event):
            nonlocal handler_called
            handler_called = True

        # Clear any existing handlers from other tests
        global_bus.clear_handlers()

        global_bus.subscribe("chunks_ingested", test_handler)
        global_bus.publish(sample_chunks_ingested_event)

        assert handler_called

        # Clean up for other tests
        global_bus.clear_handlers()


class DescribeEventBusIntegration:
    async def should_support_mixed_sync_and_async_handlers(self, event_bus, sample_chunks_ingested_event):
        sync_called = False
        async_called = False

        def sync_handler(event):
            nonlocal sync_called
            sync_called = True

        async def async_handler(event):
            nonlocal async_called
            async_called = True

        event_bus.subscribe("chunks_ingested", sync_handler)
        event_bus.subscribe_async("chunks_ingested", async_handler)

        # Test sync publish (should only call sync handlers)
        event_bus.publish(sample_chunks_ingested_event)
        assert sync_called
        assert not async_called

        # Reset
        sync_called = False
        async_called = False

        # Test async publish (should call async handlers)
        await event_bus.publish_async(sample_chunks_ingested_event)
        assert not sync_called  # Sync handlers not called in async publish
        assert async_called

    def should_count_handlers_correctly(self, event_bus):
        def sync_handler(event):
            pass

        async def async_handler(event):
            pass

        # Add various handlers
        event_bus.subscribe("event1", sync_handler)
        event_bus.subscribe("event1", sync_handler)  # Same handler, different subscription
        event_bus.subscribe_async("event1", async_handler)
        event_bus.subscribe("event2", sync_handler)
        event_bus.subscribe_global(sync_handler)
        event_bus.subscribe_global_async(async_handler)

        assert event_bus.get_handler_count("event1") == 3  # 2 sync + 1 async
        assert event_bus.get_handler_count("event2") == 1
        assert event_bus.get_handler_count("nonexistent") == 0
        assert event_bus.get_handler_count() == 6  # Total: 4 specific + 2 global
