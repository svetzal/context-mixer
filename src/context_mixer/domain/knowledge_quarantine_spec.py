"""
Tests for the Knowledge Quarantine System.

This module contains comprehensive tests for the quarantine system functionality,
including quarantining chunks, reviewing quarantined items, and resolving conflicts.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from context_mixer.domain.knowledge_quarantine import (
    KnowledgeQuarantine,
    QuarantinedChunk,
    QuarantineReason,
    ResolutionAction,
    Resolution
)
from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    ProvenanceInfo,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)


@pytest.fixture
def sample_provenance():
    """Create a sample provenance info for testing."""
    return ProvenanceInfo(
        source="test_file.py",
        project_id="test-project",
        project_name="Test Project",
        project_path="/path/to/test/project",
        created_at=datetime.utcnow().isoformat(),
        author="test_user"
    )


@pytest.fixture
def sample_metadata(sample_provenance):
    """Create sample chunk metadata for testing."""
    return ChunkMetadata(
        domains=["technical"],
        authority=AuthorityLevel.CONVENTIONAL,
        scope=["enterprise"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.CURRENT,
        tags=["python", "testing"],
        provenance=sample_provenance
    )


@pytest.fixture
def sample_chunk(sample_metadata):
    """Create a sample knowledge chunk for testing."""
    return KnowledgeChunk(
        id="test-chunk-1",
        content="This is a test knowledge chunk for quarantine testing.",
        metadata=sample_metadata
    )


@pytest.fixture
def conflicting_chunk(sample_metadata):
    """Create a conflicting knowledge chunk for testing."""
    conflicting_metadata = sample_metadata.model_copy()
    conflicting_metadata.authority = AuthorityLevel.OFFICIAL

    return KnowledgeChunk(
        id="conflicting-chunk-1",
        content="This is a conflicting knowledge chunk with higher authority.",
        metadata=conflicting_metadata
    )


@pytest.fixture
def quarantine_system():
    """Create a knowledge quarantine system for testing."""
    return KnowledgeQuarantine()


class DescribeKnowledgeQuarantine:

    def should_initialize_empty_quarantine_system(self, quarantine_system):
        stats = quarantine_system.get_quarantine_stats()

        assert stats["total_quarantined"] == 0
        assert stats["resolved"] == 0
        assert stats["unresolved"] == 0
        assert stats["reason_breakdown"] == {}
        assert stats["priority_breakdown"] == {}

    def should_quarantine_chunk_with_semantic_conflict(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            chunk=sample_chunk,
            reason=QuarantineReason.SEMANTIC_CONFLICT,
            description="Conflicts with existing knowledge about testing practices",
            conflicting_chunks=["existing-chunk-1"],
            quarantined_by="test_user",
            priority=1
        )

        assert quarantine_id is not None
        quarantined = quarantine_system.get_quarantined_chunk(quarantine_id)
        assert quarantined is not None
        assert quarantined.chunk.id == sample_chunk.id
        assert quarantined.reason == QuarantineReason.SEMANTIC_CONFLICT
        assert quarantined.description == "Conflicts with existing knowledge about testing practices"
        assert quarantined.conflicting_chunks == ["existing-chunk-1"]
        assert quarantined.quarantined_by == "test_user"
        assert quarantined.priority == 1
        assert not quarantined.is_resolved()

    def should_quarantine_chunk_with_authority_conflict(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            chunk=sample_chunk,
            reason=QuarantineReason.AUTHORITY_CONFLICT,
            description="Lower authority conflicts with official guidance",
            conflicting_chunks=["official-chunk-1"],
            priority=2
        )

        quarantined = quarantine_system.get_quarantined_chunk(quarantine_id)
        assert quarantined.reason == QuarantineReason.AUTHORITY_CONFLICT
        assert quarantined.priority == 2

    def should_review_all_quarantined_chunks(self, quarantine_system, sample_chunk, conflicting_chunk):
        # Quarantine multiple chunks
        id1 = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test conflict 1", priority=1
        )
        id2 = quarantine_system.quarantine_chunk(
            conflicting_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Test conflict 2", priority=2
        )

        all_quarantined = quarantine_system.review_quarantined_chunks()

        assert len(all_quarantined) == 2
        # Should be sorted by priority then by quarantine date
        assert all_quarantined[0].priority == 1
        assert all_quarantined[1].priority == 2

    def should_filter_quarantined_chunks_by_reason(self, quarantine_system, sample_chunk, conflicting_chunk):
        quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Semantic conflict"
        )
        quarantine_system.quarantine_chunk(
            conflicting_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Authority conflict"
        )

        semantic_conflicts = quarantine_system.review_quarantined_chunks(
            reason_filter=QuarantineReason.SEMANTIC_CONFLICT
        )
        authority_conflicts = quarantine_system.review_quarantined_chunks(
            reason_filter=QuarantineReason.AUTHORITY_CONFLICT
        )

        assert len(semantic_conflicts) == 1
        assert len(authority_conflicts) == 1
        assert semantic_conflicts[0].reason == QuarantineReason.SEMANTIC_CONFLICT
        assert authority_conflicts[0].reason == QuarantineReason.AUTHORITY_CONFLICT

    def should_filter_quarantined_chunks_by_resolution_status(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test conflict"
        )

        # Initially unresolved
        unresolved = quarantine_system.review_quarantined_chunks(resolved_filter=False)
        resolved = quarantine_system.review_quarantined_chunks(resolved_filter=True)

        assert len(unresolved) == 1
        assert len(resolved) == 0

        # Resolve the quarantine
        resolution = Resolution(
            action=ResolutionAction.ACCEPT,
            reason="Accepted after review",
            resolved_by="test_user"
        )
        quarantine_system.resolve_quarantine(quarantine_id, resolution)

        # Now should be resolved
        unresolved = quarantine_system.review_quarantined_chunks(resolved_filter=False)
        resolved = quarantine_system.review_quarantined_chunks(resolved_filter=True)

        assert len(unresolved) == 0
        assert len(resolved) == 1

    def should_filter_quarantined_chunks_by_priority(self, quarantine_system, sample_chunk, conflicting_chunk):
        quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "High priority", priority=1
        )
        quarantine_system.quarantine_chunk(
            conflicting_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Low priority", priority=3
        )

        high_priority = quarantine_system.review_quarantined_chunks(priority_filter=1)
        low_priority = quarantine_system.review_quarantined_chunks(priority_filter=3)

        assert len(high_priority) == 1
        assert len(low_priority) == 1
        assert high_priority[0].priority == 1
        assert low_priority[0].priority == 3

    def should_filter_quarantined_chunks_by_project(self, quarantine_system, sample_chunk):
        # Create chunk with different project
        different_provenance = ProvenanceInfo(
            source="other_file.py",
            project_id="other-project",
            project_name="Other Project",
            project_path="/path/to/other/project",
            created_at=datetime.utcnow().isoformat()
        )
        different_metadata = ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.CONVENTIONAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=different_provenance
        )
        different_chunk = KnowledgeChunk(
            id="different-chunk",
            content="Different project chunk",
            metadata=different_metadata
        )

        quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test project chunk"
        )
        quarantine_system.quarantine_chunk(
            different_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Other project chunk"
        )

        test_project_chunks = quarantine_system.review_quarantined_chunks(
            project_filter="test-project"
        )
        other_project_chunks = quarantine_system.review_quarantined_chunks(
            project_filter="other-project"
        )

        assert len(test_project_chunks) == 1
        assert len(other_project_chunks) == 1
        assert test_project_chunks[0].chunk.get_project_id() == "test-project"
        assert other_project_chunks[0].chunk.get_project_id() == "other-project"

    def should_resolve_quarantine_with_accept_action(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test conflict"
        )

        resolution = Resolution(
            action=ResolutionAction.ACCEPT,
            reason="Reviewed and accepted as valid knowledge",
            resolved_by="reviewer",
            notes="Conflicts were minor and acceptable"
        )

        success = quarantine_system.resolve_quarantine(quarantine_id, resolution)

        assert success is True
        quarantined = quarantine_system.get_quarantined_chunk(quarantine_id)
        assert quarantined.is_resolved()
        assert quarantined.resolution.action == ResolutionAction.ACCEPT
        assert quarantined.resolution.reason == "Reviewed and accepted as valid knowledge"
        assert quarantined.resolution.resolved_by == "reviewer"
        assert quarantined.resolution.notes == "Conflicts were minor and acceptable"

    def should_resolve_quarantine_with_modify_action(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test conflict"
        )

        # Create modified chunk
        modified_chunk = sample_chunk.model_copy()
        modified_chunk.content = "Modified content to resolve conflicts"

        resolution = Resolution(
            action=ResolutionAction.MODIFY,
            reason="Modified to resolve semantic conflicts",
            resolved_by="reviewer",
            modified_chunk=modified_chunk
        )

        success = quarantine_system.resolve_quarantine(quarantine_id, resolution)

        assert success is True
        quarantined = quarantine_system.get_quarantined_chunk(quarantine_id)
        assert quarantined.resolution.action == ResolutionAction.MODIFY
        assert quarantined.resolution.modified_chunk is not None
        assert quarantined.resolution.modified_chunk.content == "Modified content to resolve conflicts"

    def should_fail_to_resolve_nonexistent_quarantine(self, quarantine_system):
        resolution = Resolution(
            action=ResolutionAction.ACCEPT,
            reason="Test resolution"
        )

        success = quarantine_system.resolve_quarantine("nonexistent-id", resolution)

        assert success is False

    def should_remove_quarantined_chunk(self, quarantine_system, sample_chunk):
        quarantine_id = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Test conflict"
        )

        # Verify chunk exists
        assert quarantine_system.get_quarantined_chunk(quarantine_id) is not None

        # Remove chunk
        success = quarantine_system.remove_quarantined_chunk(quarantine_id)

        assert success is True
        assert quarantine_system.get_quarantined_chunk(quarantine_id) is None

    def should_fail_to_remove_nonexistent_chunk(self, quarantine_system):
        success = quarantine_system.remove_quarantined_chunk("nonexistent-id")
        assert success is False

    def should_provide_comprehensive_quarantine_stats(self, quarantine_system, sample_chunk, conflicting_chunk):
        # Add various quarantined chunks
        id1 = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Conflict 1", priority=1
        )
        quarantine_system.quarantine_chunk(
            conflicting_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Conflict 2", priority=2
        )

        # Resolve one
        resolution = Resolution(action=ResolutionAction.ACCEPT, reason="Accepted")
        quarantine_system.resolve_quarantine(id1, resolution)

        stats = quarantine_system.get_quarantine_stats()

        assert stats["total_quarantined"] == 2
        assert stats["resolved"] == 1
        assert stats["unresolved"] == 1
        assert stats["reason_breakdown"]["semantic_conflict"] == 1
        assert stats["reason_breakdown"]["authority_conflict"] == 1
        assert stats["priority_breakdown"][2] == 1  # Only unresolved chunk has priority 2

    def should_get_high_priority_unresolved_chunks(self, quarantine_system, sample_chunk, conflicting_chunk):
        # Create chunks with different priorities
        high_priority_chunk = sample_chunk.model_copy()
        high_priority_chunk.id = "high-priority"

        medium_priority_chunk = conflicting_chunk.model_copy()
        medium_priority_chunk.id = "medium-priority"

        low_priority_chunk = sample_chunk.model_copy()
        low_priority_chunk.id = "low-priority"

        quarantine_system.quarantine_chunk(
            high_priority_chunk, QuarantineReason.SEMANTIC_CONFLICT, "High priority", priority=1
        )
        quarantine_system.quarantine_chunk(
            medium_priority_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Medium priority", priority=2
        )
        quarantine_system.quarantine_chunk(
            low_priority_chunk, QuarantineReason.VALIDATION_FAILURE, "Low priority", priority=4
        )

        high_priority_unresolved = quarantine_system.get_high_priority_unresolved()

        assert len(high_priority_unresolved) == 2  # Priority 1 and 2
        assert all(chunk.priority <= 2 for chunk in high_priority_unresolved)
        assert all(not chunk.is_resolved() for chunk in high_priority_unresolved)

    def should_clear_resolved_chunks(self, quarantine_system, sample_chunk, conflicting_chunk):
        # Add and resolve some chunks
        id1 = quarantine_system.quarantine_chunk(
            sample_chunk, QuarantineReason.SEMANTIC_CONFLICT, "Conflict 1"
        )
        id2 = quarantine_system.quarantine_chunk(
            conflicting_chunk, QuarantineReason.AUTHORITY_CONFLICT, "Conflict 2"
        )

        # Resolve one
        resolution = Resolution(action=ResolutionAction.ACCEPT, reason="Accepted")
        quarantine_system.resolve_quarantine(id1, resolution)

        # Clear resolved chunks
        cleared_count = quarantine_system.clear_resolved_chunks()

        assert cleared_count == 1
        assert quarantine_system.get_quarantined_chunk(id1) is None  # Resolved chunk removed
        assert quarantine_system.get_quarantined_chunk(id2) is not None  # Unresolved chunk remains

        stats = quarantine_system.get_quarantine_stats()
        assert stats["total_quarantined"] == 1
        assert stats["resolved"] == 0
        assert stats["unresolved"] == 1


class DescribeQuarantinedChunk:

    def should_calculate_age_correctly(self, sample_chunk, sample_metadata):
        # Create a quarantined chunk with a specific timestamp
        past_time = datetime.utcnow() - timedelta(days=5)
        quarantined_chunk = QuarantinedChunk(
            chunk=sample_chunk,
            reason=QuarantineReason.SEMANTIC_CONFLICT,
            description="Test quarantine",
            quarantined_at=past_time.isoformat()
        )

        age = quarantined_chunk.get_age_days()
        assert age == 5

    def should_detect_resolved_status(self, sample_chunk):
        quarantined_chunk = QuarantinedChunk(
            chunk=sample_chunk,
            reason=QuarantineReason.SEMANTIC_CONFLICT,
            description="Test quarantine"
        )

        # Initially unresolved
        assert not quarantined_chunk.is_resolved()

        # Add resolution
        resolution = Resolution(
            action=ResolutionAction.ACCEPT,
            reason="Accepted after review"
        )
        quarantined_chunk.resolution = resolution

        # Now resolved
        assert quarantined_chunk.is_resolved()


class DescribeResolution:

    def should_create_resolution_with_default_timestamp(self):
        resolution = Resolution(
            action=ResolutionAction.ACCEPT,
            reason="Test resolution"
        )

        assert resolution.action == ResolutionAction.ACCEPT
        assert resolution.reason == "Test resolution"
        assert resolution.resolved_at is not None
        # Should be recent (within last minute)
        resolved_time = datetime.fromisoformat(resolution.resolved_at)
        now = datetime.utcnow()
        assert (now - resolved_time).total_seconds() < 60

    def should_store_modified_chunk_for_modify_action(self, sample_chunk):
        modified_chunk = sample_chunk.model_copy()
        modified_chunk.content = "Modified content"

        resolution = Resolution(
            action=ResolutionAction.MODIFY,
            reason="Modified to resolve conflicts",
            modified_chunk=modified_chunk,
            notes="Changed content to be more specific"
        )

        assert resolution.action == ResolutionAction.MODIFY
        assert resolution.modified_chunk is not None
        assert resolution.modified_chunk.content == "Modified content"
        assert resolution.notes == "Changed content to be more specific"
