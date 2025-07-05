"""
Tests for the KnowledgeStore interface and implementations.

This module tests the storage-agnostic KnowledgeStore interface and its
concrete implementations, ensuring they follow CRAFT principles.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

from context_mixer.domain.knowledge_store import KnowledgeStore, StorageError, KnowledgeStoreFactory
from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore
from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    ProvenanceInfo,
    SearchQuery,
    SearchResults,
    SearchResult,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)


@pytest.fixture
def sample_provenance():
    """Create sample provenance information."""
    return ProvenanceInfo(
        source="test_source",
        created_at="2024-01-01T00:00:00Z",
        author="test_author"
    )


@pytest.fixture
def sample_metadata(sample_provenance):
    """Create sample chunk metadata."""
    return ChunkMetadata(
        domains=["technical", "frontend"],
        authority=AuthorityLevel.OFFICIAL,
        scope=["enterprise"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.CURRENT,
        tags=["react", "javascript"],
        provenance=sample_provenance
    )


@pytest.fixture
def sample_chunk(sample_metadata):
    """Create a sample knowledge chunk."""
    return KnowledgeChunk(
        id="test-chunk-1",
        content="Use React hooks for state management in functional components.",
        metadata=sample_metadata
    )


@pytest.fixture
def conflicting_chunk(sample_provenance):
    """Create a chunk that conflicts with the sample chunk."""
    conflicting_metadata = ChunkMetadata(
        domains=["technical", "frontend"],
        authority=AuthorityLevel.DEPRECATED,
        scope=["enterprise"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.DEPRECATED,
        tags=["react", "javascript"],
        provenance=sample_provenance
    )
    return KnowledgeChunk(
        id="test-chunk-conflict",
        content="Use React class components for state management.",
        metadata=conflicting_metadata
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
async def vector_store(temp_db_path):
    """Create a VectorKnowledgeStore instance for testing."""
    store = VectorKnowledgeStore(temp_db_path)
    yield store
    # Cleanup
    try:
        await store.reset()
    except Exception:
        pass


class DescribeKnowledgeStoreFactory:
    """Test the KnowledgeStore factory."""

    def should_create_vector_store(self, temp_db_path):
        store = KnowledgeStoreFactory.create_vector_store(temp_db_path)

        assert isinstance(store, VectorKnowledgeStore)
        assert store.db_path == temp_db_path


class DescribeVectorKnowledgeStore:
    """Test the VectorKnowledgeStore implementation."""

    @pytest.mark.asyncio
    async def should_store_and_retrieve_chunks(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Retrieve the chunk
        retrieved = await vector_store.get_chunk(sample_chunk.id)

        assert retrieved is not None
        assert retrieved.id == sample_chunk.id
        assert retrieved.content == sample_chunk.content
        assert retrieved.metadata.domains == sample_chunk.metadata.domains

    @pytest.mark.asyncio
    async def should_return_none_for_nonexistent_chunk(self, vector_store):
        result = await vector_store.get_chunk("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def should_search_chunks_by_content(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Search for it
        query = SearchQuery(text="React hooks", max_results=10)
        results = await vector_store.search(query)

        assert len(results.results) > 0
        assert any(result.chunk.id == sample_chunk.id for result in results.results)

    @pytest.mark.asyncio
    async def should_filter_chunks_by_domain(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Search with domain filter
        query = SearchQuery(
            text="React",
            domains=["technical"],
            max_results=10
        )
        results = await vector_store.search(query)

        assert len(results.results) > 0
        for result in results.results:
            assert "technical" in result.chunk.metadata.domains

    @pytest.mark.asyncio
    async def should_filter_chunks_by_authority(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Search with authority filter
        query = SearchQuery(
            text="React",
            authority_levels=[AuthorityLevel.OFFICIAL],
            max_results=10
        )
        results = await vector_store.search(query)

        assert len(results.results) > 0
        for result in results.results:
            assert result.chunk.metadata.authority == AuthorityLevel.OFFICIAL

    @pytest.mark.asyncio
    async def should_get_chunks_by_domain(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Get chunks by domain
        chunks = await vector_store.get_chunks_by_domain(["technical"])

        assert len(chunks) > 0
        assert any(chunk.id == sample_chunk.id for chunk in chunks)

    @pytest.mark.asyncio
    async def should_get_chunks_by_authority(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Get chunks by authority
        chunks = await vector_store.get_chunks_by_authority([AuthorityLevel.OFFICIAL])

        assert len(chunks) > 0
        assert any(chunk.id == sample_chunk.id for chunk in chunks)

    @pytest.mark.asyncio
    async def should_delete_chunks(self, vector_store, sample_chunk):
        # Store the chunk
        await vector_store.store_chunks([sample_chunk])

        # Verify it exists
        retrieved = await vector_store.get_chunk(sample_chunk.id)
        assert retrieved is not None

        # Delete it
        deleted = await vector_store.delete_chunk(sample_chunk.id)
        assert deleted is True

        # Verify it's gone
        retrieved = await vector_store.get_chunk(sample_chunk.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def should_return_false_when_deleting_nonexistent_chunk(self, vector_store):
        deleted = await vector_store.delete_chunk("nonexistent-id")

        assert deleted is False

    @pytest.mark.asyncio
    async def should_detect_conflicts(self, vector_store, sample_chunk, conflicting_chunk):
        # Store both chunks
        await vector_store.store_chunks([sample_chunk, conflicting_chunk])

        # Detect conflicts for the sample chunk
        conflicts = await vector_store.detect_conflicts(sample_chunk)

        # Should find the conflicting chunk
        assert len(conflicts) > 0
        assert any(chunk.id == conflicting_chunk.id for chunk in conflicts)

    @pytest.mark.asyncio
    async def should_validate_dependencies(self, vector_store, sample_provenance):
        # Create a chunk with dependencies
        dependent_metadata = ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            dependencies=["dependency-1", "dependency-2"],
            provenance=sample_provenance
        )
        dependent_chunk = KnowledgeChunk(
            id="dependent-chunk",
            content="This chunk depends on others.",
            metadata=dependent_metadata
        )

        # Store the dependent chunk (but not its dependencies)
        await vector_store.store_chunks([dependent_chunk])

        # Validate dependencies
        missing = await vector_store.validate_dependencies(dependent_chunk)

        # Should find missing dependencies
        assert len(missing) == 2
        assert "dependency-1" in missing
        assert "dependency-2" in missing

    @pytest.mark.asyncio
    async def should_return_empty_list_when_all_dependencies_exist(self, vector_store, sample_chunk, sample_provenance):
        # Create dependency chunk
        dep_chunk = KnowledgeChunk(
            id="dependency-1",
            content="Dependency content",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        # Create dependent chunk
        dependent_metadata = ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            dependencies=["dependency-1"],
            provenance=sample_provenance
        )
        dependent_chunk = KnowledgeChunk(
            id="dependent-chunk",
            content="This chunk depends on others.",
            metadata=dependent_metadata
        )

        # Store both chunks
        await vector_store.store_chunks([dep_chunk, dependent_chunk])

        # Validate dependencies
        missing = await vector_store.validate_dependencies(dependent_chunk)

        # Should find no missing dependencies
        assert len(missing) == 0

    @pytest.mark.asyncio
    async def should_get_store_statistics(self, vector_store, sample_chunk):
        # Initially empty
        stats = await vector_store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["storage_type"] == "vector"
        assert stats["backend"] == "chromadb"

        # Store a chunk
        await vector_store.store_chunks([sample_chunk])

        # Check stats again
        stats = await vector_store.get_stats()
        assert stats["total_chunks"] == 1

    @pytest.mark.asyncio
    async def should_reset_store(self, vector_store, sample_chunk):
        # Store a chunk
        await vector_store.store_chunks([sample_chunk])

        # Verify it exists
        stats = await vector_store.get_stats()
        assert stats["total_chunks"] == 1

        # Reset the store
        await vector_store.reset()

        # Verify it's empty
        stats = await vector_store.get_stats()
        assert stats["total_chunks"] == 0

    @pytest.mark.asyncio
    async def should_handle_storage_errors_gracefully(self, vector_store):
        # Test with invalid chunk data that might cause storage errors
        invalid_chunk = KnowledgeChunk(
            id="",  # Invalid empty ID
            content="",
            metadata=ChunkMetadata(
                domains=[],
                authority=AuthorityLevel.OFFICIAL,
                scope=[],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=ProvenanceInfo(
                    source="test",
                    created_at="invalid-date"
                )
            )
        )

        # Should handle gracefully and raise StorageError
        with pytest.raises(StorageError):
            await vector_store.store_chunks([invalid_chunk])


class DescribeConflictDetection:
    """Test conflict detection logic."""

    def should_detect_authority_conflicts(self, sample_provenance):
        from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore

        store = VectorKnowledgeStore(Path("/tmp"))

        chunk1 = KnowledgeChunk(
            id="chunk1",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        chunk2 = KnowledgeChunk(
            id="chunk2",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.EXPERIMENTAL,  # Different authority
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        assert store._is_potential_conflict(chunk1, chunk2) is True

    def should_detect_temporal_conflicts(self, sample_provenance):
        from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore

        store = VectorKnowledgeStore(Path("/tmp"))

        chunk1 = KnowledgeChunk(
            id="chunk1",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        chunk2 = KnowledgeChunk(
            id="chunk2",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.DEPRECATED,  # Deprecated
                provenance=sample_provenance
            )
        )

        assert store._is_potential_conflict(chunk1, chunk2) is True

    def should_not_detect_conflicts_in_different_domains(self, sample_provenance):
        from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore

        store = VectorKnowledgeStore(Path("/tmp"))

        chunk1 = KnowledgeChunk(
            id="chunk1",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],  # Technical domain
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        chunk2 = KnowledgeChunk(
            id="chunk2",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["business"],  # Business domain
                authority=AuthorityLevel.EXPERIMENTAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        assert store._is_potential_conflict(chunk1, chunk2) is False

    def should_detect_explicit_conflicts(self, sample_provenance):
        from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore

        store = VectorKnowledgeStore(Path("/tmp"))

        chunk1 = KnowledgeChunk(
            id="chunk1",
            content="Use React hooks",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                conflicts=["chunk2"],  # Explicit conflict
                provenance=sample_provenance
            )
        )

        chunk2 = KnowledgeChunk(
            id="chunk2",
            content="Use React class components",
            metadata=ChunkMetadata(
                domains=["technical"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
        )

        assert store._is_potential_conflict(chunk1, chunk2) is True
