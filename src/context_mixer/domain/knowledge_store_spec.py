"""
Tests for the KnowledgeStore interface and implementations.

This module tests the storage-agnostic KnowledgeStore interface and its
concrete implementations, ensuring they follow CRAFT principles.
"""

import tempfile
from pathlib import Path

import pytest

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
from context_mixer.domain.knowledge_store import StorageError, KnowledgeStoreFactory
from context_mixer.domain.vector_knowledge_store import VectorKnowledgeStore


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
async def vector_store(temp_db_path, mocker):
    """Create a VectorKnowledgeStore instance for testing with mocked gateway."""
    # Create a mock ChromaGateway with realistic behavior
    mock_gateway = mocker.MagicMock()

    # Create an in-memory storage for the mock
    stored_chunks = {}

    def mock_store_chunks(chunks):
        from context_mixer.domain.knowledge_store import StorageError
        for chunk in chunks:
            # Simulate storage errors for invalid chunks
            if not chunk.id or not chunk.id.strip():
                raise StorageError("Cannot store chunk with empty ID")
            if not chunk.content and not chunk.content.strip():
                # Allow empty content but raise error if both ID and content are empty
                if not chunk.id or not chunk.id.strip():
                    raise StorageError("Cannot store chunk with empty ID and content")
            stored_chunks[chunk.id] = chunk

    def mock_get_chunk(chunk_id):
        return stored_chunks.get(chunk_id)

    def mock_search_knowledge(query):
        # Mock search that simulates semantic similarity for conflict detection
        results = []
        for chunk in stored_chunks.values():
            relevance_score = 0.0

            if query.text == "*":
                # Match all chunks for wildcard queries
                relevance_score = 0.8
            elif query.text.lower() in chunk.content.lower():
                # Direct text match
                relevance_score = 0.9
            elif _is_semantically_similar(query.text, chunk.content):
                # Simulate semantic similarity for React state management content
                relevance_score = 0.8

            # Only include if relevance meets minimum threshold
            if relevance_score >= query.min_relevance_score:
                # Apply domain filter if specified
                if query.domains and not any(domain in chunk.metadata.domains for domain in query.domains):
                    continue
                # Apply authority filter if specified
                if query.authority_levels and chunk.metadata.authority not in query.authority_levels:
                    continue
                # Create SearchResult with relevance score
                search_result = SearchResult(chunk=chunk, relevance_score=relevance_score)
                results.append(search_result)
                if len(results) >= query.max_results:
                    break
        return SearchResults(query=query, results=results, total_found=len(results))

    def _is_semantically_similar(text1, text2):
        # Simple semantic similarity simulation for test cases
        # Both texts are about React state management
        react_keywords = ["react", "state management", "hooks", "class components", "functional components"]
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check if both texts contain React-related keywords
        text1_has_react = any(keyword in text1_lower for keyword in react_keywords)
        text2_has_react = any(keyword in text2_lower for keyword in react_keywords)

        return text1_has_react and text2_has_react

    def mock_delete_chunk(chunk_id):
        if chunk_id in stored_chunks:
            del stored_chunks[chunk_id]
            return True
        return False

    def mock_get_chunks_by_domain(domains):
        return [chunk for chunk in stored_chunks.values() 
                if any(domain in chunk.metadata.domains for domain in domains)]

    def mock_get_chunks_by_authority(authority_levels):
        return [chunk for chunk in stored_chunks.values() 
                if chunk.metadata.authority in authority_levels]

    def mock_get_stats():
        return {
            "total_chunks": len(stored_chunks),
            "total_embeddings": len(stored_chunks),
            "collection_name": "test_collection",
            "domains": list(set(domain for chunk in stored_chunks.values() 
                               for domain in chunk.metadata.domains)),
            "authority_levels": list(set(chunk.metadata.authority.value 
                                       for chunk in stored_chunks.values()))
        }

    def mock_reset():
        stored_chunks.clear()

    # Configure the mock methods
    mock_gateway.store_knowledge_chunks.side_effect = mock_store_chunks
    mock_gateway.get_knowledge_chunk.side_effect = mock_get_chunk
    mock_gateway.search_knowledge.side_effect = mock_search_knowledge
    mock_gateway.delete_knowledge_chunk.side_effect = mock_delete_chunk
    mock_gateway.get_chunks_by_domain.side_effect = mock_get_chunks_by_domain
    mock_gateway.get_chunks_by_authority.side_effect = mock_get_chunks_by_authority
    mock_gateway.get_collection_stats.side_effect = mock_get_stats
    mock_gateway.reset_knowledge_store.side_effect = mock_reset

    # Create the store and patch its _get_gateway method
    store = VectorKnowledgeStore(temp_db_path)
    mocker.patch.object(store, '_get_gateway', return_value=mock_gateway)

    # Store the mock gateway as an attribute for test access
    store._mock_gateway = mock_gateway

    yield store
    # No cleanup needed since we're using mocks


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
