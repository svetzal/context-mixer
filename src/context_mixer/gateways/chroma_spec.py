"""
Tests for the ChromaGateway with domain objects.

This test suite verifies that the ChromaGateway properly uses domain objects
and hides ChromaDB implementation details following CRAFT principles.

These tests focus on testing our adapter logic by mocking ChromaDB operations,
following the principle of "don't test what you don't own".
"""

from pathlib import Path

import pytest

from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResults,
    ChunkMetadata,
    ProvenanceInfo,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)
from context_mixer.gateways.chroma import ChromaGateway


@pytest.fixture
def mock_chroma_client(mocker):
    """Create a mocked ChromaDB client."""
    return mocker.MagicMock()


@pytest.fixture
def mock_collection(mocker):
    """Create a mocked ChromaDB collection."""
    collection = mocker.MagicMock()
    collection.name = "knowledge"
    collection.count.return_value = 0
    return collection


@pytest.fixture
def sample_provenance():
    """Create sample provenance info for testing."""
    return ProvenanceInfo(
        source="test_source",
        created_at="2024-01-01T00:00:00Z",
        author="test_author"
    )


@pytest.fixture
def sample_metadata(sample_provenance):
    """Create sample chunk metadata for testing."""
    return ChunkMetadata(
        domains=["technical", "frontend"],
        authority=AuthorityLevel.OFFICIAL,
        scope=["production"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.CURRENT,
        tags=["react", "components"],
        provenance=sample_provenance
    )


@pytest.fixture
def sample_chunk(sample_metadata):
    """Create a sample knowledge chunk for testing."""
    return KnowledgeChunk(
        id="test-chunk-1",
        content="Use React hooks for state management in functional components.",
        metadata=sample_metadata,
        embedding=None  # Let ChromaDB generate embeddings for consistency
    )


@pytest.fixture
def gateway(mock_chroma_client, mock_collection, mocker):
    """Create a ChromaGateway instance with mocked ChromaDB components."""
    # Mock the chromadb.PersistentClient constructor to return our mock client
    mocker.patch('chromadb.PersistentClient', return_value=mock_chroma_client)

    # Configure the mock client to return our mock collection
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    # Now create the gateway - it will use our mocked client
    gateway = ChromaGateway(Path("/tmp/test"))

    return gateway


class DescribeChromaGateway:
    """Test suite for ChromaGateway domain object interface."""

    def should_initialize_with_domain_adapter(self, gateway):
        """ChromaGateway should initialize with a ChromaAdapter."""
        assert gateway.adapter is not None
        assert hasattr(gateway.adapter, 'chunks_to_chroma_format')
        assert hasattr(gateway.adapter, 'chroma_results_to_search_results')

    def should_store_knowledge_chunks_using_domain_objects(self, gateway, sample_chunk, mock_collection):
        """ChromaGateway should accept and store KnowledgeChunk domain objects."""
        # This should not raise an exception
        gateway.store_knowledge_chunks([sample_chunk])

        # Verify that the adapter converted the chunks to ChromaDB format
        # and the collection.upsert was called
        mock_collection.upsert.assert_called_once()

        # Verify the adapter was used to convert domain objects
        call_args = mock_collection.upsert.call_args
        assert 'ids' in call_args.kwargs
        assert 'documents' in call_args.kwargs
        assert 'metadatas' in call_args.kwargs
        assert sample_chunk.id in call_args.kwargs['ids']
        assert sample_chunk.content in call_args.kwargs['documents']

    def should_search_using_domain_query_objects(self, gateway, sample_chunk, mock_collection):
        """ChromaGateway should accept SearchQuery domain objects and return SearchResults."""
        # Configure mock to return sample search results
        mock_chroma_results = {
            'ids': [['test-chunk-1']],
            'documents': [['Use React hooks for state management']],
            'metadatas': [[{
                'domains': ['technical', 'frontend'],
                'authority': 'official',
                'scope': ['production'],
                'granularity': 'detailed',
                'temporal': 'current',
                'tags': ['react', 'components'],
                'provenance_source': 'test_source',
                'provenance_created_at': '2024-01-01T00:00:00Z',
                'provenance_author': 'test_author'
            }]],
            'distances': [[0.2]]
        }
        mock_collection.query.return_value = mock_chroma_results

        # Create a search query
        query = SearchQuery(
            text="React hooks",
            domains=["technical"],
            max_results=5
        )

        # Search should return SearchResults domain object
        results = gateway.search_knowledge(query)
        assert isinstance(results, SearchResults)
        assert results.query == query
        assert len(results.results) >= 0

        # Verify the collection.query was called with proper parameters
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert 'query_texts' in call_args.kwargs

    def should_return_none_for_nonexistent_chunk(self, gateway, mock_collection):
        """ChromaGateway should return None for chunks that don't exist."""
        # Configure mock to return empty results for nonexistent chunk
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        result = gateway.get_knowledge_chunk("nonexistent-id")
        assert result is None

        # Verify the collection.get was called with the correct ID
        mock_collection.get.assert_called_once_with(ids=["nonexistent-id"], include=["documents", "metadatas", "embeddings"])

    def should_delete_existing_chunks(self, gateway, sample_chunk, mock_collection):
        """ChromaGateway should be able to delete existing chunks."""
        # Configure mock to return existing chunk when checking if it exists
        mock_collection.get.return_value = {"ids": [sample_chunk.id]}

        # Delete it
        deleted = gateway.delete_knowledge_chunk(sample_chunk.id)
        assert deleted is True

        # Verify the collection.get was called to check existence
        mock_collection.get.assert_called_with(ids=[sample_chunk.id])
        # Verify the collection.delete was called
        mock_collection.delete.assert_called_once_with(ids=[sample_chunk.id])

    def should_return_false_when_deleting_nonexistent_chunk(self, gateway, mock_collection):
        """ChromaGateway should return False when trying to delete a nonexistent chunk."""
        # Configure mock to return empty results (chunk doesn't exist)
        mock_collection.get.return_value = {"ids": []}

        deleted = gateway.delete_knowledge_chunk("nonexistent-id")
        assert deleted is False

        # Verify the collection.get was called to check existence
        mock_collection.get.assert_called_once_with(ids=["nonexistent-id"])
        # Verify delete was NOT called since chunk doesn't exist
        mock_collection.delete.assert_not_called()

    def should_provide_collection_statistics(self, gateway, sample_chunk, mock_collection):
        """ChromaGateway should provide statistics about the knowledge collection."""
        # Configure mock to return count of 0 initially
        mock_collection.count.return_value = 0

        # Initially empty
        stats = gateway.get_collection_stats()
        assert stats["total_chunks"] == 0
        assert "collection_name" in stats
        assert stats["collection_name"] == "knowledge"

        # Verify the collection.count was called
        mock_collection.count.assert_called()

    def should_handle_empty_chunk_list(self, gateway):
        """ChromaGateway should handle empty chunk lists gracefully."""
        # This should not raise an exception
        gateway.store_knowledge_chunks([])

    def should_reset_knowledge_store(self, gateway, sample_chunk, mock_chroma_client):
        """ChromaGateway should be able to reset the entire knowledge store."""
        # Reset the store
        gateway.reset_knowledge_store()

        # Verify the client.reset was called
        mock_chroma_client.reset.assert_called_once()

    def should_filter_search_results_by_domain(self, gateway, sample_metadata, mock_collection):
        """ChromaGateway should filter search results by domain."""
        # Configure mock to return only frontend chunks when searching with frontend domain filter
        mock_chroma_results = {
            'ids': [['chunk-1']],
            'documents': [['Frontend React component patterns']],
            'metadatas': [[{
                'domains': ['technical', 'frontend'],
                'authority': 'official',
                'scope': ['production'],
                'granularity': 'detailed',
                'temporal': 'current',
                'tags': [],
                'provenance_source': 'test_source',
                'provenance_created_at': '2024-01-01T00:00:00Z',
                'provenance_author': 'test_author'
            }]],
            'distances': [[0.1]]
        }
        mock_collection.query.return_value = mock_chroma_results

        # Search with domain filter
        query = SearchQuery(
            text="patterns",
            domains=["frontend"],
            max_results=10
        )

        results = gateway.search_knowledge(query)

        # Verify the adapter correctly converted the search query to include domain filtering
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args

        # Should only return frontend chunks (as configured in mock)
        for result in results.results:
            assert "frontend" in result.chunk.metadata.domains

    def should_not_expose_chromadb_implementation_details(self, gateway, sample_chunk, mock_collection):
        """ChromaGateway should not expose ChromaDB-specific structures in its interface."""
        # Configure mock to return sample search results
        mock_chroma_results = {
            'ids': [['test-chunk-1']],
            'documents': [['Use React hooks for state management']],
            'metadatas': [[{
                'domains': ['technical', 'frontend'],
                'authority': 'official',
                'scope': ['production'],
                'granularity': 'detailed',
                'temporal': 'current',
                'tags': ['react', 'components'],
                'provenance_source': 'test_source',
                'provenance_created_at': '2024-01-01T00:00:00Z',
                'provenance_author': 'test_author'
            }]],
            'distances': [[0.2]]
        }
        mock_collection.query.return_value = mock_chroma_results

        # Search for it
        query = SearchQuery(text="React hooks", max_results=5)
        results = gateway.search_knowledge(query)

        # Verify all returned objects are domain objects, not ChromaDB objects
        assert isinstance(results, SearchResults)
        for result in results.results:
            assert hasattr(result, 'chunk')
            assert hasattr(result, 'relevance_score')
            assert isinstance(result.chunk, KnowledgeChunk)

        # Verify no ChromaDB-specific attributes are exposed
        assert not hasattr(results, 'ids')
        assert not hasattr(results, 'documents')
        assert not hasattr(results, 'metadatas')
        assert not hasattr(results, 'distances')
