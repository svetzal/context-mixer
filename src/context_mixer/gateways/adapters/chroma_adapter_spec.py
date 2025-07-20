"""
Tests for the ChromaAdapter.

This test suite focuses specifically on testing the adapter logic that converts
between domain objects and ChromaDB format, following the principle of testing
what we own (our adapter logic) rather than what we don't own (ChromaDB).
"""

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
from context_mixer.gateways.adapters.chroma_adapter import ChromaAdapter


@pytest.fixture
def adapter():
    """Create a ChromaAdapter instance for testing."""
    return ChromaAdapter()


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
        embedding=[0.1, 0.2, 0.3]
    )


class DescribeChromaAdapter:
    """Test suite for ChromaAdapter conversion logic."""

    def should_convert_chunks_to_chroma_format(self, adapter, sample_chunk):
        """ChromaAdapter should convert KnowledgeChunk objects to ChromaDB format."""
        chunks = [sample_chunk]

        result = adapter.chunks_to_chroma_format(chunks)

        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "embeddings" in result

        assert result["ids"] == [sample_chunk.id]
        assert result["documents"] == [sample_chunk.content]
        assert result["embeddings"] == [sample_chunk.embedding]

        # Check metadata conversion
        metadata = result["metadatas"][0]
        assert metadata["domains"] == "technical,frontend"  # Lists are converted to comma-separated strings
        assert metadata["authority"] == "official"
        assert metadata["granularity"] == "detailed"
        assert metadata["temporal"] == "current"

    def should_handle_chunks_without_embeddings(self, adapter, sample_chunk):
        """ChromaAdapter should handle chunks without embeddings."""
        sample_chunk.embedding = None
        chunks = [sample_chunk]

        result = adapter.chunks_to_chroma_format(chunks)

        assert result["embeddings"] is None

    def should_convert_chroma_results_to_search_results(self, adapter):
        """ChromaAdapter should convert ChromaDB results to SearchResults."""
        query = SearchQuery(text="React hooks", max_results=5)

        chroma_results = {
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

        result = adapter.chroma_results_to_search_results(chroma_results, query)

        assert isinstance(result, SearchResults)
        assert result.query == query
        assert len(result.results) == 1

        search_result = result.results[0]
        assert search_result.chunk.id == "test-chunk-1"
        assert search_result.chunk.content == "Use React hooks for state management"
        assert search_result.relevance_score == 0.8  # 1.0 - 0.2 distance

    def should_convert_search_query_to_chroma_params(self, adapter):
        """ChromaAdapter should convert SearchQuery to ChromaDB parameters."""
        query = SearchQuery(
            text="React hooks",
            domains=["technical"],
            max_results=10
        )

        result = adapter.search_query_to_chroma_params(query)

        assert "n_results" in result
        assert result["n_results"] == 10

        # Should include domain filtering in where clause
        if "where" in result:
            assert "domains" in str(result["where"])

    def should_filter_results_by_relevance_threshold(self, adapter):
        """ChromaAdapter should filter out results below relevance threshold."""
        query = SearchQuery(text="React hooks", min_relevance_score=0.5)

        chroma_results = {
            'ids': [['chunk-1', 'chunk-2']],
            'documents': [['High relevance content', 'Low relevance content']],
            'metadatas': [[
                {
                    'domains': ['technical'],
                    'authority': 'official',
                    'scope': ['production'],
                    'granularity': 'detailed',
                    'temporal': 'current',
                    'tags': [],
                    'provenance_source': 'test',
                    'provenance_created_at': '2024-01-01T00:00:00Z',
                    'provenance_author': 'test'
                },
                {
                    'domains': ['technical'],
                    'authority': 'official',
                    'scope': ['production'],
                    'granularity': 'detailed',
                    'temporal': 'current',
                    'tags': [],
                    'provenance_source': 'test',
                    'provenance_created_at': '2024-01-01T00:00:00Z',
                    'provenance_author': 'test'
                }
            ]],
            'distances': [[0.3, 0.8]]  # 0.7 and 0.2 relevance scores
        }

        result = adapter.chroma_results_to_search_results(chroma_results, query)

        # Should only include the first result (0.7 relevance > 0.5 threshold)
        # Second result (0.2 relevance < 0.5 threshold) should be filtered out
        assert len(result.results) == 1
        assert result.results[0].chunk.id == "chunk-1"
