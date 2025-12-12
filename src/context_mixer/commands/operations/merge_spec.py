"""
Tests for the merge operations.
"""

import pytest

from context_mixer.commands.operations.merge import format_conflict_resolutions, \
    detect_conflicts_async, detect_conflicts_batch, detect_conflicts
from context_mixer.domain.conflict import Conflict, ConflictingGuidance, ConflictList
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, AuthorityLevel, \
    ProvenanceInfo, GranularityLevel, TemporalScope
from context_mixer.gateways.llm import MockLLMGateway


class DescribeFormatConflictResolutions:
    """Test the format_conflict_resolutions pure function."""

    @pytest.fixture
    def sample_conflicts(self):
        """Create sample conflicts for testing."""
        return [
            Conflict(
                description="Indentation conflict between 4 spaces and 2 spaces",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces for indentation", source="existing"),
                ],
                resolution="Use 4 spaces for indentation consistently across the codebase"
            ),
            Conflict(
                description="Variable naming convention conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use camelCase for variables", source="existing"),
                ],
                resolution="Use camelCase for variables in JavaScript, snake_case in Python"
            )
        ]

    def should_return_empty_string_when_no_conflicts(self):
        """Test that function returns empty string when no conflicts provided."""
        result = format_conflict_resolutions(None)
        assert result == ""

        result = format_conflict_resolutions([])
        assert result == ""

    def should_handle_conflicts_with_resolution_none(self):
        """Test that conflicts with resolution=None are handled as 'not a conflict'."""
        conflicts_with_resolution_none = [
            Conflict(
                description="Not actually a conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces", source="existing"),
                    ConflictingGuidance(content="Use 2 spaces", source="new"),
                ],
                resolution=None
            )
        ]

        result = format_conflict_resolutions(conflicts_with_resolution_none)
        assert result != ""
        assert "Not actually a conflict" in result
        assert "This is not a conflict - both pieces of guidance are acceptable" in result
        assert "Use 4 spaces (from existing)" in result
        assert "Use 2 spaces (from new)" in result

    def should_format_single_conflict_with_resolution(self, sample_conflicts):
        """Test formatting of a single conflict with resolution."""
        single_conflict = [sample_conflicts[0]]

        result = format_conflict_resolutions(single_conflict)

        assert "Conflicts were detected between these documents and resolved as follows:" in result
        assert "Description: Indentation conflict between 4 spaces and 2 spaces" in result
        assert "Resolution: Use 4 spaces for indentation consistently across the codebase" in result
        assert "When merging the documents, use these resolutions to guide your work." in result

    def should_format_multiple_conflicts_with_resolutions(self, sample_conflicts):
        """Test formatting of multiple conflicts with resolutions."""
        result = format_conflict_resolutions(sample_conflicts)

        assert "Conflicts were detected between these documents and resolved as follows:" in result
        assert "Description: Indentation conflict between 4 spaces and 2 spaces" in result
        assert "Resolution: Use 4 spaces for indentation consistently across the codebase" in result
        assert "Description: Variable naming convention conflict" in result
        assert "Resolution: Use camelCase for variables in JavaScript, snake_case in Python" in result
        assert "When merging the documents, use these resolutions to guide your work." in result

    def should_handle_mixed_conflicts_with_and_without_resolutions(self):
        """Test that both resolved conflicts and conflicts with resolution=None are handled."""
        mixed_conflicts = [
            Conflict(
                description="Resolved conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use tabs", source="existing"),
                ],
                resolution="Use spaces for indentation"
            ),
            Conflict(
                description="Not a conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use semicolons", source="existing"),
                    ConflictingGuidance(content="No semicolons", source="new"),
                ],
                resolution=None  # This is not a conflict
            )
        ]

        result = format_conflict_resolutions(mixed_conflicts)

        # Should include the resolved conflict
        assert "Description: Resolved conflict" in result
        assert "Resolution: Use spaces for indentation" in result

        # Should also include the "not a conflict" resolution
        assert "Not a conflict" in result
        assert "This is not a conflict - both pieces of guidance are acceptable" in result
        assert "Use semicolons (from existing)" in result
        assert "No semicolons (from new)" in result

    def should_handle_empty_resolution_strings(self):
        """Test that conflicts with empty resolution strings are skipped."""
        conflicts_with_empty_resolution = [
            Conflict(
                description="Conflict with empty resolution",
                conflicting_guidance=[
                    ConflictingGuidance(content="Option A", source="existing"),
                ],
                resolution=""  # Empty string
            )
        ]

        result = format_conflict_resolutions(conflicts_with_empty_resolution)
        assert result == ""

    def should_preserve_newlines_and_formatting_in_descriptions_and_resolutions(self):
        """Test that newlines and formatting in descriptions and resolutions are preserved."""
        conflict_with_formatting = [
            Conflict(
                description="Multi-line\nconflict description",
                conflicting_guidance=[
                    ConflictingGuidance(content="Option A", source="existing"),
                ],
                resolution="Multi-line\nresolution with\nformatting"
            )
        ]

        result = format_conflict_resolutions(conflict_with_formatting)

        assert "Description: Multi-line\nconflict description" in result
        assert "Resolution: Multi-line\nresolution with\nformatting" in result


class DescribeDetectConflictsAsync:
    """Test the detect_conflicts_async function."""

    @pytest.fixture
    def mock_llm_gateway(self):
        """Create a mock LLM gateway for testing."""
        return MockLLMGateway(responses={
            'generate_object': ConflictList(list=[])
        })

    @pytest.fixture
    def mock_llm_gateway_with_conflicts(self):
        """Create a mock LLM gateway that returns conflicts."""
        conflict = Conflict(
            description="Test conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Content A", source="existing"),
            ]
        )
        return MockLLMGateway(responses={
            'generate_object': ConflictList(list=[conflict])
        })

    @pytest.mark.asyncio
    async def should_return_empty_conflict_list_when_no_conflicts(self, mock_llm_gateway):
        """Test that async function returns empty conflict list when no conflicts detected."""
        result = await detect_conflicts_async("content1", "content2", mock_llm_gateway)

        assert isinstance(result, ConflictList)
        assert len(result.list) == 0
        assert mock_llm_gateway.was_called('generate_object')

    @pytest.mark.asyncio
    async def should_return_conflicts_when_detected(self, mock_llm_gateway_with_conflicts):
        """Test that async function returns conflicts when detected."""
        result = await detect_conflicts_async("content1", "content2", mock_llm_gateway_with_conflicts)

        assert isinstance(result, ConflictList)
        assert len(result.list) == 1
        assert result.list[0].description == "Test conflict"
        assert mock_llm_gateway_with_conflicts.was_called('generate_object')


class DescribeDetectConflictsBatch:
    """Test the detect_conflicts_batch function."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample knowledge chunks for testing."""
        provenance1 = ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )
        provenance2 = ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )

        metadata1 = ChunkMetadata(
            domains=["test"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["test"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["chunk1"],
            provenance=provenance1
        )
        metadata2 = ChunkMetadata(
            domains=["test"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["test"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["chunk2"],
            provenance=provenance2
        )

        chunk1 = KnowledgeChunk(
            id="chunk1",
            content="Content for chunk 1",
            metadata=metadata1
        )
        chunk2 = KnowledgeChunk(
            id="chunk2",
            content="Content for chunk 2",
            metadata=metadata2
        )

        return [chunk1, chunk2]

    @pytest.fixture
    def mock_llm_gateway(self):
        """Create a mock LLM gateway for testing."""
        return MockLLMGateway(responses={
            'generate_object': ConflictList(list=[])
        })

    @pytest.fixture
    def mock_llm_gateway_with_conflicts(self):
        """Create a mock LLM gateway that returns conflicts."""
        conflict = Conflict(
            description="Batch test conflict",
            conflicting_guidance=[
                ConflictingGuidance(content="Content A", source="existing"),
            ]
        )
        return MockLLMGateway(responses={
            'generate_object': ConflictList(list=[conflict])
        })

    @pytest.mark.asyncio
    async def should_process_empty_chunk_pairs_list(self, mock_llm_gateway):
        """Test that batch function handles empty chunk pairs list."""
        result = await detect_conflicts_batch([], mock_llm_gateway)

        assert isinstance(result, list)
        assert len(result) == 0
        assert not mock_llm_gateway.was_called('generate_object')

    @pytest.mark.asyncio
    async def should_process_single_chunk_pair(self, sample_chunks, mock_llm_gateway):
        """Test that batch function processes single chunk pair correctly."""
        chunk_pairs = [(sample_chunks[0], sample_chunks[1])]

        result = await detect_conflicts_batch(chunk_pairs, mock_llm_gateway)

        assert isinstance(result, list)
        assert len(result) == 1

        chunk1, chunk2, conflicts = result[0]
        assert chunk1.id == "chunk1"
        assert chunk2.id == "chunk2"
        assert isinstance(conflicts, ConflictList)
        assert len(conflicts.list) == 0
        assert mock_llm_gateway.was_called('generate_object')

    @pytest.mark.asyncio
    async def should_process_multiple_chunk_pairs_in_batches(self, sample_chunks, mock_llm_gateway):
        """Test that batch function processes multiple chunk pairs correctly."""
        # Create additional chunks for testing
        provenance3 = ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )
        metadata3 = ChunkMetadata(
            domains=["test"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["test"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["chunk3"],
            provenance=provenance3
        )
        chunk3 = KnowledgeChunk(
            id="chunk3",
            content="Content for chunk 3",
            metadata=metadata3
        )

        chunk_pairs = [
            (sample_chunks[0], sample_chunks[1]),
            (sample_chunks[0], chunk3),
            (sample_chunks[1], chunk3)
        ]

        result = await detect_conflicts_batch(chunk_pairs, mock_llm_gateway, batch_size=2)

        assert isinstance(result, list)
        assert len(result) == 3

        # Verify all pairs were processed
        processed_pairs = [(r[0].id, r[1].id) for r in result]
        expected_pairs = [("chunk1", "chunk2"), ("chunk1", "chunk3"), ("chunk2", "chunk3")]
        assert processed_pairs == expected_pairs

        # Verify LLM was called for each pair
        assert mock_llm_gateway.get_call_count('generate_object') == 3

    @pytest.mark.asyncio
    async def should_return_conflicts_when_detected_in_batch(self, sample_chunks, mock_llm_gateway_with_conflicts):
        """Test that batch function returns conflicts when detected."""
        chunk_pairs = [(sample_chunks[0], sample_chunks[1])]

        result = await detect_conflicts_batch(chunk_pairs, mock_llm_gateway_with_conflicts)

        assert isinstance(result, list)
        assert len(result) == 1

        chunk1, chunk2, conflicts = result[0]
        assert isinstance(conflicts, ConflictList)
        assert len(conflicts.list) == 1
        assert conflicts.list[0].description == "Batch test conflict"

    @pytest.mark.asyncio
    async def should_handle_exceptions_gracefully(self, sample_chunks):
        """Test that batch function handles exceptions gracefully."""
        # Create a mock that raises an exception
        failing_mock = MockLLMGateway()

        # Override the generate_object method to raise an exception
        def failing_generate_object(*args, **kwargs):
            raise Exception("LLM service unavailable")

        failing_mock.generate_object = failing_generate_object

        chunk_pairs = [(sample_chunks[0], sample_chunks[1])]

        result = await detect_conflicts_batch(chunk_pairs, failing_mock)

        assert isinstance(result, list)
        assert len(result) == 1

        chunk1, chunk2, conflicts = result[0]
        assert chunk1.id == "chunk1"
        assert chunk2.id == "chunk2"
        assert isinstance(conflicts, ConflictList)
        assert len(conflicts.list) == 0  # Should return empty conflict list on error


class DescribeDetectConflictsArchitecturalScope:
    """Test architectural scope awareness in conflict detection."""

    @pytest.fixture
    def mock_llm_gateway_no_conflicts(self):
        """Create a mock LLM gateway that returns no conflicts."""
        return MockLLMGateway(responses={
            'generate_object': ConflictList(list=[])
        })

    def should_not_detect_conflicts_between_gateway_and_general_testing_rules(self, mock_llm_gateway_no_conflicts):
        """
        Regression test: Gateway-specific testing rules should not conflict with general testing rules.

        This test ensures that the improved prompt correctly identifies that:
        - "Gateways should not be tested" (component-specific rule)
        - "Write tests for new functionality" (general rule)
        are NOT conflicting because they apply to different architectural scopes.
        """
        gateway_content = """Use the Gateway pattern to isolate I/O (network, disk, etc.) from business logic.

- Gateways should contain minimal to no logic, simply delegate to the OS or libraries that perform I/O.
- Gateways should not be tested, to avoid mocking systems you don't own."""

        general_testing_content = """- Follow the existing project structure.
- Write tests for any new functionality and co-locate them.
- Document using numpy-style docstrings.
- Keep code complexity low (max complexity 10).
- Use type hints for all functions and methods.
- Use pydantic v2 (not dataclasses) for data objects.
- Favor list/dict comprehensions over explicit loops.
- Favor declarative styles over imperative."""

        conflicts = detect_conflicts(gateway_content, general_testing_content, mock_llm_gateway_no_conflicts)

        assert isinstance(conflicts, ConflictList)
        assert len(conflicts.list) == 0, "Gateway-specific and general testing rules should not be detected as conflicting"
        assert mock_llm_gateway_no_conflicts.was_called('generate_object')

    def should_not_detect_conflicts_between_component_specific_rules(self, mock_llm_gateway_no_conflicts):
        """
        Test that component-specific rules for different components don't conflict.
        """
        repository_content = """Repository pattern guidelines:
- Repositories should encapsulate data access logic
- Repositories should be tested with integration tests against real databases"""

        service_content = """Service layer guidelines:
- Services contain business logic
- Services should be unit tested with mocked dependencies"""

        conflicts = detect_conflicts(repository_content, service_content, mock_llm_gateway_no_conflicts)

        assert isinstance(conflicts, ConflictList)
        assert len(conflicts.list) == 0, "Component-specific rules for different components should not conflict"


class DescribeFilterPairsByEmbeddingSimilarity:
    """Test the embedding similarity filter."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for chunks."""
        return ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.CONVENTIONAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )
        )

    @pytest.fixture
    def chunk_with_embedding(self, sample_metadata):
        """Create a chunk with an embedding."""
        def make_chunk(content, embedding):
            return KnowledgeChunk(
                id=f"test-{hash(content)}",
                content=content,
                concept="testing",
                metadata=sample_metadata,
                embedding=embedding
            )
        return make_chunk

    def should_return_empty_list_for_empty_input(self):
        """Test that empty input returns empty list."""
        from context_mixer.commands.operations.merge import filter_pairs_by_embedding_similarity

        result = filter_pairs_by_embedding_similarity([])
        assert result == []

    def should_filter_out_low_similarity_pairs(self, chunk_with_embedding):
        """Test that pairs with low similarity are filtered out."""
        from context_mixer.commands.operations.merge import filter_pairs_by_embedding_similarity

        # Create chunks with very different embeddings (orthogonal vectors)
        chunk1 = chunk_with_embedding("Content 1", [1.0, 0.0, 0.0])
        chunk2 = chunk_with_embedding("Content 2", [0.0, 1.0, 0.0])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_embedding_similarity(pairs, similarity_threshold=0.70)

        # Orthogonal vectors have similarity 0, should be filtered
        assert len(result) == 0

    def should_keep_high_similarity_pairs(self, chunk_with_embedding):
        """Test that pairs with high similarity are kept."""
        from context_mixer.commands.operations.merge import filter_pairs_by_embedding_similarity

        # Create chunks with very similar embeddings
        chunk1 = chunk_with_embedding("Content 1", [1.0, 0.0, 0.0])
        chunk2 = chunk_with_embedding("Content 2", [0.9, 0.1, 0.0])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_embedding_similarity(pairs, similarity_threshold=0.70)

        # Similar vectors should be kept
        assert len(result) == 1
        assert result[0] == (chunk1, chunk2)

    def should_keep_identical_embeddings(self, chunk_with_embedding):
        """Test that identical embeddings are kept."""
        from context_mixer.commands.operations.merge import filter_pairs_by_embedding_similarity

        chunk1 = chunk_with_embedding("Content 1", [1.0, 0.0, 0.0])
        chunk2 = chunk_with_embedding("Content 2", [1.0, 0.0, 0.0])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_embedding_similarity(pairs, similarity_threshold=0.70)

        # Identical embeddings should have similarity 1.0
        assert len(result) == 1

    def should_pass_through_chunks_without_embeddings(self, chunk_with_embedding, sample_metadata):
        """Test that chunks without embeddings are passed through (safety)."""
        from context_mixer.commands.operations.merge import filter_pairs_by_embedding_similarity

        chunk1 = KnowledgeChunk(
            id="test-1",
            content="Content 1",
            concept="testing",
            metadata=sample_metadata,
            embedding=None  # No embedding
        )
        chunk2 = chunk_with_embedding("Content 2", [1.0, 0.0, 0.0])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_embedding_similarity(pairs)

        # Should pass through for safety
        assert len(result) == 1


class DescribeFilterPairsByMetadata:
    """Test the metadata filter."""

    @pytest.fixture
    def sample_provenance(self):
        """Create sample provenance info."""
        return ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )

    @pytest.fixture
    def make_chunk(self, sample_provenance):
        """Create a chunk factory."""
        def _make_chunk(content, domains, concept=None):
            metadata = ChunkMetadata(
                domains=domains,
                authority=AuthorityLevel.CONVENTIONAL,
                scope=["enterprise"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                provenance=sample_provenance
            )
            return KnowledgeChunk(
                id=f"test-{hash(content)}",
                content=content,
                concept=concept or "testing",
                metadata=metadata,
                embedding=[1.0, 0.0, 0.0]
            )
        return _make_chunk

    def should_return_empty_list_for_empty_input(self):
        """Test that empty input returns empty list."""
        from context_mixer.commands.operations.merge import filter_pairs_by_metadata

        result = filter_pairs_by_metadata([])
        assert result == []

    def should_filter_out_non_overlapping_domains(self, make_chunk):
        """Test that pairs with no domain overlap are filtered out."""
        from context_mixer.commands.operations.merge import filter_pairs_by_metadata

        chunk1 = make_chunk("Content 1", ["frontend"], concept="frontend_stuff")
        chunk2 = make_chunk("Content 2", ["backend"], concept="backend_stuff")

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_metadata(pairs)

        # No overlapping domains and different concepts, should be filtered
        assert len(result) == 0

    def should_keep_overlapping_domains(self, make_chunk):
        """Test that pairs with overlapping domains are kept."""
        from context_mixer.commands.operations.merge import filter_pairs_by_metadata

        chunk1 = make_chunk("Content 1", ["frontend", "testing"])
        chunk2 = make_chunk("Content 2", ["testing", "backend"])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_metadata(pairs)

        # "testing" domain overlaps
        assert len(result) == 1

    def should_keep_matching_concepts(self, make_chunk):
        """Test that pairs with matching concepts are kept."""
        from context_mixer.commands.operations.merge import filter_pairs_by_metadata

        chunk1 = make_chunk("Content 1", ["frontend"], concept="authentication")
        chunk2 = make_chunk("Content 2", ["backend"], concept="authentication")

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_metadata(pairs)

        # Same concept, should be kept even with different domains
        assert len(result) == 1

    def should_pass_through_chunks_without_domains(self, make_chunk):
        """Test that chunks without domains are passed through (safety)."""
        from context_mixer.commands.operations.merge import filter_pairs_by_metadata

        chunk1 = make_chunk("Content 1", [])  # Empty domains
        chunk2 = make_chunk("Content 2", ["backend"])

        pairs = [(chunk1, chunk2)]
        result = filter_pairs_by_metadata(pairs)

        # Should pass through for safety
        assert len(result) == 1


class DescribeDetectConflictsMultiPair:
    """Test the batched multi-pair conflict detection."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for chunks."""
        return ChunkMetadata(
            domains=["technical"],
            authority=AuthorityLevel.CONVENTIONAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            provenance=ProvenanceInfo(
                source="test.md",
                created_at="2024-01-01T00:00:00Z"
            )
        )

    @pytest.fixture
    def make_chunk(self, sample_metadata):
        """Create a chunk factory."""
        def _make_chunk(content, concept=None):
            return KnowledgeChunk(
                id=f"test-{hash(content)}",
                content=content,
                concept=concept or "testing",
                metadata=sample_metadata,
                embedding=[1.0, 0.0, 0.0]
            )
        return _make_chunk

    @pytest.mark.asyncio
    async def should_return_empty_list_for_empty_input(self):
        """Test that empty input returns empty list."""
        from context_mixer.commands.operations.merge import detect_conflicts_multi_pair

        mock_gateway = MockLLMGateway()
        result = await detect_conflicts_multi_pair([], mock_gateway)

        assert result == []

    @pytest.mark.asyncio
    async def should_detect_conflicts_in_batch(self, make_chunk, mocker):
        """Test that conflicts are detected across multiple pairs in a batch."""
        from context_mixer.commands.operations.merge import detect_conflicts_multi_pair, MultiPairConflictResult, PairConflictResult

        chunk1 = make_chunk("Use 4 spaces for indentation")
        chunk2 = make_chunk("Use 2 spaces for indentation")
        chunk3 = make_chunk("Use tabs for indentation")

        pairs = [(chunk1, chunk2), (chunk2, chunk3)]

        # Create mock LLM gateway that returns conflicts
        mock_result = MultiPairConflictResult(
            pair_results=[
                PairConflictResult(
                    pair_index=0,
                    has_conflict=True,
                    conflicts=[
                        Conflict(
                            description="Indentation conflict",
                            conflicting_guidance=[
                                ConflictingGuidance(content="Use 4 spaces", source="chunk1"),
                            ]
                        )
                    ]
                ),
                PairConflictResult(
                    pair_index=1,
                    has_conflict=True,
                    conflicts=[
                        Conflict(
                            description="Indentation conflict",
                            conflicting_guidance=[
                                ConflictingGuidance(content="Use 2 spaces", source="chunk2"),
                            ]
                        )
                    ]
                )
            ]
        )

        mock_gateway = mocker.MagicMock()
        mock_gateway.generate_object.return_value = mock_result

        result = await detect_conflicts_multi_pair(pairs, mock_gateway, pairs_per_batch=2)

        assert len(result) == 2
        # First pair should have conflicts
        assert len(result[0][2].list) > 0
        # Second pair should have conflicts
        assert len(result[1][2].list) > 0

    @pytest.mark.asyncio
    async def should_handle_batches_larger_than_input(self, make_chunk, mocker):
        """Test that batch size larger than input works correctly."""
        from context_mixer.commands.operations.merge import detect_conflicts_multi_pair, MultiPairConflictResult, PairConflictResult

        chunk1 = make_chunk("Content 1")
        chunk2 = make_chunk("Content 2")

        pairs = [(chunk1, chunk2)]

        mock_result = MultiPairConflictResult(
            pair_results=[
                PairConflictResult(
                    pair_index=0,
                    has_conflict=False,
                    conflicts=[]
                )
            ]
        )

        mock_gateway = mocker.MagicMock()
        mock_gateway.generate_object.return_value = mock_result

        # Batch size of 10 for just 1 pair
        result = await detect_conflicts_multi_pair(pairs, mock_gateway, pairs_per_batch=10)

        assert len(result) == 1
