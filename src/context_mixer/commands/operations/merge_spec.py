"""
Tests for the merge operations.
"""

import asyncio
import pytest

from context_mixer.commands.operations.merge import format_conflict_resolutions, detect_conflicts_async, detect_conflicts_batch
from context_mixer.domain.conflict import Conflict, ConflictingGuidance, ConflictList
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, AuthorityLevel, ProvenanceInfo, GranularityLevel, TemporalScope
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
                    ConflictingGuidance(content="Use 2 spaces for indentation", source="new")
                ],
                resolution="Use 4 spaces for indentation consistently across the codebase"
            ),
            Conflict(
                description="Variable naming convention conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use camelCase for variables", source="existing"),
                    ConflictingGuidance(content="Use snake_case for variables", source="new")
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

    def should_return_empty_string_when_conflicts_have_no_resolutions(self):
        """Test that function returns empty string when conflicts have no resolutions."""
        conflicts_without_resolution = [
            Conflict(
                description="Unresolved conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces", source="existing"),
                    ConflictingGuidance(content="Use 2 spaces", source="new")
                ],
                resolution=None
            )
        ]

        result = format_conflict_resolutions(conflicts_without_resolution)
        assert result == ""

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

    def should_skip_conflicts_without_resolutions_in_mixed_list(self):
        """Test that conflicts without resolutions are skipped in mixed lists."""
        mixed_conflicts = [
            Conflict(
                description="Resolved conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use tabs", source="existing"),
                    ConflictingGuidance(content="Use spaces", source="new")
                ],
                resolution="Use spaces for indentation"
            ),
            Conflict(
                description="Unresolved conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use semicolons", source="existing"),
                    ConflictingGuidance(content="No semicolons", source="new")
                ],
                resolution=None  # No resolution
            )
        ]

        result = format_conflict_resolutions(mixed_conflicts)

        # Should include the resolved conflict
        assert "Description: Resolved conflict" in result
        assert "Resolution: Use spaces for indentation" in result

        # Should not include the unresolved conflict
        assert "Unresolved conflict" not in result
        assert "Use semicolons" not in result

    def should_handle_empty_resolution_strings(self):
        """Test that conflicts with empty resolution strings are skipped."""
        conflicts_with_empty_resolution = [
            Conflict(
                description="Conflict with empty resolution",
                conflicting_guidance=[
                    ConflictingGuidance(content="Option A", source="existing"),
                    ConflictingGuidance(content="Option B", source="new")
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
                    ConflictingGuidance(content="Option B", source="new")
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
                ConflictingGuidance(content="Content B", source="new")
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
            source="test_file1.py",
            created_at="2024-01-01T00:00:00Z"
        )
        provenance2 = ProvenanceInfo(
            source="test_file2.py",
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
                ConflictingGuidance(content="Content B", source="new")
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
            source="test_file3.py",
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
