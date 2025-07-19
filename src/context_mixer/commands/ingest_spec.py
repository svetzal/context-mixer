"""
Tests for the ingest command.
"""

import pytest
from pathlib import Path

from rich.panel import Panel

from context_mixer.commands.ingest import do_ingest, IngestCommand, apply_conflict_resolutions
from context_mixer.commands.base import CommandContext, CommandResult
from context_mixer.commands.operations.merge import detect_conflicts
from context_mixer.commands.interactions.resolve_conflicts import resolve_conflicts
from context_mixer.commands.operations.commit import CommitOperation
from context_mixer.config import DEFAULT_ROOT_CONTEXT_FILENAME
from context_mixer.domain.conflict import Conflict, ConflictingGuidance
from context_mixer.domain.knowledge import (
    KnowledgeChunk, 
    ChunkMetadata, 
    ProvenanceInfo,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)
from context_mixer.domain.commit_message import CommitMessage
from context_mixer.gateways.git import GitGateway
from context_mixer.gateways.llm import LLMGateway
from context_mixer.domain.knowledge_store import KnowledgeStore
from context_mixer.spec_helpers import MessageMatcher

MERGED_CONTENT_FROM_LLM = "Merged content from LLM"


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()

@pytest.fixture
def mock_llm_gateway(mocker):
    mock = mocker.MagicMock(spec=LLMGateway)
    # Set up the generate method to return the merged content
    mock.generate.return_value = MERGED_CONTENT_FROM_LLM
    return mock

@pytest.fixture
def mock_config(tmp_path, mocker):
    config = mocker.MagicMock()
    library_path = tmp_path / "library"
    library_path.mkdir(exist_ok=True)
    config.library_path = library_path
    return config

@pytest.fixture
def mock_path(tmp_path):
    test_file = tmp_path / "test-file.md"
    test_file.write_text("Test content")
    return test_file

@pytest.fixture
def mock_git_gateway(mocker):
    mock = mocker.MagicMock(spec=GitGateway)
    # Set up the commit method to return success
    mock.commit.return_value = (True, "Commit successful")
    # Set up the add_all method to return success
    mock.add_all.return_value = (True, "All files staged")
    # Set up the get_diff method to return a sample diff
    mock.get_diff.return_value = ("Sample diff", None)
    return mock

@pytest.fixture
def mock_commit_operation(mocker, mock_git_gateway):
    mock = mocker.MagicMock(spec=CommitOperation)
    # Set up the commit_changes method to return success
    commit_msg = CommitMessage(short="Test commit message", long="Detailed test commit message")
    mock.commit_changes.return_value = (True, "Commit successful", commit_msg)
    return mock


@pytest.fixture
def mock_knowledge_store(mocker):
    """Mock knowledge store for dependency injection testing."""
    mock = mocker.MagicMock(spec=KnowledgeStore)
    # Set up common return values
    mock.detect_conflicts.return_value = []  # No conflicts by default
    mock.store_chunks.return_value = None  # Successful storage
    mock.get_all_chunks.return_value = []  # Empty store by default
    return mock


@pytest.fixture
def ingest_command(mock_knowledge_store):
    """Create IngestCommand with mocked dependencies."""
    return IngestCommand(mock_knowledge_store)


class DescribeIngestCommand:
    """Test the IngestCommand class with dependency injection."""

    async def should_execute_with_injected_knowledge_store(self, ingest_command, mock_console, mock_config, mock_llm_gateway, mock_path, mock_knowledge_store):
        """Test that IngestCommand uses the injected knowledge store."""
        # Create CommandContext with the required parameters
        context = CommandContext(
            console=mock_console,
            config=mock_config,
            llm_gateway=mock_llm_gateway,
            parameters={
                'path': mock_path,
                'commit': False,
                'detect_boundaries': False
            }
        )

        result = await ingest_command.execute(context)

        # Verify that the command executed successfully
        assert isinstance(result, CommandResult)
        assert result.success is True

        # Verify that the injected knowledge store was used
        assert ingest_command.knowledge_store is mock_knowledge_store

        # Verify console output
        assert mock_console.print.call_count >= 3


class DescribeDoIngest:

    async def should_print_messages_when_ingesting_to_empty_library(self, mock_console, mock_config, mock_llm_gateway, mock_path, mock_knowledge_store):
        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=mock_path, commit=False, detect_boundaries=False, knowledge_store=mock_knowledge_store)

        assert mock_console.print.call_count >= 3  # At least 3 print calls
        panel_call = mock_console.print.call_args_list[0]
        assert isinstance(panel_call[0][0], Panel)
        # Check that the success message appears in one of the print calls
        success_message_found = any("Successfully imported prompt as context.md" in str(call[0][0]) 
                                   for call in mock_console.print.call_args_list)
        assert success_message_found

    async def should_create_context_file_with_correct_content(self, mock_console, mock_config, mock_llm_gateway, mock_path, mock_knowledge_store):
        test_content = mock_path.read_text()

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=mock_path, commit=False, detect_boundaries=False, knowledge_store=mock_knowledge_store)

        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        assert output_file.exists()
        assert output_file.read_text() == test_content

    async def should_merge_with_existing_context_file(self, mock_console, mock_config, mock_llm_gateway, mock_path, mock_knowledge_store, mocker):
        # Mock detect_conflicts to return an empty ConflictList (no conflicts)
        from context_mixer.domain.conflict import ConflictList
        mocker.patch('context_mixer.commands.operations.merge.detect_conflicts', return_value=ConflictList(list=[]))

        # Create an existing context.md file
        existing_content = "Existing line 1\nExisting line 2\nShared line"
        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        output_file.write_text(existing_content)

        # Set up new content with some overlap
        mock_path.write_text("New line 1\nNew line 2\nShared line")

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=mock_path, commit=False, detect_boundaries=False, knowledge_store=mock_knowledge_store)

        # Check that the merged content is the response from the LLM broker
        merged_content = output_file.read_text()
        assert merged_content == MERGED_CONTENT_FROM_LLM

        # Verify that generate was called with a message containing both contents
        expected_content = ["Existing line 1", "New line 1"]
        mock_llm_gateway.generate.assert_called_once_with(messages=MessageMatcher(expected_content))

        # Check that the correct message was printed
        success_message_found = any("Successfully merged prompt with existing context.md" in str(call[0][0]) 
                                   for call in mock_console.print.call_args_list)
        assert success_message_found

    async def should_detect_and_resolve_conflicts(self, mock_console, mock_config, mock_llm_gateway, mock_path, mocker):
        # Create a mock Conflict object
        indent_4_string = "Use 4 spaces for indentation"
        indent_2_string = "Use 2 spaces for indentation"
        conflict = Conflict(
            description="There is a conflict in the indentation guidance",
            conflicting_guidance=[
                ConflictingGuidance(content=indent_4_string, source="existing"),
                ConflictingGuidance(content=indent_2_string, source="new")
            ],
            resolution=None
        )

        # Mock detect_conflicts to return a ConflictList containing the mock Conflict
        from context_mixer.domain.conflict import ConflictList
        mock_detect_conflicts = mocker.patch('context_mixer.commands.operations.merge.detect_conflicts', return_value=ConflictList(list=[conflict]))

        # Mock resolve_conflict to return the list of conflicts with resolutions set
        # Create a copy of the conflict with resolution set
        resolved_conflict = Conflict(
            description="There is a conflict in the indentation guidance",
            conflicting_guidance=[
                ConflictingGuidance(content=indent_4_string, source="existing"),
                ConflictingGuidance(content=indent_2_string, source="new")
            ],
            resolution=indent_4_string
        )
        mock_resolve_conflict = mocker.patch('context_mixer.commands.operations.merge.resolve_conflicts', return_value=[resolved_conflict])

        # Create an existing context.md file
        existing_content = "# Project Guidelines\n\n## Code Style\n\n- Use 4 spaces for indentation"
        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        output_file.write_text(existing_content)

        # Set up new content with conflicting guidance
        new_content = "# Coding Standards\n\n## Style Guide\n\n- Use 2 spaces for indentation"
        mock_path.write_text(new_content)

        # Set up mock_console.input to return "1" (choosing the first option)
        mock_console.input.return_value = "1"

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=mock_path, commit=False, detect_boundaries=False)

        # Check that detect_conflicts was called with the correct arguments
        mock_detect_conflicts.assert_called_once_with(existing_content, new_content, mock_llm_gateway)

        # Check that resolve_conflict was called with the list of conflicts and console
        mock_resolve_conflict.assert_called_once_with([conflict], mock_console, None)

        # Check that the merged content is the response from the LLM broker
        merged_content = output_file.read_text()
        assert merged_content == MERGED_CONTENT_FROM_LLM

        # Verify that generate was called with a message containing the resolved conflict
        expected_content = ["Resolution: Use 4 spaces for indentation"]
        mock_llm_gateway.generate.assert_called_once_with(messages=MessageMatcher(expected_content))

        # Check that the correct message was printed
        print_calls = [str(call[0][0]) for call in mock_console.print.call_args_list]
        success_message_found = any("Successfully merged prompt with existing context.md" in msg for msg in print_calls)
        assert success_message_found

    async def should_commit_changes_after_ingestion(self, mock_console, mock_config, mock_llm_gateway, mock_path, mocker):
        # Create a mock CommitMessage
        commit_msg = CommitMessage(short="Test commit message", long="Detailed test commit message")

        # Create a mock CommitOperation instance
        mock_commit_operation_instance = mocker.MagicMock(spec=CommitOperation)
        mock_commit_operation_instance.commit_changes.return_value = (True, "Commit successful", commit_msg)

        # Create a mock GitGateway instance
        mock_git_gateway_instance = mocker.MagicMock(spec=GitGateway)

        # Mock the GitGateway class
        mocker.patch('context_mixer.commands.ingest.GitGateway', return_value=mock_git_gateway_instance)

        # Mock the CommitOperation class
        mocker.patch('context_mixer.commands.ingest.CommitOperation', return_value=mock_commit_operation_instance)

        # Call do_ingest with commit=True
        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=mock_path, commit=True, detect_boundaries=False)

        # Verify that commit_changes was called with the correct arguments
        mock_commit_operation_instance.commit_changes.assert_called_once_with(mock_config.library_path)

        # Verify that the success message was printed
        print_calls = [str(call[0][0]) for call in mock_console.print.call_args_list]
        commit_success_message_found = any("Successfully committed changes: Test commit message" in msg for msg in print_calls)
        assert commit_success_message_found

    async def should_ingest_directory_with_multiple_files(self, mock_console, mock_config, mock_llm_gateway, tmp_path, mocker):
        # Create a temporary directory with multiple files
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()

        # Create some test files
        readme_file = test_dir / "README.md"
        readme_file.write_text("# Project README\nThis is the main documentation.")

        docs_dir = test_dir / "docs"
        docs_dir.mkdir()
        guide_file = docs_dir / "guide.md"
        guide_file.write_text("# User Guide\nDetailed instructions for users.")

        copilot_dir = test_dir / ".github"
        copilot_dir.mkdir()
        copilot_file = copilot_dir / "copilot-instructions.md"
        copilot_file.write_text("# Copilot Instructions\nCoding guidelines for AI.")

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=test_dir, commit=False, detect_boundaries=False)

        # Verify that multiple files were processed
        output_file = mock_config.library_path / DEFAULT_ROOT_CONTEXT_FILENAME
        assert output_file.exists()

        content = output_file.read_text()
        # Check that content from all files is included
        assert "Project README" in content
        assert "User Guide" in content
        assert "Copilot Instructions" in content

        # Verify console messages - check that ingestion completed successfully
        print_calls = [str(call[0][0]) for call in mock_console.print.call_args_list]
        # With progress tracking, individual "Processing:" messages are handled by the progress tracker
        # Instead, verify that the ingestion process completed successfully
        success_messages = [msg for msg in print_calls if "File reading completed" in msg or "Successfully merged" in msg]
        assert len(success_messages) >= 1  # Should have at least one success message

    async def should_handle_empty_directory(self, mock_console, mock_config, mock_llm_gateway, tmp_path):
        # Create an empty directory
        empty_dir = tmp_path / "empty_project"
        empty_dir.mkdir()

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=empty_dir, commit=False, detect_boundaries=False)

        # Verify that no files were processed and appropriate message was shown
        print_calls = [str(call[0][0]) for call in mock_console.print.call_args_list]
        no_files_message_found = any("No ingestible files found" in msg for msg in print_calls)
        assert no_files_message_found

    async def should_handle_nonexistent_path(self, mock_console, mock_config, mock_llm_gateway, tmp_path):
        # Use a path that doesn't exist
        nonexistent_path = tmp_path / "does_not_exist"

        await do_ingest(console=mock_console, config=mock_config, llm_gateway=mock_llm_gateway, path=nonexistent_path, commit=False, detect_boundaries=False)

        # Verify that error message was shown
        print_calls = [str(call[0][0]) for call in mock_console.print.call_args_list]
        error_message_found = any("does not exist" in msg for msg in print_calls)
        assert error_message_found


class DescribeApplyConflictResolutions:
    """Test the apply_conflict_resolutions pure function."""

    @pytest.fixture
    def sample_provenance(self):
        """Create sample provenance information."""
        return ProvenanceInfo(
            source="test_source.md",
            created_at="2024-01-01T00:00:00Z",
            author="test_author"
        )

    @pytest.fixture
    def sample_chunks(self, sample_provenance):
        """Create sample knowledge chunks for testing."""
        metadata1 = ChunkMetadata(
            domains=["technical", "coding"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["indentation", "style"],
            provenance=sample_provenance
        )

        metadata2 = ChunkMetadata(
            domains=["technical", "coding"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["indentation", "style"],
            provenance=sample_provenance
        )

        metadata3 = ChunkMetadata(
            domains=["technical", "coding"],
            authority=AuthorityLevel.OFFICIAL,
            scope=["enterprise"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["naming", "variables"],
            provenance=sample_provenance
        )

        return [
            KnowledgeChunk(
                id="chunk1",
                content="Use 4 spaces for indentation",
                metadata=metadata1,
                embedding=None
            ),
            KnowledgeChunk(
                id="chunk2", 
                content="Use 2 spaces for indentation",
                metadata=metadata2,
                embedding=None
            ),
            KnowledgeChunk(
                id="chunk3",
                content="Use camelCase for variables",
                metadata=metadata3,
                embedding=None
            )
        ]

    @pytest.fixture
    def sample_conflicts(self):
        """Create sample conflicts for testing."""
        return [
            Conflict(
                description="Indentation conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces for indentation", source="existing"),
                    ConflictingGuidance(content="Use 2 spaces for indentation", source="new")
                ],
                resolution="Use 4 spaces for indentation consistently"
            )
        ]

    def should_return_empty_results_when_no_conflicts(self, sample_chunks):
        """Test that function handles empty conflict list correctly."""
        resolved_conflicts = []
        valid_chunks = sample_chunks
        existing_chunks_to_store = [sample_chunks[2]]  # Non-conflicting chunk

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            resolved_conflicts, valid_chunks, existing_chunks_to_store
        )

        assert filtered_existing == [sample_chunks[2]]
        assert len(additional_chunks) == 2  # The other two chunks should be added
        assert len(messages) == 0

    def should_filter_conflicting_chunks_from_existing(self, sample_chunks, sample_conflicts):
        """Test that conflicting chunks are filtered from existing chunks to store."""
        valid_chunks = sample_chunks
        existing_chunks_to_store = sample_chunks.copy()  # All chunks initially marked for storage

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            sample_conflicts, valid_chunks, existing_chunks_to_store
        )

        # Should filter out chunks with conflicting content
        assert len(filtered_existing) == 1  # Only the non-conflicting chunk should remain
        assert filtered_existing[0].content == "Use camelCase for variables"

    def should_create_resolved_chunks_for_conflicts_with_resolutions(self, sample_chunks, sample_conflicts):
        """Test that resolved chunks are created for conflicts with resolutions."""
        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            sample_conflicts, valid_chunks, existing_chunks_to_store
        )

        # Should create one resolved chunk
        resolved_chunks = [chunk for chunk in additional_chunks if "_resolved_" in chunk.id]
        assert len(resolved_chunks) == 1

        resolved_chunk = resolved_chunks[0]
        assert resolved_chunk.content == "Use 4 spaces for indentation consistently"
        assert "_resolved_0" in resolved_chunk.id
        assert resolved_chunk.metadata == sample_chunks[0].metadata  # Uses template metadata

    def should_generate_resolution_messages(self, sample_chunks, sample_conflicts):
        """Test that resolution messages are generated for resolved conflicts."""
        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            sample_conflicts, valid_chunks, existing_chunks_to_store
        )

        assert len(messages) == 1
        assert "Created resolved chunk for conflict: Indentation conflict" in messages[0]

    def should_add_non_conflicting_chunks(self, sample_chunks, sample_conflicts):
        """Test that non-conflicting chunks are added to the result."""
        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            sample_conflicts, valid_chunks, existing_chunks_to_store
        )

        # Should include the non-conflicting chunk
        non_conflicting_chunks = [chunk for chunk in additional_chunks 
                                 if chunk.content == "Use camelCase for variables"]
        assert len(non_conflicting_chunks) == 1

    def should_handle_conflicts_without_resolutions(self, sample_chunks):
        """Test that conflicts without resolutions are handled gracefully."""
        conflicts_without_resolution = [
            Conflict(
                description="Unresolved conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces for indentation", source="existing"),
                    ConflictingGuidance(content="Use 2 spaces for indentation", source="new")
                ],
                resolution=None  # No resolution
            )
        ]

        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            conflicts_without_resolution, valid_chunks, existing_chunks_to_store
        )

        # Should not create any resolved chunks
        resolved_chunks = [chunk for chunk in additional_chunks if "_resolved_" in chunk.id]
        assert len(resolved_chunks) == 0
        assert len(messages) == 0

    def should_handle_multiple_conflicts(self, sample_chunks):
        """Test handling of multiple conflicts with resolutions."""
        multiple_conflicts = [
            Conflict(
                description="Indentation conflict",
                conflicting_guidance=[
                    ConflictingGuidance(content="Use 4 spaces for indentation", source="existing"),
                    ConflictingGuidance(content="Use 2 spaces for indentation", source="new")
                ],
                resolution="Use 4 spaces consistently"
            ),
            Conflict(
                description="Variable naming conflict", 
                conflicting_guidance=[
                    ConflictingGuidance(content="Use camelCase for variables", source="existing")
                ],
                resolution="Use camelCase for all variables"
            )
        ]

        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            multiple_conflicts, valid_chunks, existing_chunks_to_store
        )

        # Should create resolved chunks for both conflicts
        resolved_chunks = [chunk for chunk in additional_chunks if "_resolved_" in chunk.id]
        assert len(resolved_chunks) == 2
        assert len(messages) == 2

        # Check that IDs are unique
        assert "_resolved_0" in resolved_chunks[0].id
        assert "_resolved_1" in resolved_chunks[1].id

    def should_preserve_chunk_metadata_in_resolved_chunks(self, sample_chunks, sample_conflicts):
        """Test that resolved chunks preserve metadata from template chunks."""
        valid_chunks = sample_chunks
        existing_chunks_to_store = []

        filtered_existing, additional_chunks, messages = apply_conflict_resolutions(
            sample_conflicts, valid_chunks, existing_chunks_to_store
        )

        resolved_chunks = [chunk for chunk in additional_chunks if "_resolved_" in chunk.id]
        assert len(resolved_chunks) == 1

        resolved_chunk = resolved_chunks[0]
        # Should use metadata from one of the conflicting chunks (template chunk)
        assert resolved_chunk.metadata is not None
        assert resolved_chunk.embedding is None  # Should be None for regeneration
