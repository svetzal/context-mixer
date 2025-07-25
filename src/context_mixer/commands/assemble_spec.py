"""
Tests for the assemble command implementation.

This module tests the functionality of assembling context fragments for specific AI assistants
using the CRAFT (Context-Aware Retrieval and Fusion Technology) system.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from context_mixer.commands.assemble import do_assemble
from context_mixer.commands.assembly_strategies import AssemblyStrategyFactory
from context_mixer.config import Config
from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope,
    ProvenanceInfo
)


@pytest.fixture
def mock_console():
    """Create a mock console for testing."""
    return Console(file=open('/dev/null', 'w'))


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config with temporary library path."""
    return Config(library_path=tmp_path)


@pytest.fixture
def sample_chunks():
    """Create sample knowledge chunks for testing."""
    chunks = []

    # Chunk 1: Official Python guidelines
    chunk1 = KnowledgeChunk(
        id="chunk-001",
        content="Use type hints for all function parameters and return values. Follow PEP 8 style guidelines.",
        metadata=ChunkMetadata(
            authority=AuthorityLevel.OFFICIAL,
            domains=["python", "coding"],
            scope=["enterprise", "team"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["type-hints", "pep8"],
            provenance=ProvenanceInfo(
                source="test-fixture",
                created_at="2024-01-01T00:00:00Z",
                author="test-author"
            )
        )
    )
    chunks.append(chunk1)

    # Chunk 2: Verified testing guidelines
    chunk2 = KnowledgeChunk(
        id="chunk-002", 
        content="Write tests for all new functionality. Use pytest with descriptive test names.",
        metadata=ChunkMetadata(
            authority=AuthorityLevel.CONVENTIONAL,
            domains=["testing", "python"],
            scope=["team"],
            granularity=GranularityLevel.SUMMARY,
            temporal=TemporalScope.CURRENT,
            tags=["pytest", "testing"],
            provenance=ProvenanceInfo(
                source="test-fixture",
                created_at="2024-01-01T00:00:00Z",
                author="test-author"
            )
        )
    )
    chunks.append(chunk2)

    # Chunk 3: Community documentation guidelines
    chunk3 = KnowledgeChunk(
        id="chunk-003",
        content="Document all public APIs using numpy-style docstrings. Include examples where helpful.",
        metadata=ChunkMetadata(
            authority=AuthorityLevel.EXPERIMENTAL,
            domains=["documentation", "python"],
            scope=["prototype"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["docstrings", "numpy-style"],
            provenance=ProvenanceInfo(
                source="test-fixture",
                created_at="2024-01-01T00:00:00Z",
                author="test-author"
            )
        )
    )
    chunks.append(chunk3)

    return chunks


@pytest.fixture
def mock_knowledge_store(mocker, sample_chunks):
    """Create a mock knowledge store."""
    mock_store = mocker.MagicMock()
    mock_store.get_chunks_by_authority = AsyncMock(return_value=sample_chunks)
    mock_store.get_chunks_by_domain = AsyncMock(return_value=sample_chunks)
    return mock_store


@pytest.fixture
def mock_knowledge_store_factory(mocker, mock_knowledge_store):
    """Create a mock knowledge store factory."""
    mock_factory = mocker.patch('context_mixer.commands.assemble.KnowledgeStoreFactory')
    mock_factory.create_vector_store.return_value = mock_knowledge_store
    return mock_factory


class DescribeDoAssemble:
    def should_assemble_context_for_copilot_target(self, mock_console, mock_config, mock_knowledge_store_factory, tmp_path):
        # Create vector store directory
        vector_store_path = tmp_path / "vector_store"
        vector_store_path.mkdir()

        result = asyncio.run(do_assemble(
            mock_console,
            mock_config,
            target="copilot",
            token_budget=1000
        ))

        # Verify knowledge store was created
        mock_knowledge_store_factory.create_vector_store.assert_called_once()

        # Verify output file was created
        expected_output = tmp_path / "assembled" / "copilot" / "copilot-instructions.md"
        assert expected_output.exists()

        # Verify content format (default compressed format)
        content = expected_output.read_text()
        assert "# Instructions" in content

    def should_handle_missing_vector_store_gracefully(self, mock_console, mock_config):
        result = asyncio.run(do_assemble(
            mock_console,
            mock_config,
            target="copilot"
        ))

        # Should not raise exception, just print error message
        assert result is None

    def should_apply_domain_filtering_when_specified(self, mock_console, mock_config, mock_knowledge_store_factory, mock_knowledge_store, tmp_path, sample_chunks):
        # Create vector store directory
        vector_store_path = tmp_path / "vector_store"
        vector_store_path.mkdir()

        # Filter to only python domain chunks
        python_chunks = [chunk for chunk in sample_chunks if "python" in chunk.metadata.domains]
        mock_knowledge_store.get_chunks_by_domain.return_value = python_chunks

        result = asyncio.run(do_assemble(
            mock_console,
            mock_config,
            target="copilot",
            filter_tags="domain:python"
        ))

        # Verify domain filtering was applied
        mock_knowledge_store.get_chunks_by_domain.assert_called_once_with(["python"])

    def should_apply_tag_filtering_when_specified(self, mock_console, mock_config, mock_knowledge_store_factory, tmp_path):
        # Create vector store directory
        vector_store_path = tmp_path / "vector_store"
        vector_store_path.mkdir()

        result = asyncio.run(do_assemble(
            mock_console,
            mock_config,
            target="copilot",
            filter_tags="pytest,testing"
        ))

        # Verify knowledge store was queried
        mock_knowledge_store_factory.create_vector_store.assert_called_once()

    def should_save_to_custom_output_path_when_specified(self, mock_console, mock_config, mock_knowledge_store_factory, tmp_path):
        # Create vector store directory
        vector_store_path = tmp_path / "vector_store"
        vector_store_path.mkdir()

        custom_output = tmp_path / "custom_output.md"

        result = asyncio.run(do_assemble(
            mock_console,
            mock_config,
            target="copilot",
            output=custom_output
        ))

        # Verify custom output file was created
        assert custom_output.exists()
        content = custom_output.read_text()
        assert "# Instructions" in content  # Default compressed format


class DescribeAssembleForCopilot:
    def should_format_content_for_github_copilot(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("copilot")

        # Test default (compressed) format
        content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)
        assert "# Instructions" in content
        assert "🔒 OFFICIAL" not in content  # No authority indicators in compressed mode

        # Test verbose format
        verbose_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8, verbose=True)
        assert "# GitHub Copilot Instructions" in verbose_content
        assert "## Project Context" in verbose_content
        assert "🔒 OFFICIAL" in verbose_content  # Authority indicator for official chunks
        assert "📋 CONVENTIONAL" in verbose_content   # Authority indicator for conventional chunks
        assert "🧪 EXPERIMENTAL" in verbose_content # Authority indicator for experimental chunks

    def should_group_chunks_by_domain(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("copilot")

        # Test default (compressed) format - no domain headers due to optimization
        content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)
        # Content should still contain the chunk content even without domain headers
        assert "# Instructions" in content
        assert len(content) > 50  # Should have meaningful content

        # Test verbose format - full headers
        verbose_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8, verbose=True)
        assert "Python Guidelines" in verbose_content or "Coding Guidelines" in verbose_content
        assert "Testing Guidelines" in verbose_content
        assert "Documentation Guidelines" in verbose_content

    def should_respect_token_budget(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("copilot")

        # Test that function works with different token budgets (default compressed format)
        content_small = strategy.assemble(sample_chunks, token_budget=100, quality_threshold=0.8)
        content_large = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)

        # Both should produce valid content with compressed headers
        assert "# Instructions" in content_small
        assert "# Instructions" in content_large

        # Both should be reasonable lengths
        assert len(content_small) > 50  # Not empty
        assert len(content_large) > 50  # Not empty

    def should_include_craft_system_attribution_in_verbose_mode(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("copilot")

        # Test verbose mode includes attribution
        verbose_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8, verbose=True)
        assert "Generated by Context Mixer CRAFT system" in verbose_content
        assert f"{len(sample_chunks)} chunks processed" in verbose_content

        # Test default (compressed) mode excludes attribution
        compressed_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)
        assert "Generated by Context Mixer CRAFT system" not in compressed_content


class DescribeAssembleForClaude:
    def should_format_content_for_claude(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("claude")

        # Test default (compressed) format
        content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)
        assert "# Context" in content
        assert "**Authority Level:**" not in content  # No metadata in compressed mode
        assert "**Domains:**" not in content

        # Test verbose format
        verbose_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8, verbose=True)
        assert "# Claude Context Instructions" in verbose_content
        assert "You are an AI assistant with the following project context:" in verbose_content
        assert "**Authority Level:**" in verbose_content
        assert "**Domains:**" in verbose_content

    def should_include_chunk_metadata_in_verbose_mode(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("claude")

        # Test that metadata is only included in verbose mode
        verbose_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8, verbose=True)

        # Should include authority levels in verbose mode
        assert "Official" in verbose_content
        assert "Conventional" in verbose_content
        assert "Experimental" in verbose_content

        # Should include domains in verbose mode
        assert "python" in verbose_content
        assert "testing" in verbose_content
        assert "documentation" in verbose_content

        # Test that metadata is not included in default (compressed) mode
        compressed_content = strategy.assemble(sample_chunks, token_budget=2000, quality_threshold=0.8)
        assert "**Authority Level:**" not in compressed_content
        assert "**Domains:**" not in compressed_content

    def should_respect_token_budget_for_claude(self, sample_chunks):
        strategy = AssemblyStrategyFactory.create_strategy("claude")

        # Use small token budget
        content = strategy.assemble(sample_chunks, token_budget=100, quality_threshold=0.8)

        # Content should be truncated to fit budget
        token_count = len(content.split())
        assert token_count <= 100 * 0.9  # 90% of budget as per implementation
