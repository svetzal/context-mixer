"""
Tests for the ChunkingEngine module.

This module contains comprehensive tests for the ChunkingEngine's semantic boundary
detection and intelligent content chunking capabilities.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from context_mixer.domain.chunking_engine import (
    ChunkingEngine,
    ChunkBoundary,
    BoundaryDetectionResult,
    ConceptAnalysis,
    ValidationResult,
    StructuredChunkOutput,
    ChunkData
)
from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    ProvenanceInfo,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)
from context_mixer.gateways.llm import LLMGateway


@pytest.fixture
def mock_llm_gateway(mocker):
    return mocker.MagicMock(spec=LLMGateway)


@pytest.fixture
def chunking_engine(mock_llm_gateway):
    return ChunkingEngine(mock_llm_gateway)


@pytest.fixture
def sample_content():
    return """# Project Setup Guide

This guide explains how to set up the development environment for our project.

## Prerequisites

Before starting, ensure you have Python 3.8+ installed on your system.
You'll also need Git for version control.

## Installation Steps

1. Clone the repository from GitHub
2. Create a virtual environment
3. Install dependencies using pip
4. Run the initial setup script

## Configuration

The application uses environment variables for configuration.
Create a .env file in the root directory with the following variables:
- DATABASE_URL
- SECRET_KEY
- DEBUG_MODE

## Testing

Run the test suite using pytest to ensure everything is working correctly.
The tests are located in the tests/ directory."""


@pytest.fixture
def sample_boundary_result():
    return BoundaryDetectionResult(
        boundaries=[
            ChunkBoundary(
                start_position=0,
                end_position=100,
                concept="Project Setup Introduction",
                confidence=0.9,
                reasoning="Clear introduction section"
            ),
            ChunkBoundary(
                start_position=100,
                end_position=200,
                concept="Prerequisites",
                confidence=0.8,
                reasoning="Distinct prerequisites section"
            ),
            ChunkBoundary(
                start_position=200,
                end_position=350,
                concept="Installation Steps",
                confidence=0.85,
                reasoning="Step-by-step installation guide"
            )
        ],
        total_concepts=3,
        confidence_score=0.85
    )


@pytest.fixture
def sample_concept_analysis():
    return ConceptAnalysis(
        concept="Development Setup",
        domains=["technical", "development"],
        authority_level=AuthorityLevel.OFFICIAL,
        granularity=GranularityLevel.DETAILED,
        tags=["setup", "development", "guide"],
        dependencies=["python", "git"],
        conflicts_with=[]
    )


@pytest.fixture
def sample_metadata():
    return ChunkMetadata(
        domains=["technical"],
        authority=AuthorityLevel.CONVENTIONAL,
        scope=["general"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.CURRENT,
        dependencies=[],
        conflicts=[],
        tags=["test"],
        provenance=ProvenanceInfo(
            source="test",
            created_at="2024-01-01T00:00:00",
            author="test"
        )
    )


class DescribeChunkingEngine:

    def should_initialize_with_llm_gateway(self, mock_llm_gateway):
        engine = ChunkingEngine(mock_llm_gateway)

        assert engine.llm_gateway == mock_llm_gateway

    def should_detect_semantic_boundaries_using_llm(self, chunking_engine, mock_llm_gateway, sample_content, sample_boundary_result):
        mock_llm_gateway.generate_object.return_value = sample_boundary_result

        result = chunking_engine.detect_semantic_boundaries(sample_content)

        assert result == sample_boundary_result
        assert mock_llm_gateway.generate_object.called
        call_args = mock_llm_gateway.generate_object.call_args
        assert call_args[0][1] == BoundaryDetectionResult  # Second argument should be the model class

    def should_fallback_to_paragraph_detection_when_llm_fails(self, chunking_engine, mock_llm_gateway, sample_content):
        mock_llm_gateway.generate_object.side_effect = Exception("LLM error")

        result = chunking_engine.detect_semantic_boundaries(sample_content)

        assert isinstance(result, BoundaryDetectionResult)
        assert len(result.boundaries) > 0
        assert result.confidence_score == 0.5  # Fallback confidence
        assert all("Paragraph" in boundary.concept for boundary in result.boundaries)

    def should_chunk_content_by_concepts(self, chunking_engine, mock_llm_gateway, sample_content, sample_boundary_result, sample_concept_analysis):
        mock_llm_gateway.generate_object.side_effect = [sample_boundary_result, sample_concept_analysis, sample_concept_analysis, sample_concept_analysis]

        chunks = chunking_engine.chunk_by_concepts(sample_content, source="test.md")

        assert len(chunks) == len(sample_boundary_result.boundaries)
        assert all(isinstance(chunk, KnowledgeChunk) for chunk in chunks)
        assert all(chunk.metadata.provenance.source == "test.md" for chunk in chunks)
        assert all(chunk.id.startswith("chunk_") for chunk in chunks)

    def should_create_single_chunk_when_no_boundaries_detected(self, chunking_engine, mock_llm_gateway, sample_content, sample_concept_analysis):
        empty_boundary_result = BoundaryDetectionResult(boundaries=[], total_concepts=0, confidence_score=0.0)
        mock_llm_gateway.generate_object.side_effect = [empty_boundary_result, sample_concept_analysis]

        chunks = chunking_engine.chunk_by_concepts(sample_content, source="test.md")

        assert len(chunks) == 1
        assert chunks[0].content == sample_content
        assert chunks[0].metadata.provenance.source == "test.md"

    def should_validate_chunk_completeness_using_llm(self, chunking_engine, mock_llm_gateway, sample_metadata):
        chunk = KnowledgeChunk(
            id="test_chunk",
            content="This is a complete chunk with proper content.",
            metadata=sample_metadata
        )
        mock_validation_result = ValidationResult(
            is_complete=True,
            reason="The text represents a complete and coherent concept",
            confidence=0.9,
            issues=[]
        )
        mock_llm_gateway.generate_object.return_value = mock_validation_result

        result = chunking_engine.validate_chunk_completeness(chunk)

        assert result.is_complete is True
        assert result.reason == "The text represents a complete and coherent concept"
        assert result.confidence == 0.9
        assert mock_llm_gateway.generate_object.called

    def should_detect_incomplete_chunks_using_llm(self, chunking_engine, mock_llm_gateway, sample_metadata):
        chunk = KnowledgeChunk(
            id="test_chunk",
            content="This chunk appears to be truncated but doesn't have obvious indicators",
            metadata=sample_metadata
        )
        mock_validation_result = ValidationResult(
            is_complete=False,
            reason="The text appears to be cut off mid-sentence without proper conclusion",
            confidence=0.85,
            issues=["incomplete_thought"]
        )
        mock_llm_gateway.generate_object.return_value = mock_validation_result

        result = chunking_engine.validate_chunk_completeness(chunk)

        assert result.is_complete is False
        assert result.reason == "The text appears to be cut off mid-sentence without proper conclusion"
        assert "incomplete_thought" in result.issues

    def should_reject_empty_chunks(self, chunking_engine, sample_metadata):
        chunk = KnowledgeChunk(
            id="test_chunk",
            content="",
            metadata=sample_metadata
        )

        result = chunking_engine.validate_chunk_completeness(chunk)

        assert result.is_complete is False
        assert "empty" in result.reason.lower()
        assert "empty_content" in result.issues

    def should_reject_very_short_chunks(self, chunking_engine, sample_metadata):
        chunk = KnowledgeChunk(
            id="test_chunk",
            content="Short",
            metadata=sample_metadata
        )

        result = chunking_engine.validate_chunk_completeness(chunk)

        assert result.is_complete is False
        assert "too short" in result.reason.lower()
        assert "too_short" in result.issues

    def should_detect_truncation_indicators(self, chunking_engine, sample_metadata):
        truncated_contents = [
            "This content ends with...",
            "This content ends with etc.",
            "This content ends with,",
            "This content ends with and",
            "This content ends with or"
        ]

        for content in truncated_contents:
            chunk = KnowledgeChunk(
                id="test_chunk",
                content=content,
                metadata=sample_metadata
            )

            result = chunking_engine.validate_chunk_completeness(chunk)
            assert result.is_complete is False
            assert "truncated" in result.reason.lower()
            assert len(result.issues) > 0

    def should_fallback_gracefully_when_llm_validation_fails(self, chunking_engine, mock_llm_gateway, sample_metadata):
        chunk = KnowledgeChunk(
            id="test_chunk",
            content="This is a reasonable chunk of content.",
            metadata=sample_metadata
        )
        mock_llm_gateway.generate_object.side_effect = Exception("LLM error")

        result = chunking_engine.validate_chunk_completeness(chunk)

        assert result.is_complete is True  # Should assume complete on error
        assert "llm validation failed" in result.reason.lower()
        assert "llm_failure" in result.issues
        assert result.confidence < 0.5  # Low confidence due to fallback

    def should_generate_unique_chunk_ids(self, chunking_engine):
        content1 = "First piece of content"
        content2 = "Second piece of content"
        concept = "Test Concept"

        id1 = chunking_engine._generate_chunk_id(content1, concept)
        id2 = chunking_engine._generate_chunk_id(content2, concept)

        assert id1 != id2
        assert id1.startswith("chunk_")
        assert id2.startswith("chunk_")
        assert len(id1) == len("chunk_") + 12  # chunk_ + 12 hex chars

    def should_generate_consistent_chunk_ids(self, chunking_engine):
        content = "Same content"
        concept = "Same concept"

        id1 = chunking_engine._generate_chunk_id(content, concept)
        id2 = chunking_engine._generate_chunk_id(content, concept)

        assert id1 == id2

    def should_analyze_chunk_concepts_using_llm(self, chunking_engine, mock_llm_gateway, sample_concept_analysis):
        mock_llm_gateway.generate_object.return_value = sample_concept_analysis
        content = "Sample content for analysis"

        analysis = chunking_engine._analyze_chunk_concept(content)

        assert analysis == sample_concept_analysis
        assert mock_llm_gateway.generate_object.called
        call_args = mock_llm_gateway.generate_object.call_args
        assert call_args[0][1] == ConceptAnalysis

    def should_provide_fallback_concept_analysis(self, chunking_engine, mock_llm_gateway):
        mock_llm_gateway.generate_object.side_effect = Exception("LLM error")
        content = "Sample content for analysis"

        analysis = chunking_engine._analyze_chunk_concept(content)

        assert isinstance(analysis, ConceptAnalysis)
        assert analysis.concept == "General content"
        assert analysis.domains == ["general"]
        assert analysis.authority_level == AuthorityLevel.CONVENTIONAL
        assert analysis.granularity == GranularityLevel.DETAILED


class DescribeBoundaryDetectionResult:

    def should_contain_boundary_information(self, sample_boundary_result):
        assert len(sample_boundary_result.boundaries) == 3
        assert sample_boundary_result.total_concepts == 3
        assert sample_boundary_result.confidence_score == 0.85

    def should_have_valid_boundary_objects(self, sample_boundary_result):
        for boundary in sample_boundary_result.boundaries:
            assert isinstance(boundary, ChunkBoundary)
            assert boundary.start_position >= 0
            assert boundary.end_position > boundary.start_position
            assert len(boundary.concept) > 0
            assert 0.0 <= boundary.confidence <= 1.0
            assert len(boundary.reasoning) > 0


class DescribeConceptAnalysis:

    def should_contain_concept_metadata(self, sample_concept_analysis):
        assert sample_concept_analysis.concept == "Development Setup"
        assert "technical" in sample_concept_analysis.domains
        assert sample_concept_analysis.authority_level == AuthorityLevel.OFFICIAL
        assert sample_concept_analysis.granularity == GranularityLevel.DETAILED
        assert "setup" in sample_concept_analysis.tags
        assert "python" in sample_concept_analysis.dependencies
        assert len(sample_concept_analysis.conflicts_with) == 0


class DescribeStructuredOutputChunking:
    """Test the new structured output chunking approach."""

    @pytest.fixture
    def structured_output_sample(self):
        """Sample structured output for testing."""
        return StructuredChunkOutput(
            chunks=[
                ChunkData(
                    content="# Project Setup Guide\n\nThis guide explains how to set up the development environment for our project.",
                    concept="Project Setup Introduction",
                    domains=["technical", "development"],
                    authority="official",
                    scope=["general"],
                    granularity="detailed",
                    tags=["setup", "guide", "introduction"],
                    dependencies=[],
                    conflicts=[]
                ),
                ChunkData(
                    content="## Prerequisites\n\nBefore starting, ensure you have Python 3.8+ installed on your system.\nYou'll also need Git for version control.",
                    concept="Prerequisites",
                    domains=["technical", "development"],
                    authority="official",
                    scope=["general"],
                    granularity="detailed",
                    tags=["prerequisites", "python", "git"],
                    dependencies=["python", "git"],
                    conflicts=[]
                ),
                ChunkData(
                    content="## Installation Steps\n\n1. Clone the repository from GitHub\n2. Create a virtual environment\n3. Install dependencies using pip\n4. Run the initial setup script",
                    concept="Installation Steps",
                    domains=["technical", "development"],
                    authority="official",
                    scope=["general"],
                    granularity="detailed",
                    tags=["installation", "setup", "steps"],
                    dependencies=["python", "git"],
                    conflicts=[]
                )
            ]
        )

    def should_chunk_using_structured_output(self, chunking_engine, mock_llm_gateway, sample_content, structured_output_sample):
        """Test that structured output chunking produces complete chunks."""
        # Mock the LLM to return structured output
        mock_llm_gateway.generate_object.return_value = structured_output_sample

        chunks = chunking_engine.chunk_by_structured_output(sample_content, source="test.md")

        # Should produce the expected number of chunks
        assert len(chunks) == 3

        # Each chunk should be complete and have proper metadata
        for chunk in chunks:
            assert isinstance(chunk, KnowledgeChunk)
            assert chunk.content.strip()  # Non-empty content
            assert chunk.metadata.domains  # Has domains
            assert chunk.metadata.authority  # Has authority level
            assert chunk.metadata.tags  # Has tags
            assert chunk.metadata.provenance.source == "test.md"
            assert chunk.metadata.provenance.author == "LLM-StructuredChunking"

    def should_fallback_to_character_based_when_structured_fails(self, chunking_engine, mock_llm_gateway, sample_content):
        """Test that structured output falls back to character-based approach when LLM fails."""
        # Mock the LLM to raise an exception
        mock_llm_gateway.generate_object.side_effect = Exception("LLM connection failed")

        chunks = chunking_engine.chunk_by_structured_output(sample_content, source="test.md")

        # Should still produce chunks via fallback
        assert len(chunks) > 0

        # Chunks should still be valid KnowledgeChunk objects
        for chunk in chunks:
            assert isinstance(chunk, KnowledgeChunk)
            assert chunk.content.strip()
            assert chunk.metadata.provenance.source == "test.md"

    def should_handle_empty_structured_output(self, chunking_engine, mock_llm_gateway, sample_content):
        """Test handling of empty structured output."""
        # Mock the LLM to return empty chunks
        empty_output = StructuredChunkOutput(chunks=[])
        mock_llm_gateway.generate_object.return_value = empty_output

        chunks = chunking_engine.chunk_by_structured_output(sample_content, source="test.md")

        # Should fallback and still produce chunks
        assert len(chunks) > 0
