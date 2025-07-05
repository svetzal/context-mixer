"""
ChunkingEngine for semantic boundary detection and intelligent content chunking.

This module implements the ChunkingEngine that can detect semantic boundaries
in content and create concept-based knowledge chunks following CRAFT principles.
"""

import hashlib
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field
from mojentic.llm import LLMMessage, MessageRole

from context_mixer.domain.knowledge import (
    KnowledgeChunk, 
    ChunkMetadata, 
    AuthorityLevel, 
    GranularityLevel, 
    TemporalScope,
    ProvenanceInfo
)
from context_mixer.gateways.llm import LLMGateway


class ChunkBoundary(BaseModel):
    """Represents a semantic boundary in content."""
    start_position: int = Field(..., description="Character position where the boundary starts")
    end_position: int = Field(..., description="Character position where the boundary ends")
    concept: str = Field(..., description="The main concept or topic of this chunk")
    confidence: float = Field(..., description="Confidence score for this boundary (0.0-1.0)")
    reasoning: str = Field(..., description="Explanation of why this is a semantic boundary")


class BoundaryDetectionResult(BaseModel):
    """Result of semantic boundary detection."""
    boundaries: List[ChunkBoundary] = Field(..., description="List of detected semantic boundaries")
    total_concepts: int = Field(..., description="Total number of distinct concepts found")
    confidence_score: float = Field(..., description="Overall confidence in the boundary detection")


class ConceptAnalysis(BaseModel):
    """Analysis of a content chunk's concept and metadata."""
    concept: str = Field(..., description="Main concept or topic")
    domains: List[str] = Field(..., description="Knowledge domains (technical, business, design)")
    authority_level: AuthorityLevel = Field(..., description="Suggested authority level")
    granularity: GranularityLevel = Field(..., description="Content detail level")
    tags: List[str] = Field(..., description="Relevant tags for searchability")
    dependencies: List[str] = Field(default_factory=list, description="Concepts this depends on")
    conflicts_with: List[str] = Field(default_factory=list, description="Concepts this might conflict with")


class ValidationResult(BaseModel):
    """Result of chunk validation with detailed reasoning."""
    is_complete: bool = Field(..., description="Whether the chunk is complete and coherent")
    reason: str = Field(..., description="Detailed explanation of the validation result")
    confidence: float = Field(..., description="Confidence score for the validation (0.0-1.0)")
    issues: List[str] = Field(default_factory=list, description="Specific issues found if incomplete")


class ChunkingEngine:
    """
    Engine for intelligent content chunking with semantic boundary detection.

    This engine uses LLM capabilities to detect semantic boundaries in content
    and create concept-based knowledge chunks following CRAFT principles.
    """

    def __init__(self, llm_gateway: LLMGateway):
        """
        Initialize the ChunkingEngine.

        Args:
            llm_gateway: Gateway for LLM interactions
        """
        self.llm_gateway = llm_gateway

    def detect_semantic_boundaries(self, content: str) -> BoundaryDetectionResult:
        """
        Detect semantic boundaries in the given content.

        This method uses LLM analysis to identify where distinct concepts
        begin and end in the content, enabling concept-based chunking.

        Args:
            content: The text content to analyze

        Returns:
            BoundaryDetectionResult containing detected boundaries
        """
        # Prepare the prompt for boundary detection
        messages = [
            LLMMessage(
                role=MessageRole.System,
                content="""You are an expert at analyzing text content to identify semantic boundaries where distinct concepts begin and end.

Your task is to analyze the given content and identify semantic boundaries that would create coherent, concept-focused chunks. Each boundary should represent a transition from one main concept or topic to another.

Guidelines:
- Look for transitions between different concepts, topics, or domains
- Consider natural paragraph breaks, but don't rely on them exclusively
- Aim for chunks that are conceptually coherent and self-contained
- Prefer boundaries that create chunks of 200-800 tokens when possible
- Each chunk should focus on a single main concept or closely related concepts
- Provide confidence scores based on how clear the conceptual transition is

Return your analysis as a structured response."""
            ),
            LLMMessage(
                role=MessageRole.User, 
                content=f"Analyze this content and identify semantic boundaries:\n\n{content}"
            )
        ]

        try:
            result = self.llm_gateway.generate_object(messages, BoundaryDetectionResult)
            return result
        except Exception as e:
            # Fallback to simple paragraph-based boundaries if LLM fails
            return self._fallback_boundary_detection(content)

    def chunk_by_concepts(self, content: str, source: str = "unknown") -> List[KnowledgeChunk]:
        """
        Chunk content by concepts using semantic boundary detection.

        Args:
            content: The content to chunk
            source: Source identifier for provenance tracking

        Returns:
            List of KnowledgeChunk objects
        """
        # First detect semantic boundaries
        boundary_result = self.detect_semantic_boundaries(content)

        chunks = []

        # If no boundaries detected, treat entire content as one chunk
        if not boundary_result.boundaries:
            chunk = self._create_single_chunk(content, source)
            if chunk:
                chunks.append(chunk)
            return chunks

        # Create chunks based on detected boundaries
        for i, boundary in enumerate(boundary_result.boundaries):
            # Adjust boundary positions to align with word boundaries
            start_pos = self._adjust_to_word_boundary(content, boundary.start_position, direction="start")
            end_pos = self._adjust_to_word_boundary(content, boundary.end_position, direction="end")

            chunk_content = content[start_pos:end_pos].strip()

            if not chunk_content:
                continue

            # Analyze the chunk to determine its metadata
            concept_analysis = self._analyze_chunk_concept(chunk_content)

            # Create chunk metadata
            metadata = ChunkMetadata(
                domains=concept_analysis.domains,
                authority=concept_analysis.authority_level,
                scope=["general"],  # Default scope, could be enhanced
                granularity=concept_analysis.granularity,
                temporal=TemporalScope.CURRENT,
                dependencies=concept_analysis.dependencies,
                conflicts=concept_analysis.conflicts_with,
                tags=concept_analysis.tags,
                provenance=ProvenanceInfo(
                    source=source,
                    created_at=datetime.now().isoformat(),
                    author="chunking_engine"
                )
            )

            # Generate chunk ID based on content hash
            chunk_id = self._generate_chunk_id(chunk_content, boundary.concept)

            # Create the knowledge chunk
            chunk = KnowledgeChunk(
                id=chunk_id,
                content=chunk_content,
                metadata=metadata
            )

            chunks.append(chunk)

        return chunks

    def validate_chunk_completeness(self, chunk: KnowledgeChunk) -> ValidationResult:
        """
        Validate that a chunk is complete and coherent.

        Args:
            chunk: The knowledge chunk to validate

        Returns:
            ValidationResult with detailed reasoning about the validation
        """
        # Basic validation checks
        if not chunk.content.strip():
            return ValidationResult(
                is_complete=False,
                reason="Chunk content is empty or contains only whitespace",
                confidence=1.0,
                issues=["empty_content"]
            )

        if len(chunk.content.strip()) < 10:  # Too short to be meaningful
            return ValidationResult(
                is_complete=False,
                reason=f"Chunk content is too short ({len(chunk.content.strip())} characters) to be meaningful",
                confidence=1.0,
                issues=["too_short"]
            )

        # Check for obvious truncation indicators
        truncation_indicators = {
            "ellipsis": chunk.content.endswith("..."),
            "etc": chunk.content.endswith("etc."),
            "trailing_comma": chunk.content.rstrip().endswith(","),
            "trailing_and": chunk.content.rstrip().endswith("and"),
            "trailing_or": chunk.content.rstrip().endswith("or"),
        }

        found_indicators = [name for name, found in truncation_indicators.items() if found]
        if found_indicators:
            return ValidationResult(
                is_complete=False,
                reason=f"Chunk appears truncated due to: {', '.join(found_indicators)}",
                confidence=0.9,
                issues=found_indicators
            )

        # Use LLM for more sophisticated completeness check
        try:
            messages = [
                LLMMessage(
                    role=MessageRole.System,
                    content="""You are an expert at evaluating whether a piece of text represents a complete, coherent concept or if it appears to be truncated or incomplete.

Analyze the given text and provide a detailed assessment including:
1. Whether it represents a complete thought or concept
2. Whether it has a clear beginning and end
3. Whether it appears to be cut off mid-sentence or mid-concept
4. Whether it contains enough context to be understood independently
5. Your confidence in this assessment
6. Specific issues if any are found

Provide your analysis as a structured response."""
                ),
                LLMMessage(
                    role=MessageRole.User,
                    content=f"Evaluate this text for completeness:\n\n{chunk.content}"
                )
            ]

            result = self.llm_gateway.generate_object(messages, ValidationResult)
            return result

        except Exception as e:
            # Fallback to basic validation if LLM fails
            return ValidationResult(
                is_complete=True,
                reason=f"LLM validation failed ({str(e)}), assuming complete based on basic checks",
                confidence=0.3,
                issues=["llm_failure"]
            )

    def _fallback_boundary_detection(self, content: str) -> BoundaryDetectionResult:
        """
        Fallback boundary detection using simple heuristics.

        Args:
            content: Content to analyze

        Returns:
            BoundaryDetectionResult with simple paragraph-based boundaries
        """
        boundaries = []
        paragraphs = content.split('\n\n')
        current_pos = 0

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                start_pos = current_pos
                end_pos = current_pos + len(paragraph)

                boundaries.append(ChunkBoundary(
                    start_position=start_pos,
                    end_position=end_pos,
                    concept=f"Paragraph {i+1}",
                    confidence=0.5,  # Low confidence for fallback
                    reasoning="Fallback paragraph-based boundary detection"
                ))

            current_pos += len(paragraph) + 2  # +2 for \n\n

        return BoundaryDetectionResult(
            boundaries=boundaries,
            total_concepts=len(boundaries),
            confidence_score=0.5
        )

    def _analyze_chunk_concept(self, chunk_content: str) -> ConceptAnalysis:
        """
        Analyze a chunk to determine its concept and metadata.

        Args:
            chunk_content: The content to analyze

        Returns:
            ConceptAnalysis with suggested metadata
        """
        messages = [
            LLMMessage(
                role=MessageRole.System,
                content="""You are an expert at analyzing text content to determine its main concept, domain, and characteristics.

Analyze the given text and determine:
1. The main concept or topic
2. Knowledge domains it belongs to (technical, business, design, legal, etc.)
3. Appropriate authority level (foundational, official, conventional, experimental, deprecated)
4. Granularity level (summary, overview, detailed, comprehensive)
5. Relevant tags for searchability
6. Any concepts this might depend on
7. Any concepts this might conflict with

Consider the CRAFT principles:
- Coherence: Does it focus on a single concept?
- Resistance: Does it avoid mixing conflicting information?
- Authority: What level of authority does this represent?
- Freshness: Is this current information?
- Traceability: Can we identify its domain and scope?"""
            ),
            LLMMessage(
                role=MessageRole.User,
                content=f"Analyze this content:\n\n{chunk_content}"
            )
        ]

        try:
            return self.llm_gateway.generate_object(messages, ConceptAnalysis)
        except Exception:
            # Fallback analysis
            return ConceptAnalysis(
                concept="General content",
                domains=["general"],
                authority_level=AuthorityLevel.CONVENTIONAL,
                granularity=GranularityLevel.DETAILED,
                tags=["general"],
                dependencies=[],
                conflicts_with=[]
            )

    def _create_single_chunk(self, content: str, source: str) -> Optional[KnowledgeChunk]:
        """
        Create a single chunk from content when no boundaries are detected.

        Args:
            content: The content to chunk
            source: Source identifier

        Returns:
            KnowledgeChunk or None if content is invalid
        """
        if not content.strip():
            return None

        concept_analysis = self._analyze_chunk_concept(content)

        metadata = ChunkMetadata(
            domains=concept_analysis.domains,
            authority=concept_analysis.authority_level,
            scope=["general"],
            granularity=concept_analysis.granularity,
            temporal=TemporalScope.CURRENT,
            dependencies=concept_analysis.dependencies,
            conflicts=concept_analysis.conflicts_with,
            tags=concept_analysis.tags,
            provenance=ProvenanceInfo(
                source=source,
                created_at=datetime.now().isoformat(),
                author="chunking_engine"
            )
        )

        chunk_id = self._generate_chunk_id(content, concept_analysis.concept)

        return KnowledgeChunk(
            id=chunk_id,
            content=content,
            metadata=metadata
        )

    def _adjust_to_word_boundary(self, content: str, position: int, direction: str) -> int:
        """
        Adjust a position to align with word boundaries.

        Args:
            content: The full content text
            position: The original position
            direction: "start" to find the beginning of a word/sentence, "end" to find the end

        Returns:
            Adjusted position that aligns with word boundaries
        """
        if position <= 0:
            return 0
        if position >= len(content):
            return len(content)

        # For start direction, move backward to find a good starting point
        if direction == "start":
            # Look for sentence boundaries first (. ! ?)
            for i in range(position, max(0, position - 200), -1):
                if i > 0 and content[i-1] in '.!?':
                    # Skip whitespace after sentence boundary
                    while i < len(content) and content[i].isspace():
                        i += 1
                    return i

            # If no sentence boundary found, look for paragraph boundaries
            for i in range(position, max(0, position - 200), -1):
                if i > 0 and content[i-1:i+1] == '\n\n':
                    return i + 1

            # If no paragraph boundary, look for line boundaries
            for i in range(position, max(0, position - 100), -1):
                if i > 0 and content[i-1] == '\n':
                    return i

            # Last resort: find word boundary
            for i in range(position, max(0, position - 50), -1):
                if i > 0 and content[i-1].isspace() and not content[i].isspace():
                    return i

            return max(0, position)

        # For end direction, move forward to find a good ending point
        else:  # direction == "end"
            # Look for sentence boundaries first
            for i in range(position, min(len(content), position + 200)):
                if content[i] in '.!?':
                    return i + 1

            # If no sentence boundary found, look for paragraph boundaries
            for i in range(position, min(len(content), position + 200)):
                if i < len(content) - 1 and content[i:i+2] == '\n\n':
                    return i

            # If no paragraph boundary, look for line boundaries
            for i in range(position, min(len(content), position + 100)):
                if content[i] == '\n':
                    return i

            # Last resort: find word boundary
            for i in range(position, min(len(content), position + 50)):
                if content[i].isspace():
                    return i

            return min(len(content), position)

    def _generate_chunk_id(self, content: str, concept: str) -> str:
        """
        Generate a unique ID for a chunk based on its content and concept.

        Args:
            content: The chunk content
            concept: The main concept

        Returns:
            Unique chunk identifier
        """
        # Create a hash of the content and concept
        combined = f"{concept}:{content}"
        hash_object = hashlib.sha256(combined.encode())
        hash_hex = hash_object.hexdigest()

        # Return first 12 characters for a shorter ID
        return f"chunk_{hash_hex[:12]}"
