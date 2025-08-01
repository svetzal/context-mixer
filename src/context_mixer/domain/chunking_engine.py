"""
ChunkingEngine for semantic boundary detection and intelligent content chunking.

This module implements the ChunkingEngine that can detect semantic boundaries
in content and create concept-based knowledge chunks following CRAFT principles.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Dict, Any

from mojentic.llm import LLMMessage, MessageRole
from pydantic import BaseModel, Field

from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope,
    ProvenanceInfo
)
from context_mixer.gateways.llm import LLMGateway


def generate_chunk_id(content: str, concept: str) -> str:
    """
    Pure function to generate a unique ID for a chunk based on its content and concept.

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


def split_content_on_blank_lines(content: str) -> List[Dict[str, Any]]:
    """
    Pure function to extract natural text units (paragraphs, sections, headers) from content.

    Args:
        content: The content to split

    Returns:
        List of units with their content, type, and position info
    """
    units = []

    # Use a more robust approach that tracks positions during splitting
    # Split by double newlines but keep track of separators
    parts = re.split(r'(\n\s*\n)', content)

    current_pos = 0

    for i in range(0, len(parts), 2):  # Skip separators (odd indices)
        if i >= len(parts):
            break

        part = parts[i].strip()
        if not part:
            # Update position for empty parts
            current_pos += len(parts[i])
            if i + 1 < len(parts):  # Add separator length
                current_pos += len(parts[i + 1])
            continue

        # Calculate exact positions
        start_pos = current_pos
        # Find where the stripped content actually starts
        original_part = parts[i]
        leading_whitespace = len(original_part) - len(original_part.lstrip())
        start_pos += leading_whitespace
        end_pos = start_pos + len(part)

        units.append({
            'content': part,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': len(part)
        })

        # Update current position
        current_pos += len(parts[i])
        if i + 1 < len(parts):  # Add separator length
            current_pos += len(parts[i + 1])

    return units






def create_fallback_groupings(num_units: int) -> List[List[int]]:
    """
    Pure function to create simple fallback groupings.

    Args:
        num_units: Total number of units to group

    Returns:
        List of groups (0-based indices)
    """
    if num_units <= 3:
        return [list(range(num_units))]

    # Group units in pairs/triples
    groupings = []
    i = 0
    while i < num_units:
        if i + 2 < num_units:
            groupings.append([i, i+1, i+2])
            i += 3
        elif i + 1 < num_units:
            groupings.append([i, i+1])
            i += 2
        else:
            groupings.append([i])
            i += 1

    return groupings


def fallback_grouping(units: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Pure function for fallback grouping based on unit sizes.

    Args:
        units: List of text units

    Returns:
        List of groups
    """
    groupings = []
    current_group = []
    current_size = 0
    max_group_size = 1000  # characters

    for i, unit in enumerate(units):
        # Add to current group if it won't make it too large
        if current_size + unit['length'] <= max_group_size:
            current_group.append(i)
            current_size += unit['length']
        # Start new group if current one is getting too large
        else:
            if current_group:
                groupings.append(current_group)
            current_group = [i]
            current_size = unit['length']

    # Add the last group
    if current_group:
        groupings.append(current_group)

    return groupings


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


class ChunkData(BaseModel):
    """Individual chunk data for structured output."""
    content: str = Field(..., description="The complete content of this knowledge chunk")
    concept: str = Field(..., description="The main concept or topic this chunk covers")
    domains: List[str] = Field(..., description="Knowledge domains (e.g., technical, business, design)")
    authority: str = Field(..., description="Authority level: foundational, official, conventional, experimental, deprecated")
    scope: List[str] = Field(..., description="Applicable scopes (e.g., enterprise, prototype, mobile-only)")
    granularity: str = Field(..., description="Detail level: summary, overview, detailed, comprehensive")
    tags: List[str] = Field(..., description="Searchable tags for this chunk")
    dependencies: List[str] = Field(default_factory=list, description="Concepts this chunk depends on")
    conflicts: List[str] = Field(default_factory=list, description="Concepts this chunk conflicts with")


class StructuredChunkOutput(BaseModel):
    """Structured output model for LLM to emit complete chunks directly."""
    chunks: List[ChunkData] = Field(..., description="List of complete knowledge chunks")


class UnitGroupingResponse(BaseModel):
    """Structured response model for unit grouping analysis."""
    groups: List[List[int]] = Field(..., description="Groups of unit indices that should be chunked together")
    reasoning: Optional[str] = Field(None, description="Explanation of the grouping decisions")


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
        Detect semantic boundaries in the given content using a hierarchical approach.

        This method first splits content into natural units (paragraphs, sections)
        and then uses LLM analysis to group these units semantically, avoiding
        the character position problem that leads to poor splits.

        Args:
            content: The text content to analyze

        Returns:
            BoundaryDetectionResult containing detected boundaries
        """
        # Use the improved hierarchical approach
        return self._hierarchical_boundary_detection(content)

    def chunk_by_concepts(self, content: str, source: str = "unknown") -> List[KnowledgeChunk]:
        """
        Chunk content by concepts using hierarchical unit-based approach.

        Args:
            content: The content to chunk
            source: Source identifier for provenance tracking

        Returns:
            List of KnowledgeChunk objects
        """
        # Use the new unit-based approach that avoids character position problems
        return self._chunk_by_units(content, source)

    def chunk_by_structured_output(self, content: str, source: str = "unknown", project_id: str = None, project_name: str = None, project_path: str = None) -> List[KnowledgeChunk]:
        """
        Chunk content using structured LLM output to directly emit complete chunks.

        This approach bypasses character-based parsing entirely and has the LLM
        directly output complete, semantically coherent chunks with all metadata.
        This should achieve much higher completion rates than character-based approaches.

        Args:
            content: The content to chunk
            source: Source identifier for provenance tracking
            project_id: Project identifier for organizing knowledge by project
            project_name: Human-readable project name
            project_path: Root path of the source project

        Returns:
            List of KnowledgeChunk objects
        """
        try:
            # Ask the LLM to directly emit complete chunks with all metadata
            messages = [
                LLMMessage(
                    role=MessageRole.System,
                    content="""You are an expert knowledge curator following CRAFT principles for chunking content into domain-coherent units.

Your task is to analyze the given content and break it into complete, semantically coherent knowledge chunks. Each chunk should:

1. Contain a complete concept or idea that can stand alone
2. Be semantically bounded to prevent knowledge interference
3. Include all necessary context to be understood independently
4. Follow domain separation principles (technical, business, design, etc.)
5. RESPECT SIZE LIMITS based on granularity level

For each chunk, you must provide:
- Complete content (the full text of the chunk)
- Concept name (what this chunk is about)
- Domains (technical, business, design, process, etc.)
- Authority level (foundational, official, conventional, experimental, deprecated)
- Scope tags (enterprise, prototype, mobile-only, etc.)
- Granularity (summary, overview, detailed, comprehensive)
- Searchable tags
- Dependencies (concepts this chunk requires)
- Conflicts (concepts this chunk contradicts)

CRITICAL SIZE CONSTRAINTS - Each chunk must respect these token limits:
- summary: 50-100 tokens (roughly 40-80 words)
- overview: 200-300 tokens (roughly 150-240 words)
- detailed: 500-800 tokens (roughly 400-640 words)
- comprehensive: 1000+ tokens (roughly 800+ words)

Choose the appropriate granularity level for each chunk and ensure the content length matches that level. If content is too long for a granularity level, either split it into multiple chunks or choose a higher granularity level.

CRITICAL: Do not try to preserve exact character positions or markdown formatting. Focus on semantic completeness and conceptual coherence while respecting size limits. Each chunk should be a complete, self-contained unit of knowledge within its size constraints.

Output complete chunks directly - do not emit metadata about character positions or line numbers."""
                ),
                LLMMessage(
                    role=MessageRole.User,
                    content=f"Please analyze this content and break it into complete, semantically coherent knowledge chunks:\n\n{content}"
                )
            ]

            # Use structured output to get complete chunks directly
            structured_output = self.llm_gateway.generate_object(messages, StructuredChunkOutput)

            # Check if we got any chunks
            if not structured_output.chunks:
                # Fallback to character-based approach if no chunks returned
                return self._chunk_by_units(content, source)

            # Convert the structured output to KnowledgeChunk objects
            chunks = []
            for i, chunk_data in enumerate(structured_output.chunks):
                # Create provenance info
                provenance = ProvenanceInfo(
                    source=source,
                    project_id=project_id,
                    project_name=project_name,
                    project_path=project_path,
                    created_at=datetime.now().isoformat(),
                    author="LLM-StructuredChunking"
                )

                # Map string values to enums
                try:
                    authority = AuthorityLevel(chunk_data.authority.lower())
                except ValueError:
                    authority = AuthorityLevel.CONVENTIONAL

                try:
                    granularity = GranularityLevel(chunk_data.granularity.lower())
                except ValueError:
                    granularity = GranularityLevel.DETAILED

                # Create metadata
                metadata = ChunkMetadata(
                    domains=chunk_data.domains,
                    authority=authority,
                    scope=chunk_data.scope,
                    granularity=granularity,
                    temporal=TemporalScope.CURRENT,
                    dependencies=chunk_data.dependencies,
                    conflicts=chunk_data.conflicts,
                    tags=chunk_data.tags,
                    provenance=provenance
                )

                # Generate chunk ID
                chunk_id = self._generate_chunk_id(chunk_data.content, chunk_data.concept)

                # Create the knowledge chunk
                chunk = KnowledgeChunk(
                    id=chunk_id,
                    content=chunk_data.content,
                    metadata=metadata
                )

                chunks.append(chunk)

            return chunks

        except Exception as e:
            # Fallback to the old approach if structured output fails
            print(f"Structured output chunking failed: {e}")
            return self._chunk_by_units(content, source)

    def _chunk_by_units(self, content: str, source: str) -> List[KnowledgeChunk]:
        """
        Chunk content by analyzing and grouping natural text units.

        This approach avoids character position problems by working directly
        with natural text units and reconstructing content from grouped units.

        Args:
            content: The content to chunk
            source: Source identifier for provenance tracking

        Returns:
            List of KnowledgeChunk objects
        """
        # Step 1: Extract natural units
        units = self._extract_natural_units(content)

        if not units:
            return []

        if len(units) == 1:
            # Single unit - create one chunk
            chunk = self._create_chunk_from_units([units[0]], source, 0)
            return [chunk] if chunk else []

        # Step 2: Group related units
        try:
            groupings = self._analyze_unit_relationships(units)
        except Exception:
            # Fallback to simple grouping
            groupings = self._fallback_grouping(units)

        # Step 3: Create chunks from grouped units
        chunks = []
        for i, group in enumerate(groupings):
            if not group:
                continue

            # Get the units for this group
            group_units = [units[idx] for idx in group if idx < len(units)]
            if not group_units:
                continue

            # Create chunk from these units
            chunk = self._create_chunk_from_units(group_units, source, i)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk_from_units(self, units: List[Dict[str, Any]], source: str, group_idx: int) -> Optional[KnowledgeChunk]:
        """
        Create a knowledge chunk from a group of units.

        Args:
            units: List of text units to combine
            source: Source identifier
            group_idx: Index of the group (for concept naming)

        Returns:
            KnowledgeChunk or None if invalid
        """
        if not units:
            return None

        # Reconstruct content by joining units with appropriate separators
        chunk_content = self._reconstruct_content_from_units(units)

        if not chunk_content.strip():
            return None

        # Generate concept name from the units
        concept = self._generate_concept_from_units(units, group_idx)

        # Analyze the chunk content
        try:
            concept_analysis = self._analyze_chunk_concept(chunk_content)
            # Ensure we got the right type of object with valid enum values
            if (not hasattr(concept_analysis, 'domains') or 
                not isinstance(concept_analysis.authority_level, AuthorityLevel) or
                not isinstance(concept_analysis.granularity, GranularityLevel)):
                raise ValueError("Invalid concept analysis result")
        except Exception:
            # Fallback analysis
            concept_analysis = ConceptAnalysis(
                concept=concept,
                domains=["general"],
                authority_level=AuthorityLevel.CONVENTIONAL,
                granularity=GranularityLevel.DETAILED,
                tags=[concept.lower().replace(" ", "-")],
                dependencies=[],
                conflicts_with=[]
            )

        # Create chunk metadata
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

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(chunk_content, concept)

        # Create the knowledge chunk
        return KnowledgeChunk(
            id=chunk_id,
            content=chunk_content,
            metadata=metadata
        )

    def _reconstruct_content_from_units(self, units: List[Dict[str, Any]]) -> str:
        """
        Reconstruct content from a list of units with appropriate separators.

        Args:
            units: List of text units

        Returns:
            Reconstructed content string
        """
        if not units:
            return ""

        if len(units) == 1:
            return units[0]['content']

        # Join units with double newlines to preserve paragraph structure
        content_parts = []
        for unit in units:
            content_parts.append(unit['content'])

        return '\n\n'.join(content_parts)

    def _generate_concept_from_units(self, units: List[Dict[str, Any]], group_idx: int) -> str:
        """
        Generate a concept name from a group of units.

        Args:
            units: List of text units
            group_idx: Index of the group

        Returns:
            Concept name
        """
        # Look for headers in the units (check content directly)
        for unit in units:
            content = unit['content']
            header_match = re.match(r'^#+\s*(.+)', content)
            if header_match:
                return header_match.group(1).strip()

        # Look for title-like units (short content without newlines)
        for unit in units:
            content = unit['content']
            if len(content) < 100 and '\n' not in content:
                return content[:50].strip()

        # Look for the first meaningful line in any unit
        for unit in units:
            lines = unit['content'].split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and not line.startswith('#'):
                    # Extract first few words as concept
                    words = line.split()[:5]
                    return ' '.join(words)

        # Fallback to generic name
        return f"Section {group_idx + 1}"

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

        if len(chunk.content.strip()) < 10:  # Too short to be meaningful (relaxed for instruction fragments)
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

        # Additional check for mid-sentence truncation (more sophisticated)
        content_stripped = chunk.content.rstrip()
        if (content_stripped.endswith(",") and 
            not any(word in content_stripped.lower().split()[-10:] for word in ["list", "items", "options", "choices", "examples"])):
            truncation_indicators["suspicious_comma"] = True

        found_indicators = [name for name, found in truncation_indicators.items() if found]
        if found_indicators:
            return ValidationResult(
                is_complete=False,
                reason=f"Chunk appears truncated due to: {', '.join(found_indicators)}",
                confidence=0.85,  # Reduced confidence since we're being more lenient
                issues=found_indicators
            )

        # Use LLM for more sophisticated completeness check
        try:
            messages = [
                LLMMessage(
                    role=MessageRole.System,
                    content="""You evaluate text for usefulness. Be extremely lenient and accept virtually all content as complete and useful.

ACCEPT these as complete and useful:
- Any instruction, even if brief or partial
- Any code snippet or command
- Any configuration example
- Any principle or concept definition
- Any requirement or prerequisite
- Any troubleshooting tip
- Headers, titles, or section names (even without detailed content)
- Brief statements or rules
- Partial lists or incomplete sentences that convey meaning
- Any text that provides guidance or information, however minimal

ONLY mark as incomplete if the text is:
- Completely garbled or corrupted
- Random meaningless characters
- Empty or only whitespace
- Clearly cut off mid-word (not mid-sentence)

Be very generous in your evaluation. If there's any doubt, mark it as complete.
Provide your analysis as a structured response."""
                ),
                LLMMessage(
                    role=MessageRole.User,
                    content=f"Evaluate this text for completeness:\n\n{chunk.content}"
                )
            ]

            result = self.llm_gateway.generate_object(messages, ValidationResult)

            # Override LLM decision only if it's being too strict with low confidence
            # If the chunk has meaningful content and LLM has low confidence, accept it
            if not result.is_complete and len(chunk.content.strip()) > 5 and result.confidence < 0.7:
                # Check if content has meaningful words
                words = chunk.content.strip().split()
                if len(words) >= 2:  # At least 2 words suggests meaningful content
                    return ValidationResult(
                        is_complete=True,
                        reason="Overriding LLM decision - content appears meaningful despite low-confidence LLM assessment",
                        confidence=0.8,
                        issues=[]
                    )

            return result

        except Exception as e:
            # Fallback to basic validation if LLM fails
            return ValidationResult(
                is_complete=True,
                reason=f"LLM validation failed ({str(e)}), assuming complete based on basic checks",
                confidence=0.3,
                issues=["llm_failure"]
            )

    def validate_chunks_parallel(self, chunks: List[KnowledgeChunk], max_workers: Optional[int] = None) -> List[ValidationResult]:
        """
        Validate multiple chunks in parallel for improved performance.

        Args:
            chunks: List of knowledge chunks to validate
            max_workers: Maximum number of worker threads (defaults to min(32, len(chunks) + 4))

        Returns:
            List of ValidationResult objects in the same order as input chunks

        Raises:
            Exception: If any individual chunk validation fails critically
        """
        if not chunks:
            return []

        # Handle single chunk case - no need for parallelization
        if len(chunks) == 1:
            return [self.validate_chunk_completeness(chunks[0])]

        # Determine optimal number of workers
        if max_workers is None:
            # Use Python's default: min(32, len(chunks) + 4)
            max_workers = min(32, len(chunks) + 4)
        else:
            # Ensure we don't use more workers than chunks
            max_workers = min(max_workers, len(chunks))

        results = [None] * len(chunks)  # Pre-allocate results list to maintain order
        failed_chunks = []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all validation tasks
                future_to_index = {
                    executor.submit(self.validate_chunk_completeness, chunk): idx
                    for idx, chunk in enumerate(chunks)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    chunk_index = future_to_index[future]
                    try:
                        result = future.result()
                        results[chunk_index] = result
                    except Exception as e:
                        # Handle individual chunk validation failure
                        failed_chunks.append((chunk_index, str(e)))
                        # Create a fallback validation result for failed chunks
                        results[chunk_index] = ValidationResult(
                            is_complete=False,
                            reason=f"Validation failed with error: {str(e)}",
                            confidence=0.0,
                            issues=["validation_error"]
                        )

        except Exception as e:
            # Handle thread pool execution failure
            raise Exception(f"Parallel validation failed: {str(e)}")

        # Log any individual failures for debugging
        if failed_chunks:
            # Note: In a real implementation, you might want to use proper logging
            # For now, we'll just ensure the results are still returned
            pass

        return results

    def _hierarchical_boundary_detection(self, content: str) -> BoundaryDetectionResult:
        """
        Hierarchical boundary detection that avoids character position problems.

        This method:
        1. Splits content into natural units (paragraphs, sections, headers)
        2. Uses LLM to analyze semantic relationships between units
        3. Groups related units into coherent chunks
        4. Creates boundaries based on natural text structure

        Args:
            content: Content to analyze

        Returns:
            BoundaryDetectionResult with semantically meaningful boundaries
        """
        # Step 1: Split content into natural units
        units = self._extract_natural_units(content)

        if len(units) <= 1:
            # If only one unit, no boundaries needed
            return BoundaryDetectionResult(
                boundaries=[],
                total_concepts=1,
                confidence_score=0.9
            )

        # Step 2: Analyze semantic relationships between units
        try:
            groupings = self._analyze_unit_relationships(units)

            # Step 3: Create boundaries based on semantic groupings
            boundaries = self._create_boundaries_from_groupings(units, groupings, content)

            return BoundaryDetectionResult(
                boundaries=boundaries,
                total_concepts=len(boundaries) if boundaries else 1,
                confidence_score=0.85
            )

        except Exception as e:
            # Fallback to simple unit-based boundaries if LLM fails
            return self._create_unit_based_boundaries(units, content)

    def _extract_natural_units(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract natural text units (paragraphs, sections, headers) from content.

        Args:
            content: The content to split

        Returns:
            List of units with their content, type, and position info
        """
        return split_content_on_blank_lines(content)


    def _analyze_unit_relationships(self, units: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Analyze semantic relationships between units to determine groupings.

        Args:
            units: List of text units

        Returns:
            List of groups, where each group is a list of unit indices
        """
        if len(units) <= 2:
            return [list(range(len(units)))]

        # Create a summary of each unit for analysis
        unit_summaries = []
        for i, unit in enumerate(units):
            summary = f"Unit {i+1}: {unit['content'][:200]}..."
            unit_summaries.append(summary)

        # Ask LLM to group related units
        messages = [
            LLMMessage(
                role=MessageRole.System,
                content="""You are an expert at analyzing text structure and semantic relationships.

Your task is to group related text units that should belong together in the same conceptual chunk. 

Guidelines:
- Group units that discuss the same concept, topic, or theme
- Keep headers with their related content
- Keep code examples with their explanations
- Keep list items together when they form a coherent set
- Prefer groups of 2-5 units when possible
- Each group should be conceptually coherent and self-contained

Provide groups as lists of unit numbers (1-based indexing). For example, if units 1-2 should be grouped together, units 3-5 should form another group, and unit 6 should stand alone, return groups: [[1, 2], [3, 4, 5], [6]]."""
            ),
            LLMMessage(
                role=MessageRole.User,
                content=f"Analyze these text units and group related ones:\n\n" + "\n\n".join(unit_summaries)
            )
        ]

        try:
            # Use structured output for reliable parsing
            response = self.llm_gateway.generate_object(messages, UnitGroupingResponse)

            # Convert from 1-based to 0-based indices and validate
            groupings = []
            for group in response.groups:
                if isinstance(group, list):
                    zero_based_group = [i-1 for i in group if isinstance(i, int) and 1 <= i <= len(units)]
                    if zero_based_group:
                        groupings.append(zero_based_group)

            # Ensure all units are covered
            if groupings:
                covered_units = set()
                for group in groupings:
                    covered_units.update(group)

                # Add missing units as individual groups
                for i in range(len(units)):
                    if i not in covered_units:
                        groupings.append([i])

                return groupings

        except Exception:
            pass

        # Fallback: group consecutive units of similar types
        return fallback_grouping(units)


    def _create_fallback_groupings(self, num_units: int) -> List[List[int]]:
        """Create simple fallback groupings."""
        return create_fallback_groupings(num_units)

    def _fallback_grouping(self, units: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Fallback grouping based on unit types and sizes.

        Args:
            units: List of text units

        Returns:
            List of groups
        """
        return fallback_grouping(units)

    def _create_boundaries_from_groupings(self, units: List[Dict[str, Any]], 
                                        groupings: List[List[int]], 
                                        content: str) -> List[ChunkBoundary]:
        """
        Create chunk boundaries based on unit groupings.

        Args:
            units: List of text units
            groupings: List of unit index groups
            content: Original content

        Returns:
            List of ChunkBoundary objects
        """
        boundaries = []

        for group_idx, group in enumerate(groupings):
            if not group:
                continue

            # Find the start and end positions for this group
            start_unit = units[group[0]]
            end_unit = units[group[-1]]

            start_pos = start_unit['start_pos']
            end_pos = end_unit['end_pos']

            # Generate a concept name based on the group content
            group_content = []
            for unit_idx in group:
                unit_content = units[unit_idx]['content']
                # Take first line or first 50 chars as concept hint
                first_line = unit_content.split('\n')[0][:50]
                group_content.append(first_line)

            concept = self._generate_concept_name(group_content, group_idx)

            boundaries.append(ChunkBoundary(
                start_position=start_pos,
                end_position=end_pos,
                concept=concept,
                confidence=0.8,
                reasoning=f"Semantic grouping of {len(group)} related units"
            ))

        return boundaries

    def _generate_concept_name(self, group_content: List[str], group_idx: int) -> str:
        """
        Generate a concept name for a group of units.

        Args:
            group_content: List of content snippets from the group
            group_idx: Index of the group

        Returns:
            Concept name
        """
        # Look for headers or titles in the group
        for content in group_content:
            if content.startswith('#'):
                # Extract header text
                header = re.sub(r'^#+\s*', '', content).strip()
                if header:
                    return header

        # Look for other title-like patterns
        for content in group_content:
            if len(content) < 100 and not content.endswith('.'):
                return content.strip()

        # Fallback to generic name
        return f"Section {group_idx + 1}"

    def _create_unit_based_boundaries(self, units: List[Dict[str, Any]], 
                                    content: str) -> BoundaryDetectionResult:
        """
        Create simple boundaries based on natural units when LLM analysis fails.

        Args:
            units: List of text units
            content: Original content

        Returns:
            BoundaryDetectionResult
        """
        boundaries = []

        for i, unit in enumerate(units):
            concept = f"Unit {i+1}"
            unit_type = "content"
            
            # Check if it's a header (content-based detection)
            if unit['content'].startswith('#'):
                header_text = re.sub(r'^#+\s*', '', unit['content'].split('\n')[0]).strip()
                if header_text:
                    concept = header_text
                    unit_type = "header"
            # Check if it's title-like (short content without newlines)
            elif len(unit['content']) < 100 and '\n' not in unit['content']:
                concept = unit['content'][:50].strip()
                unit_type = "title"

            boundaries.append(ChunkBoundary(
                start_position=unit['start_pos'],
                end_position=unit['end_pos'],
                concept=concept,
                confidence=0.6,
                reasoning=f"Natural unit boundary ({unit_type})"
            ))

        return BoundaryDetectionResult(
            boundaries=boundaries,
            total_concepts=len(boundaries),
            confidence_score=0.6
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



    def _generate_chunk_id(self, content: str, concept: str) -> str:
        """
        Generate a unique ID for a chunk based on its content and concept.

        Args:
            content: The chunk content
            concept: The main concept

        Returns:
            Unique chunk identifier
        """
        return generate_chunk_id(content, concept)
