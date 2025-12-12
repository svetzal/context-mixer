import asyncio
import logging
from typing import List, Tuple, Optional

import numpy as np
from mojentic.llm import LLMMessage
from pydantic import BaseModel, Field

from context_mixer.commands.interactions.resolve_conflicts import resolve_conflicts, \
    ConflictResolver
from context_mixer.domain.conflict import ConflictList, Conflict, ConflictingGuidance
from context_mixer.domain.context_aware_prompts import ContextAwarePromptBuilder
from context_mixer.domain.knowledge import KnowledgeChunk
from context_mixer.domain.llm_instructions import ingest_system_message, clean_prompt
from context_mixer.gateways.llm import LLMGateway

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Batched Conflict Detection
# ============================================================================

class PairConflictResult(BaseModel):
    """Conflict result for a single pair in multi-pair analysis."""
    pair_index: int = Field(..., description="Index of the pair in the batch")
    has_conflict: bool = Field(..., description="Whether conflicts were detected")
    conflicts: List[Conflict] = Field(default_factory=list, description="List of detected conflicts")


class MultiPairConflictResult(BaseModel):
    """Result of multi-pair conflict analysis."""
    pair_results: List[PairConflictResult] = Field(..., description="Results for each analyzed pair")


# ============================================================================
# Phase 1: Embedding Similarity Filter
# ============================================================================

def filter_pairs_by_embedding_similarity(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]],
    similarity_threshold: float = 0.70
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk]]:
    """
    Filter chunk pairs to only those with high embedding similarity.

    Conflicts can only exist between semantically similar chunks.
    Low similarity pairs are guaranteed non-conflicts.

    Args:
        chunk_pairs: All candidate pairs to filter
        similarity_threshold: Minimum cosine similarity to consider (default 0.70)

    Returns:
        Filtered list of pairs worth checking with LLM
    """
    if not chunk_pairs:
        return []

    filtered_pairs = []
    pairs_without_embeddings = []

    for chunk1, chunk2 in chunk_pairs:
        # If either chunk lacks embeddings, pass through to LLM (safety)
        if chunk1.embedding is None or chunk2.embedding is None:
            pairs_without_embeddings.append((chunk1, chunk2))
            continue

        # Compute cosine similarity using NumPy
        vec1 = np.array(chunk1.embedding)
        vec2 = np.array(chunk2.embedding)

        # Cosine similarity = dot product / (norm1 * norm2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            pairs_without_embeddings.append((chunk1, chunk2))
            continue

        similarity = dot_product / (norm1 * norm2)

        # Only keep pairs above threshold
        if similarity >= similarity_threshold:
            filtered_pairs.append((chunk1, chunk2))

    # Include pairs without embeddings (safety: let LLM decide)
    filtered_pairs.extend(pairs_without_embeddings)

    logger.info(f"Embedding filter: {len(chunk_pairs)} → {len(filtered_pairs)} pairs (threshold={similarity_threshold})")
    return filtered_pairs


# ============================================================================
# Phase 2: Domain/Concept Partitioning
# ============================================================================

def filter_pairs_by_metadata(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]]
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk]]:
    """
    Filter pairs to only those with overlapping domains or matching concepts.

    Chunks in completely different domains cannot conflict.

    Args:
        chunk_pairs: Pairs to filter by metadata

    Returns:
        Filtered list of pairs with potential for conflict
    """
    if not chunk_pairs:
        return []

    filtered_pairs = []

    for chunk1, chunk2 in chunk_pairs:
        # Safety: if either chunk has no metadata, pass through to LLM
        if not hasattr(chunk1, 'metadata') or not hasattr(chunk2, 'metadata'):
            filtered_pairs.append((chunk1, chunk2))
            continue

        # Check for matching concepts - if concepts match exactly, always compare
        if chunk1.concept and chunk2.concept and chunk1.concept == chunk2.concept:
            filtered_pairs.append((chunk1, chunk2))
            continue

        # If concepts are present but different, check if they're related
        # Different concepts (e.g., "docstring-requirements" vs "test-conventions")
        # likely apply to different scopes and shouldn't be compared unless domains overlap
        if chunk1.concept and chunk2.concept and chunk1.concept != chunk2.concept:
            # Only compare if they share domains AND have related keywords
            domains1 = set(chunk1.metadata.domains) if chunk1.metadata.domains else set()
            domains2 = set(chunk2.metadata.domains) if chunk2.metadata.domains else set()

            # Check for domain overlap
            if domains1 and domains2 and domains1.intersection(domains2):
                # Check if concepts are semantically related (share keywords)
                concept1_words = set(chunk1.concept.lower().split('-'))
                concept2_words = set(chunk2.concept.lower().split('-'))

                # If concepts share significant keywords, they might conflict
                if concept1_words.intersection(concept2_words):
                    filtered_pairs.append((chunk1, chunk2))
                # Otherwise, different concepts in same domain are likely complementary, not conflicting
            continue

        # Check for overlapping domains when concepts aren't both present
        domains1 = set(chunk1.metadata.domains) if chunk1.metadata.domains else set()
        domains2 = set(chunk2.metadata.domains) if chunk2.metadata.domains else set()

        # If either has no domains, pass through (safety)
        if not domains1 or not domains2:
            filtered_pairs.append((chunk1, chunk2))
            continue

        # Keep if domains overlap
        if domains1.intersection(domains2):
            filtered_pairs.append((chunk1, chunk2))

    logger.info(f"Metadata filter: {len(chunk_pairs)} → {len(filtered_pairs)} pairs")
    return filtered_pairs


# ============================================================================
# Phase 3: Batched LLM Analysis
# ============================================================================

async def detect_conflicts_multi_pair(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]],
    llm_gateway: LLMGateway,
    pairs_per_batch: int = 10,
    progress_callback=None
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk, ConflictList]]:
    """
    Detect conflicts for multiple pairs in a single LLM call (batched).

    This dramatically reduces API calls by analyzing multiple pairs per request.

    Args:
        chunk_pairs: Pairs to analyze
        llm_gateway: LLM gateway
        pairs_per_batch: Number of pairs per LLM call (default 10)
        progress_callback: Optional callback function to report progress

    Returns:
        Results for each pair as (chunk1, chunk2, ConflictList)
    """
    if not chunk_pairs:
        return []

    results = []
    total_processed = 0

    # Process in batches
    for batch_start in range(0, len(chunk_pairs), pairs_per_batch):
        batch = chunk_pairs[batch_start:batch_start + pairs_per_batch]

        # Build prompt with all pairs in this batch
        pair_descriptions = []
        for i, (chunk1, chunk2) in enumerate(batch):
            # Build rich metadata descriptions
            def build_metadata_description(chunk):
                parts = []
                if chunk.concept:
                    parts.append(f"Concept: {chunk.concept}")
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    if chunk.metadata.domains:
                        parts.append(f"Domains: {', '.join(chunk.metadata.domains)}")
                    if chunk.metadata.scope:
                        parts.append(f"Scope: {', '.join(chunk.metadata.scope)}")
                return "[" + " | ".join(parts) + "]"

            metadata1 = build_metadata_description(chunk1)
            metadata2 = build_metadata_description(chunk2)

            content1 = f"{metadata1}\n{chunk1.content}"
            content2 = f"{metadata2}\n{chunk2.content}"

            pair_descriptions.append(f"""
PAIR {i}:
Content A:
{content1}

Content B:
{content2}
""")

        prompt = clean_prompt(f"""
Analyze the following pairs of content for conflicts.
For each pair, determine if they contain contradictory or incompatible guidance.

**CRITICAL: Understanding Concepts and Scope**

Each chunk has metadata that indicates its CONCEPT and SCOPE:
- **Concept**: The specific topic or area this rule applies to (e.g., "docstring-requirements", "test-conventions", "gateway-testing")
- **Domains**: The knowledge domains (technical, business, design, etc.)
- **Scope**: Where this rule applies (general, enterprise, mobile-only, etc.)

**Different concepts usually mean different scopes - NOT conflicts!**

Examples of NON-CONFLICTS due to different scopes:
- [Concept: docstring-requirements] "Use docstrings for all functions"
  vs [Concept: test-conventions] "Don't use docstrings in test fixtures"
  → These apply to DIFFERENT contexts (general code vs test fixtures) - COMPLEMENTARY, not conflicting!

- [Concept: gateway-testing] "Don't test gateway logic"
  vs [Concept: testing-guidelines] "Write tests for all new functionality"
  → General rule applies, gateways are a specific exception - COMPLEMENTARY!

- [Concept: indentation] "Use 4 spaces"
  vs [Concept: line-length] "Maximum 100 characters"
  → Different aspects of code style - can both be followed - COMPLEMENTARY!

A REAL conflict exists when:
1. Both chunks address the SAME specific topic (same or overlapping concepts)
2. They provide CONTRADICTORY guidance (can't follow both)
3. Following both would be impossible or inconsistent
4. They apply to the SAME scope/context

Examples of REAL conflicts:
- [Concept: indentation] "Use 4 spaces" vs [Concept: indentation] "Use 2 spaces"
- [Concept: error-handling] "Always throw exceptions" vs [Concept: error-handling] "Never use exceptions"
- [Concept: api-versioning] "Use URL versioning" vs [Concept: api-versioning] "Use header versioning"

NOT conflicts when:
- Different concepts (different topics/scopes) - these are COMPLEMENTARY
- Different domains (one technical, one business) - these are COMPLEMENTARY
- Different detail levels on same topic - these are COMPLEMENTARY
- One is general rule, other is specific exception - these are COMPLEMENTARY
- Both can be followed simultaneously - these are COMPATIBLE

{chr(10).join(pair_descriptions)}

For each pair, indicate whether conflicts exist and describe them.
Be especially careful with pairs that have different concepts - they are usually complementary, not conflicting.
""")

        messages = [
            ingest_system_message,
            LLMMessage(content=prompt)
        ]

        try:
            # Make single LLM call for entire batch
            batch_result = await asyncio.to_thread(
                llm_gateway.generate_object,
                messages=messages,
                object_model=MultiPairConflictResult
            )

            # Convert results to expected format
            for i, (chunk1, chunk2) in enumerate(batch):
                # Find result for this pair
                pair_result = next(
                    (pr for pr in batch_result.pair_results if pr.pair_index == i),
                    None
                )

                if pair_result and pair_result.has_conflict:
                    conflict_list = ConflictList(list=pair_result.conflicts)
                else:
                    conflict_list = ConflictList(list=[])

                results.append((chunk1, chunk2, conflict_list))

        except Exception as e:
            logger.warning(f"Batch conflict detection failed for batch starting at {batch_start}, falling back to individual detection: {e}")
            # Fallback: process pairs individually if batch fails
            for chunk1, chunk2 in batch:
                try:
                    conflicts = await detect_conflicts_async(chunk1.content, chunk2.content, llm_gateway)
                    results.append((chunk1, chunk2, conflicts))
                except Exception as inner_e:
                    logger.warning(f"Individual conflict detection also failed: {inner_e}")
                    results.append((chunk1, chunk2, ConflictList(list=[])))

        total_processed += len(batch)
        if progress_callback:
            progress_callback(total_processed)

    return results


# ============================================================================
# Original Conflict Detection Functions
# ============================================================================


def detect_conflicts(existing_content: str,
                     new_content: str,
                     llm_gateway: LLMGateway,
                     prompt_builder: ContextAwarePromptBuilder = None) -> ConflictList:
    """
    Detect conflicts between existing content and new content using context-aware analysis.

    This function uses the LLM to detect conflicts between the existing content and new content,
    with enhanced contextual awareness to reduce false positives.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for detecting conflicts
        prompt_builder: Optional context-aware prompt builder (creates default if None)

    Returns:
        A ConflictList object containing any detected conflicts
    """
    # Create context-aware prompt builder if not provided
    if prompt_builder is None:
        prompt_builder = ContextAwarePromptBuilder()

    # Build context-aware prompt
    prompt = clean_prompt(prompt_builder.build_conflict_detection_prompt(existing_content, new_content))

    messages = [
        ingest_system_message,
        LLMMessage(content=prompt)
    ]

    # Use the LLM to detect conflicts
    conflicts = llm_gateway.generate_object(messages=messages, object_model=ConflictList)
    return conflicts


async def detect_conflicts_async(existing_content: str,
                                new_content: str,
                                llm_gateway: LLMGateway) -> ConflictList:
    """
    Async version of detect_conflicts for batch processing.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for detecting conflicts

    Returns:
        A ConflictList object containing any detected conflicts
    """
    # Run the synchronous detect_conflicts in a thread to avoid blocking
    return await asyncio.to_thread(detect_conflicts, existing_content, new_content, llm_gateway)


async def detect_conflicts_batch(chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]],
                                llm_gateway: LLMGateway,
                                batch_size: int = 5,
                                progress_callback=None) -> List[Tuple[KnowledgeChunk, KnowledgeChunk, ConflictList]]:
    """
    Detect conflicts between multiple chunk pairs in parallel batches.

    This function processes chunk pairs in batches to improve performance while
    avoiding overwhelming the LLM service with too many concurrent requests.

    Args:
        chunk_pairs: List of tuples containing pairs of chunks to check for conflicts
        llm_gateway: The LLM gateway to use for detecting conflicts
        batch_size: Number of conflict detections to process concurrently (default: 5)
        progress_callback: Optional callback function to report progress as batches complete

    Returns:
        List of tuples containing (chunk1, chunk2, conflicts) for each pair
    """
    results = []

    # Process chunk pairs in batches
    for i in range(0, len(chunk_pairs), batch_size):
        batch = chunk_pairs[i:i + batch_size]

        # Create async tasks for this batch
        tasks = []
        for chunk1, chunk2 in batch:
            # Include concept metadata to help LLM understand architectural scope
            content1 = f"[Concept: {chunk1.concept}]\n{chunk1.content}" if chunk1.concept else chunk1.content
            content2 = f"[Concept: {chunk2.concept}]\n{chunk2.content}" if chunk2.concept else chunk2.content
            task = detect_conflicts_async(content1, content2, llm_gateway)
            tasks.append((chunk1, chunk2, task))

        # Execute all tasks in this batch concurrently using asyncio.gather
        task_list = [task for _, _, task in tasks]
        try:
            # Run all tasks concurrently and wait for all to complete
            conflict_results = await asyncio.gather(*task_list, return_exceptions=True)

            # Build batch results, handling any exceptions
            batch_results = []
            for i, (chunk1, chunk2, _) in enumerate(tasks):
                result = conflict_results[i]
                if isinstance(result, Exception):
                    # If conflict detection fails for a pair, create an empty conflict list
                    empty_conflicts = ConflictList(list=[])
                    batch_results.append((chunk1, chunk2, empty_conflicts))
                else:
                    batch_results.append((chunk1, chunk2, result))
        except Exception as e:
            # Fallback: if gather fails entirely, create empty results for all tasks
            batch_results = []
            for chunk1, chunk2, _ in tasks:
                empty_conflicts = ConflictList(list=[])
                batch_results.append((chunk1, chunk2, empty_conflicts))

        results.extend(batch_results)

        # Call progress callback if provided
        if progress_callback:
            progress_callback(len(results))

    return results


def format_conflict_resolutions(resolved_conflicts) -> str:
    """
    Pure function to format resolved conflicts into a prompt string.

    Args:
        resolved_conflicts: List of resolved Conflict objects, or None

    Returns:
        Formatted string for inclusion in LLM prompt, or empty string if no conflicts
    """
    if not resolved_conflicts:
        return ""

    # Build a string with all resolved conflicts
    conflict_resolutions = []
    for conflict in resolved_conflicts:
        if conflict.resolution:
            conflict_resolutions.append(f"Description: {conflict.description}\nResolution: {conflict.resolution}")
        elif conflict.resolution is None:
            # Handle "This is not a conflict" case - both pieces of guidance should be preserved
            guidance_list = []
            for guidance in conflict.conflicting_guidance:
                guidance_list.append(f"- {guidance.content} (from {guidance.source})")

            conflict_resolutions.append(f"Description: {conflict.description}\nResolution: This is not a conflict - both pieces of guidance are acceptable:\n" + "\n".join(guidance_list))

    if not conflict_resolutions:
        return ""

    return clean_prompt(f"""
        Conflicts were detected between these documents and resolved as follows:
        ```
        {"\n\n".join(conflict_resolutions)}
        ```

        When merging the documents, use these resolutions to guide your work.
    """)


def build_merge_prompt(existing_content: str, new_content: str, resolved_conflicts=None) -> str:
    """
    Pure function to build the merge prompt for LLM processing.

    Args:
        existing_content: The existing content to merge
        new_content: The new content to merge
        resolved_conflicts: Optional list of resolved conflicts

    Returns:
        Complete prompt string for LLM
    """
    base_prompt = clean_prompt(f"""
        Your task is to merge two documents into a single coherent document, accurately representing
        the content of both documents, and without adding anything extra.

        The first document contains existing content, and the second document contains new content
        to be merged.

        Please combine these documents, ensuring that:
        1. All detail and unique information from both documents is preserved
        2. Duplicate information appears only once
        3. The resulting document is well-structured and coherent
        4. Related information is grouped together logically

        Document 1 (Existing Content):
        ```
        {existing_content}
        ```

        Document 2 (New Content):
        ```
        {new_content}
        ```
    """)

    # If there were resolved conflicts, include them in the prompt
    conflict_info = format_conflict_resolutions(resolved_conflicts)

    conclusion = clean_prompt("""
        Please provide only the merged document as your response, without any additional commentary
        or wrappers.
    """)

    prompt_parts = [base_prompt]
    if conflict_info:
        prompt_parts.append(conflict_info)
    prompt_parts.append(conclusion)

    return "\n\n".join(prompt_parts)


def merge_content(existing_content: str, new_content: str, llm_gateway: LLMGateway,
                  console=None, resolver: ConflictResolver = None) -> str:
    """
    Merge existing content with new content using an LLM to create a coherent document.

    This function first checks for conflicts between the existing and new content.
    If conflicts are detected, it resolves them either interactively or using an automated resolver.
    Then it merges the content, incorporating the resolved conflicts.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for generating the merged content
        console: Rich console for output (required if conflicts need to be resolved interactively)
        resolver: Optional automated conflict resolver. If provided, conflicts will be
                 resolved automatically without user input.

    Returns:
        The merged content
    """
    # First, detect any conflicts
    conflicts = detect_conflicts(existing_content, new_content, llm_gateway)

    # If conflicts are detected, resolve them
    resolved_conflicts = None
    if conflicts.list:
        if not console and not resolver:
            raise ValueError("Console or resolver is required to resolve conflicts")

        resolved_conflicts = resolve_conflicts(conflicts.list, console, resolver)

    # Create a prompt for the LLM to merge the content using the pure function
    prompt = build_merge_prompt(existing_content, new_content, resolved_conflicts)

    messages = [
        ingest_system_message,
        LLMMessage(content=prompt)
    ]

    response = llm_gateway.generate(messages=messages)

    return response
