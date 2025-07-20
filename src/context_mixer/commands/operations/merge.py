import asyncio
import logging
from textwrap import dedent
from typing import List, Tuple

from mojentic.llm import LLMMessage

from context_mixer.commands.interactions.resolve_conflicts import resolve_conflicts, ConflictResolver
from context_mixer.domain.conflict import ConflictList, Conflict, ConflictingGuidance
from context_mixer.domain.llm_instructions import ingest_system_message, clean_prompt
from context_mixer.domain.knowledge import KnowledgeChunk
from context_mixer.domain.context_aware_prompts import ContextAwarePromptBuilder
from context_mixer.gateways.llm import LLMGateway


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


async def detect_conflicts_batch_with_clustering(chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]], 
                                              knowledge_store=None,
                                              llm_gateway: LLMGateway = None,
                                              batch_size: int = 5,
                                              progress_callback=None) -> List[Tuple[KnowledgeChunk, KnowledgeChunk, ConflictList]]:
    """
    Detect conflicts between multiple chunk pairs using clustering optimization when available.

    This function uses clustering optimization if a VectorKnowledgeStore with clustering
    enabled is provided, otherwise falls back to the standard batch processing.

    Args:
        chunk_pairs: List of tuples containing pairs of chunks to check for conflicts
        knowledge_store: Optional VectorKnowledgeStore with clustering support
        llm_gateway: The LLM gateway to use for detecting conflicts (required if no knowledge_store)
        batch_size: Number of conflict detections to process concurrently (default: 5)
        progress_callback: Optional callback function to report progress as batches complete

    Returns:
        List of tuples containing (chunk1, chunk2, conflicts) for each pair
    """
    # Check if we can use clustering optimization
    if (knowledge_store and 
        hasattr(knowledge_store, 'detect_conflicts_batch_with_clustering') and
        hasattr(knowledge_store, 'enable_clustering') and
        knowledge_store.enable_clustering):
        
        # Extract unique chunks from pairs
        unique_chunks = []
        chunk_ids_seen = set()
        
        for chunk1, chunk2 in chunk_pairs:
            if chunk1.id not in chunk_ids_seen:
                unique_chunks.append(chunk1)
                chunk_ids_seen.add(chunk1.id)
            if chunk2.id not in chunk_ids_seen:
                unique_chunks.append(chunk2)
                chunk_ids_seen.add(chunk2.id)
        
        try:
            # Use clustering-optimized batch conflict detection
            cluster_conflicts = await knowledge_store.detect_conflicts_batch_with_clustering(unique_chunks)
            
            # Convert clustering results to the expected format
            conflict_map = {}
            for chunk1, chunk2, has_conflict in cluster_conflicts:
                pair_key = (chunk1.id, chunk2.id)
                reverse_key = (chunk2.id, chunk1.id)
                
                if has_conflict:
                    # Create a basic conflict for the pair
                    conflict = Conflict(
                        description=f"Content conflict detected between chunks",
                        conflicting_guidance=[
                            ConflictingGuidance(content=chunk1.content, source=f"chunk {chunk1.id[:12]}..."),
                            ConflictingGuidance(content=chunk2.content, source=f"chunk {chunk2.id[:12]}...")
                        ]
                    )
                    conflicts = ConflictList(list=[conflict])
                else:
                    conflicts = ConflictList(list=[])
                
                conflict_map[pair_key] = conflicts
                conflict_map[reverse_key] = conflicts
            
            # Build results for the original chunk pairs
            results = []
            for chunk1, chunk2 in chunk_pairs:
                pair_key = (chunk1.id, chunk2.id)
                conflicts = conflict_map.get(pair_key, ConflictList(list=[]))
                results.append((chunk1, chunk2, conflicts))
                
                # Report progress if callback provided
                if progress_callback:
                    progress_callback(len(results))
            
            return results
            
        except Exception as e:
            # Fall back to standard batch processing if clustering fails
            logging.warning(f"Clustering-based conflict detection failed: {e}. "
                          f"Falling back to standard batch processing.")
    
    # Fall back to standard batch processing
    if not llm_gateway:
        raise ValueError("LLM gateway is required when clustering optimization is not available")
    
    return await detect_conflicts_batch(chunk_pairs, llm_gateway, batch_size, progress_callback)


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
            task = detect_conflicts_async(chunk1.content, chunk2.content, llm_gateway)
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
