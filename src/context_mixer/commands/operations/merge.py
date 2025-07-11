from textwrap import dedent

from mojentic.llm import LLMMessage

from context_mixer.commands.interactions.resolve_conflicts import resolve_conflicts, ConflictResolver
from context_mixer.domain.conflict import ConflictList
from context_mixer.domain.llm_instructions import ingest_system_message, clean_prompt
from context_mixer.gateways.llm import LLMGateway


def detect_conflicts(existing_content: str,
                     new_content: str,
                     llm_gateway: LLMGateway) -> ConflictList:
    """
    Detect conflicts between existing content and new content.

    This function uses the LLM to detect conflicts between the existing content and new content.
    If a conflict is detected, it returns a Conflict object. Otherwise, it returns None.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for detecting conflicts

    Returns:
        A Conflict object if a conflict is detected, None otherwise
    """
    # Create a prompt for the LLM to detect conflicts
    prompt = clean_prompt(f"""
        Your task is to carefully analyze these two documents and identify ONLY genuine conflicts
        where they provide contradictory guidance on the same specific topic.

        Existing
        ```
        {existing_content}
        ```

        Incoming
        ```
        {new_content}
        ```

        A CONFLICT exists ONLY when:
        1. Both documents address the SAME specific topic or rule
        2. They provide CONTRADICTORY or MUTUALLY EXCLUSIVE guidance
        3. Following both pieces of guidance would be impossible or inconsistent

        Examples of REAL conflicts:
        - Different formatting rules for the same code element
        - Contradictory performance recommendations for the same operation
        - Mutually exclusive architectural patterns for the same component

        Examples of NOT conflicts (complementary information):
        - A header/title and its content details
        - General guidance and specific implementation details
        - Different rules for different contexts (e.g., "camelCase for variables" AND "PascalCase for classes" - these are DIFFERENT contexts)
        - Different naming conventions for different code elements (variables, classes, functions, etc.)
        - One document being more detailed than another on the same topic
        - Multiple related but distinct rules that can all be followed simultaneously

        IMPORTANT: If the documents are complementary (one provides headers/structure, 
        the other provides details), or if they address different aspects of the same 
        domain, this is NOT a conflict.

        Only create conflict entries for genuine contradictions where both documents 
        give opposing instructions for the exact same thing.
    """)

    messages = [
        ingest_system_message,
        LLMMessage(content=prompt)
    ]

    # Use the LLM to detect conflicts
    conflicts = llm_gateway.generate_object(messages=messages, object_model=ConflictList)
    return conflicts


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

    # Create a prompt for the LLM to merge the content
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
    conflict_info = ""
    if resolved_conflicts:
        # Build a string with all resolved conflicts
        conflict_resolutions = []
        for conflict in resolved_conflicts:
            if conflict.resolution:
                conflict_resolutions.append(f"Description: {conflict.description}\nResolution: {conflict.resolution}")

        if conflict_resolutions:
            conflict_info = clean_prompt(f"""
                Conflicts were detected between these documents and resolved as follows:
                ```
                {"\n\n".join(conflict_resolutions)}
                ```

                When merging the documents, use these resolutions to guide your work.
            """)

    conclusion = clean_prompt("""
        Please provide only the merged document as your response, without any additional commentary
        or wrappers.
    """)

    prompt_parts = [base_prompt]
    if conflict_info:
        prompt_parts.append(conflict_info)
    prompt_parts.append(conclusion)

    prompt = "\n\n".join(prompt_parts)

    messages = [
        ingest_system_message,
        LLMMessage(content=prompt)
    ]

    response = llm_gateway.generate(messages=messages)

    return response
