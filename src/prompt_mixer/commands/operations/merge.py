from textwrap import dedent
from typing import List

from mojentic.llm import LLMMessage

from prompt_mixer.domain.conflict import Conflict, ConflictList
from prompt_mixer.gateways.llm import LLMGateway


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
    prompt = dedent(f"""
        Your task is to carefully analyze these two documents and identify any clearly conflicting
        guidance or information.

        Existing
        ```
        {existing_content}
        ```

        Incoming
        ```
        {new_content}
        ```

        Look for any contradictions or conflicts where the documents provide different 
        or opposing guidance on the same topic. Focus on finding a single, obvious conflict 
        where one document is recommending something very different or opposite from the other.

        Create a structured representation of any conflicts you find.
    """).strip()

    # Create an LLMMessage with the prompt
    message = LLMMessage(content=prompt)

    # Use the LLM to detect conflicts
    conflicts = llm_gateway.generate_object(messages=[message], object_model=ConflictList)
    return conflicts


def resolve_conflict(conflicts: List[Conflict], console) -> List[Conflict]:
    """
    Resolve conflicts by consulting the user.

    This function presents each conflict to the user and asks them to choose
    which guidance is correct.

    Args:
        conflicts: A list of Conflict objects to resolve
        console: Rich console for output

    Returns:
        The list of conflicts with resolutions set
    """
    # If conflicts is empty, return it as is
    if not conflicts:
        return []

    # Iterate through each conflict and resolve it
    for conflict in conflicts:
        console.print("\n[bold red]Conflict Detected![/bold red]")
        console.print(f"[bold]Description:[/bold] {conflict.description}")
        console.print("\n[bold]Conflicting Guidance:[/bold]")

        for i, guidance in enumerate(conflict.conflicting_guidance):
            console.print(f"\n[bold]{i + 1}. From {guidance.source}:[/bold]")
            console.print(f"{guidance.content}")

        # Ask the user to choose which guidance is correct
        while True:
            choice = console.input("\n[bold]Which guidance is correct? (Enter the number):[/bold] ")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(conflict.conflicting_guidance):
                    # Set the resolution to the chosen guidance
                    conflict.resolution = conflict.conflicting_guidance[choice_idx].content
                    break
                else:
                    console.print("[red]Invalid choice. Please enter a valid number.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")

    return conflicts


def merge_content(existing_content: str, new_content: str, llm_gateway: LLMGateway,
                  console=None) -> str:
    """
    Merge existing content with new content using an LLM to create a coherent document.

    This function first checks for conflicts between the existing and new content.
    If conflicts are detected, it consults the user to resolve them.
    Then it merges the content, incorporating the resolved conflicts.

    Args:
        existing_content: The existing content in context.md
        new_content: The new content to merge
        llm_gateway: The LLM gateway to use for generating the merged content
        console: Rich console for output (required if conflicts need to be resolved)

    Returns:
        The merged content
    """
    # First, detect any conflicts
    conflicts = detect_conflicts(existing_content, new_content, llm_gateway)

    # If conflicts are detected, resolve them
    resolved_conflicts = None
    if conflicts.list:
        if not console:
            raise ValueError("Console is required to resolve conflicts")

        resolved_conflicts = resolve_conflict(conflicts.list, console)

    # Create a prompt for the LLM to merge the content
    base_prompt = dedent(f"""
        Your task is to merge two documents into a single coherent document, accurately representing
        the content of both documents, and without adding anything extra.

        The first document contains existing content, and the second document contains new content
        to be merged.

        Please combine these documents, ensuring that:
        1. All unique information from both documents is preserved
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
    """).strip()

    # If there were resolved conflicts, include them in the prompt
    conflict_info = ""
    if resolved_conflicts:
        # Build a string with all resolved conflicts
        conflict_resolutions = []
        for conflict in resolved_conflicts:
            if conflict.resolution:
                conflict_resolutions.append(f"Description: {conflict.description}\nResolution: {conflict.resolution}")

        if conflict_resolutions:
            conflict_info = dedent(f"""
                Conflicts were detected between these documents and resolved as follows:
                ```
                {"\n\n".join(conflict_resolutions)}
                ```

                When merging the documents, use these resolutions to guide your work.
            """).strip()

    conclusion = dedent("""
        Please provide only the merged document as your response, without any additional commentary
        or wrappers.
    """).strip()

    # Combine all parts of the prompt
    prompt_parts = [base_prompt]
    if conflict_info:
        prompt_parts.append(conflict_info)
    prompt_parts.append(conclusion)

    prompt = "\n\n".join(prompt_parts)

    # Create an LLMMessage with the prompt
    message = LLMMessage(content=prompt)

    # Generate the merged content using the LLM gateway
    response = llm_gateway.generate(messages=[message])

    # Return the generated content
    return response
