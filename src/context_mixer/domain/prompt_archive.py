"""
Archive of contexts previously used in the context-mixer application.

This module contains contexts that were previously used for extracting facets
from monolithic content. These are kept for reference purposes only and are
no longer actively used in the application.
"""

from textwrap import dedent

# Archive of facet extraction contexts
FACET_EXTRACTION_CONTEXTS = {
    "PURPOSE": dedent("""
        Extract the PURPOSE section from the following monolithic context.

        The PURPOSE should describe the overall goal or objective of the context.
        It should answer questions like:
        - What is this context designed to do?
        - What is the intended outcome when using this context?
        - Who is the target audience or user?

        Return only the extracted PURPOSE content without any additional commentary.

        {content}
    """).strip(),

    "PRINCIPLES": dedent("""
        Extract the PRINCIPLES section from the following monolithic context.

        The PRINCIPLES should describe the key guidelines or values that inform the context.
        It should answer questions like:
        - What are the core values or guidelines that shape this context?
        - What principles should be followed when using this context?
        - What are the fundamental beliefs or assumptions underlying this context?

        Return only the extracted PRINCIPLES content without any additional commentary.

        {content}
    """).strip(),

    "RULES": dedent("""
        Extract the RULES section from the following monolithic context.

        The RULES should describe specific constraints or requirements for the context.
        It should answer questions like:
        - What specific constraints must be followed?
        - What are the hard requirements or limitations?
        - What should never be done when using this context?
        - What must always be included or excluded?

        Return only the extracted RULES content without any additional commentary.

        {content}
    """).strip(),

    "PROCESSES": dedent("""
        Extract the PROCESSES section from the following monolithic context.

        The PROCESSES should describe step-by-step procedures or workflows for the context.
        It should answer questions like:
        - What are the specific steps to follow in this process?
        - How should the process be structured or organized?
        - What methodology should be applied?
        - When should the process be followed?

        Return only the extracted PROCESSES content without any additional commentary.

        {content}
    """).strip()
}

# Archive of category extraction context
CATEGORIES_CONTEXT_TEMPLATE = dedent("""
    Examine the following content, and extract any categories you can see
    (eg from the headings). Organize those categories under the following
    facets: purpose, principles, rules, and processes.

    The purpose should describe the overall goal or objective of the context.
    It should answer questions like:
    - What is this context designed to do?
    - What is the intended outcome when using this context?
    - Who is the target audience or user?

    The principles should describe the key guidelines or values that inform the context.
    It should answer questions like:
    - What are the core values or guidelines that shape this context?
    - What principles should be followed when using this context?
    - What are the fundamental beliefs or assumptions underlying this context?

    The rules should describe specific constraints or requirements for the context.
    It should answer questions like:
    - What specific constraints must be followed?
    - What are the hard requirements or limitations?
    - What should never be done when using this context?
    - What must always be included or excluded?

    The processes should describe step-by-step procedures or workflows for the context.
    It should answer questions like:
    - What are the specific steps to follow in this process?
    - How should the process be structured or organized?
    - What methodology should be applied?
    - When should the process be followed?

    You will need to reduce the categories you sense to a single lowercase word.
    If it's unclear what single word would define it, you may need a further
    category decomposition, like rules > python > testing.

    Try and stick to a breakdown like:
    - facet > technology > subtechnology
    - facet > technology > practice

    Something that pertains to controlling how testing is done in python
    would break down to:
    - rules > python > testing

    Something that pertains to controlling how testing is done in typescript
    would break down to:
    - rules > typescript > testing
""").strip()

# Archive of content extraction context
CONTENT_EXTRACTION_CONTEXT_TEMPLATE = """
    Your job is to examine the attached document and compose a new document
    containing only a specific aspect of the original content.

    The full set of categorized documents we are extracting will be:
    {filenames}

    Try not to extract duplicate content for the single category you are extracting.

    Original Document:
    ```
    {content}
    ```
    Extract content as a markdown document (raw content only, no code-fences) for
    the specific category filename {category_filename}.
"""
