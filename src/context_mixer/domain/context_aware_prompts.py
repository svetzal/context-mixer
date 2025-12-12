from textwrap import dedent
from typing import List, Set

from .context import ContextType, ContextAnalysis
from .context_detection import ContextDetectionEngine


class ContextAwarePromptBuilder:
    """
    Builds context-aware prompts for conflict detection.

    This class generates dynamic prompts that include context-specific
    examples and guidance, helping the LLM better understand when
    rules apply to different contexts and should not be considered conflicting.
    """

    def __init__(self, context_engine: ContextDetectionEngine = None):
        """
        Initialize the context-aware prompt builder.

        Args:
            context_engine: Context detection engine to use
        """
        self.context_engine = context_engine or ContextDetectionEngine()

        # Context-specific examples for different types
        self.context_examples = {
            ContextType.ARCHITECTURAL: {
                'examples': [
                    '"Don\'t test gateways" AND "Write tests for business logic" - these apply to DIFFERENT architectural layers',
                    '"Repositories use integration tests" AND "Services use unit tests" - different component testing strategies',
                    '"Controllers handle HTTP" AND "Services contain business logic" - different layer responsibilities'
                ],
                'guidance': 'Rules that apply to specific architectural layers or components are NOT conflicting'
            },
            ContextType.PLATFORM: {
                'examples': [
                    '"Use localStorage for storage" AND "Use AsyncStorage for storage" - web vs mobile platforms',
                    '"Use DOM manipulation" AND "Use native APIs" - web vs native platforms',
                    '"Use HTTP requests" AND "Use native networking" - different platform capabilities'
                ],
                'guidance': 'Rules that apply to different platforms (web, mobile, desktop) are NOT conflicting'
            },
            ContextType.ENVIRONMENT: {
                'examples': [
                    '"Enable debug logging" AND "Disable logging for performance" - development vs production',
                    '"Use mock data" AND "Use real APIs" - testing vs production environments',
                    '"Allow CORS from anywhere" AND "Restrict CORS origins" - development vs production security'
                ],
                'guidance': 'Rules that apply to different environments (dev, staging, prod) are NOT conflicting'
            },
            ContextType.LANGUAGE: {
                'examples': [
                    '"Use camelCase" AND "Use snake_case" - JavaScript vs Python naming conventions',
                    '"Use interfaces" AND "Use protocols" - different language constructs for same concept',
                    '"Use npm" AND "Use pip" - different package managers for different languages'
                ],
                'guidance': 'Rules that apply to different programming languages are NOT conflicting'
            },
            ContextType.FRAMEWORK: {
                'examples': [
                    '"Use useState" AND "Use reactive data" - React vs Vue state management',
                    '"Use decorators" AND "Use middleware" - different framework patterns',
                    '"Use dependency injection" AND "Use service locator" - different framework approaches'
                ],
                'guidance': 'Rules that apply to different frameworks are NOT conflicting'
            },
            ContextType.TEAM: {
                'examples': [
                    '"Use Jest for testing" AND "Use Vitest for testing" - different team preferences',
                    '"Use tabs for indentation" AND "Use spaces for indentation" - different team standards',
                    '"Use TypeScript" AND "Use JavaScript" - different team technology choices'
                ],
                'guidance': 'Rules that apply to different teams or projects are NOT conflicting'
            }
        }

    def build_conflict_detection_prompt(self, existing_content: str, new_content: str) -> str:
        """
        Build a context-aware conflict detection prompt.

        Args:
            existing_content: The existing content to compare
            new_content: The new content to compare

        Returns:
            Context-aware prompt for conflict detection
        """
        # Detect contexts in both contents
        existing_analysis = self.context_engine.detect_contexts(existing_content)
        new_analysis = self.context_engine.detect_contexts(new_content)

        # Build the base prompt
        base_prompt = self._build_base_prompt(existing_content, new_content)

        # Add context-specific guidance
        context_guidance = self._build_context_guidance(existing_analysis, new_analysis)

        # Combine into final prompt
        return f"{base_prompt}\n\n{context_guidance}"

    def _build_base_prompt(self, existing_content: str, new_content: str) -> str:
        """Build the base conflict detection prompt."""
        return dedent(f"""
            Your task is to carefully analyze these two documents and identify ONLY genuine conflicts
            where they provide contradictory guidance on the same specific topic within the same context.

            Existing
            ```
            {existing_content}
            ```

            Incoming
            ```
            {new_content}
            ```

            IMPORTANT: If content is prefixed with "[Concept: X]", this indicates the architectural
            scope or domain the rule applies to. Rules with DIFFERENT concepts are NOT in conflict
            even if they seem contradictory at first glance.

            For example:
            - "[Concept: gateway]" rules apply ONLY to gateway/I/O boundary components
            - "[Concept: testing]" rules apply to GENERAL testing practices
            - A gateway-specific "don't test gateways" rule does NOT conflict with a general
              "write tests for new functionality" rule because they apply to DIFFERENT scopes

            A CONFLICT exists ONLY when:
            1. Both documents address the SAME specific topic or rule
            2. They provide CONTRADICTORY or MUTUALLY EXCLUSIVE guidance
            3. Following both pieces of guidance would be impossible or inconsistent
            4. They apply to the SAME context (same platform, environment, architectural layer, etc.)
            5. They have the SAME or overlapping concept scope (or no concept specified)

            Examples of REAL conflicts:
            - Different formatting rules for the same code element in the same language
            - Contradictory performance recommendations for the same operation in the same environment
            - Mutually exclusive architectural patterns for the same component type

            Examples of NOT conflicts (different scopes):
            - "[Concept: gateway] Don't test gateways" vs "[Concept: testing] Write tests for new functionality"
              These are COMPLEMENTARY - general testing applies, but gateways are a specific exception
            - "[Concept: repository] Use SQL" vs "[Concept: cache] Use Redis"
              These apply to different architectural components
        """).strip()

    def _build_context_guidance(self, existing_analysis: ContextAnalysis, new_analysis: ContextAnalysis) -> str:
        """
        Build context-specific guidance based on detected contexts.

        Args:
            existing_analysis: Context analysis for existing content
            new_analysis: Context analysis for new content

        Returns:
            Context-specific guidance text
        """
        # Collect all detected context types
        all_context_types = set()
        for context in existing_analysis.detected_contexts + new_analysis.detected_contexts:
            all_context_types.add(context.type)

        # Build context-specific examples and guidance
        context_sections = []

        # Add examples for detected context types
        for context_type in all_context_types:
            if context_type in self.context_examples:
                examples = self.context_examples[context_type]['examples']
                guidance = self.context_examples[context_type]['guidance']

                section = f"""
                **{context_type.value.title()} Context Rules**:
                {guidance}
                Examples: {'; '.join(examples)}
                """
                context_sections.append(section.strip())

        # Add general context guidance
        general_guidance = """
        Examples of NOT conflicts (complementary information):
        - A header/title and its content details
        - General guidance and specific implementation details
        - Different rules for different contexts (platforms, environments, languages, etc.)
        - Different naming conventions for different code elements (variables, classes, functions, etc.)
        - One document being more detailed than another on the same topic
        - Multiple related but distinct rules that can all be followed simultaneously
        """

        # Build the final context guidance
        if context_sections:
            detected_contexts_text = self._format_detected_contexts(existing_analysis, new_analysis)
            context_guidance = f"""
            DETECTED CONTEXTS:
            {detected_contexts_text}

            CONTEXT-SPECIFIC CONSIDERATIONS:
            {chr(10).join(context_sections)}

            GENERAL GUIDANCE:
            {general_guidance.strip()}

            IMPORTANT: If the documents apply to different contexts (different platforms, environments,
            architectural layers, languages, etc.), they are likely NOT conflicting even if they
            appear to give different guidance.

            Only create conflict entries for genuine contradictions where both documents
            give opposing instructions for the exact same thing within the same context.
            """
        else:
            context_guidance = f"""
            GENERAL GUIDANCE:
            {general_guidance.strip()}

            IMPORTANT: If the documents are complementary (one provides headers/structure,
            the other provides details), or if they address different aspects of the same
            domain, this is NOT a conflict.

            Only create conflict entries for genuine contradictions where both documents
            give opposing instructions for the exact same thing.
            """

        return context_guidance.strip()

    def _format_detected_contexts(self, existing_analysis: ContextAnalysis, new_analysis: ContextAnalysis) -> str:
        """Format detected contexts for display in the prompt."""
        lines = []

        if existing_analysis.detected_contexts:
            existing_contexts = [str(ctx) for ctx in existing_analysis.detected_contexts]
            lines.append(f"Existing content contexts: {', '.join(existing_contexts)}")

        if new_analysis.detected_contexts:
            new_contexts = [str(ctx) for ctx in new_analysis.detected_contexts]
            lines.append(f"New content contexts: {', '.join(new_contexts)}")

        if not lines:
            lines.append("No specific contexts detected - apply general conflict detection rules")

        return '\n'.join(lines)

    def build_context_analysis_prompt(self, content: str) -> str:
        """
        Build a prompt for analyzing contexts in content.

        This can be used with an LLM to detect contexts that the
        rule-based detectors might miss.

        Args:
            content: The content to analyze

        Returns:
            Prompt for context analysis
        """
        return dedent(f"""
            Analyze the following content and identify the contexts where the rules or guidance apply.

            Content:
            ```
            {content}
            ```

            Please identify contexts such as:
            - Architectural contexts (gateway, service, repository, controller, etc.)
            - Platform contexts (web, mobile, desktop, server, cloud, etc.)
            - Environment contexts (development, staging, production, etc.)
            - Language contexts (Python, JavaScript, Java, etc.)
            - Framework contexts (React, Vue, Django, Spring, etc.)
            - Team contexts (frontend team, backend team, etc.)
            - Any other relevant contexts

            For each context you identify, provide:
            1. Context type (architectural, platform, environment, etc.)
            2. Context value (specific technology, pattern, or scope)
            3. Confidence level (high, medium, low)
            4. Brief explanation of why this context applies

            Focus on contexts that would help distinguish when rules might appear conflicting
            but actually apply to different situations.
        """).strip()

    def add_context_examples(self, context_type: ContextType, examples: List[str], guidance: str):
        """
        Add custom context examples for a specific context type.

        Args:
            context_type: The type of context
            examples: List of example scenarios
            guidance: General guidance for this context type
        """
        self.context_examples[context_type] = {
            'examples': examples,
            'guidance': guidance
        }

    def get_supported_context_types(self) -> Set[ContextType]:
        """Get all supported context types."""
        return self.context_engine.get_supported_types()