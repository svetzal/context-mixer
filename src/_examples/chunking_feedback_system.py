#!/usr/bin/env python3
"""
Feedback system for iteratively improving chunking prompts based on failure analysis.
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.gateways.llm import LLMGateway
from mojentic.llm import LLMMessage, MessageRole


class ChunkingFeedbackSystem:
    """System for analyzing chunking failures and improving prompts iteratively."""

    def __init__(self, llm_gateway: LLMGateway):
        self.llm_gateway = llm_gateway
        self.chunking_engine = ChunkingEngine(llm_gateway)
        self.iteration_history = []

    def analyze_failure_patterns(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns to identify common issues."""

        # Categorize failures
        failure_categories = {
            'title_only': [],
            'too_brief': [],
            'incomplete_explanation': [],
            'lacks_context': [],
            'other': []
        }

        for failure in failures:
            reason = failure['reason'].lower()
            issues = [issue.lower() for issue in failure.get('issues', [])]

            if 'title' in reason or 'heading' in reason:
                failure_categories['title_only'].append(failure)
            elif 'brief' in reason or 'short' in reason or len(failure.get('content_preview', '')) < 50:
                failure_categories['too_brief'].append(failure)
            elif 'context' in reason or any('context' in issue for issue in issues):
                failure_categories['lacks_context'].append(failure)
            elif 'incomplete' in reason or 'lacks' in reason:
                failure_categories['incomplete_explanation'].append(failure)
            else:
                failure_categories['other'].append(failure)

        # Calculate percentages
        total_failures = len(failures)
        analysis = {
            'total_failures': total_failures,
            'categories': {}
        }

        for category, category_failures in failure_categories.items():
            if category_failures:
                percentage = (len(category_failures) / total_failures) * 100
                analysis['categories'][category] = {
                    'count': len(category_failures),
                    'percentage': percentage,
                    'examples': category_failures[:2]  # Keep first 2 examples
                }

        return analysis

    def generate_prompt_improvements(self, failure_analysis: Dict[str, Any], current_prompt: str) -> str:
        """Generate improved prompt based on failure analysis."""

        # Create improvement suggestions based on failure patterns
        improvement_messages = [
            LLMMessage(
                role=MessageRole.System,
                content="""You are an expert prompt engineer specializing in improving LLM prompts for content chunking.

Your task is to analyze the current chunking prompt and failure patterns, then suggest specific improvements to address the identified issues.

Focus on:
1. Making chunks more semantically complete and self-contained
2. Avoiding overly granular chunks (like title-only or single-sentence chunks)
3. Ensuring each chunk provides sufficient context to be understood independently
4. Balancing chunk size - not too small, not too large

Provide a revised prompt that addresses the specific failure patterns identified."""
            ),
            LLMMessage(
                role=MessageRole.User,
                content=f"""Current chunking prompt:
{current_prompt}

Failure analysis:
{json.dumps(failure_analysis, indent=2)}

Please provide an improved prompt that addresses these specific failure patterns. Focus on the most common failure types and provide concrete guidance to prevent them."""
            )
        ]

        try:
            response = self.llm_gateway.generate(improvement_messages)
            # Handle both string and object responses
            if hasattr(response, 'content'):
                improved_prompt = response.content
            else:
                improved_prompt = str(response)

            # Basic validation that we got a meaningful response
            if len(improved_prompt.strip()) < 100:
                print(f"Generated prompt too short, using fallback improvements")
                return self._generate_fallback_improvements(failure_analysis, current_prompt)

            return improved_prompt
        except Exception as e:
            print(f"Failed to generate prompt improvements: {e}")
            return self._generate_fallback_improvements(failure_analysis, current_prompt)

    def _generate_fallback_improvements(self, failure_analysis: Dict[str, Any], current_prompt: str) -> str:
        """Generate rule-based prompt improvements when LLM generation fails."""

        # Start with the current prompt
        improved_prompt = current_prompt

        # Get the most common failure categories
        categories = failure_analysis.get('categories', {})

        # Add specific guidance based on failure patterns
        additional_guidance = []

        if 'title_only' in categories:
            additional_guidance.append("""
AVOID TITLE-ONLY CHUNKS: Never create chunks that contain only titles, headers, or single phrases without explanatory content. Always combine titles with their associated content to form complete, meaningful chunks.""")

        if 'too_brief' in categories:
            additional_guidance.append("""
MINIMUM CHUNK SIZE: Each chunk must contain at least 2-3 complete sentences and provide sufficient detail to be understood independently. Avoid single-sentence or phrase-only chunks.""")

        if 'lacks_context' in categories:
            additional_guidance.append("""
CONTEXT REQUIREMENT: Every chunk must include enough context to be understood without referring to other chunks. Include relevant background information, definitions, and explanations within each chunk.""")

        if 'incomplete_explanation' in categories:
            additional_guidance.append("""
COMPLETENESS CHECK: Ensure each chunk contains a complete explanation of its concept. Don't create chunks that introduce topics without explaining them or that end abruptly without conclusion.""")

        # Add the guidance to the prompt
        if additional_guidance:
            improved_prompt += "\n\nADDITIONAL CHUNKING RULES:\n" + "\n".join(additional_guidance)

            # Also add a size guideline
            improved_prompt += """

CHUNK SIZE GUIDELINES:
- Minimum: 50-100 words per chunk (except for very specific technical details)
- Optimal: 100-300 words per chunk
- Maximum: 500 words per chunk
- Always prioritize semantic completeness over size constraints"""

        return improved_prompt

    def test_prompt_iteration(self, test_cases: List[Dict[str, str]], prompt: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Test a specific prompt iteration and return completion rate and failures."""

        # Temporarily update the chunking engine's prompt
        original_method = self.chunking_engine.chunk_by_structured_output

        def custom_chunk_method(content: str, source: str = "unknown"):
            # Use the custom prompt
            messages = [
                LLMMessage(role=MessageRole.System, content=prompt),
                LLMMessage(
                    role=MessageRole.User,
                    content=f"Please analyze this content and break it into complete, semantically coherent knowledge chunks:\n\n{content}"
                )
            ]

            try:
                from context_mixer.domain.chunking_engine import StructuredChunkOutput
                structured_output = self.llm_gateway.generate_object(messages, StructuredChunkOutput)

                if not structured_output.chunks:
                    return self.chunking_engine._chunk_by_units(content, source)

                # Convert to KnowledgeChunk objects (simplified version)
                chunks = []
                for i, chunk_data in enumerate(structured_output.chunks):
                    from context_mixer.domain.knowledge import (
                        KnowledgeChunk, ChunkMetadata, AuthorityLevel, 
                        GranularityLevel, TemporalScope, ProvenanceInfo
                    )
                    from datetime import datetime

                    provenance = ProvenanceInfo(
                        source=source,
                        created_at=datetime.now().isoformat(),
                        author="LLM-StructuredChunking"
                    )

                    try:
                        authority = AuthorityLevel(chunk_data.authority.lower())
                    except ValueError:
                        authority = AuthorityLevel.CONVENTIONAL

                    try:
                        granularity = GranularityLevel(chunk_data.granularity.lower())
                    except ValueError:
                        granularity = GranularityLevel.DETAILED

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

                    chunk_id = self.chunking_engine._generate_chunk_id(chunk_data.content, chunk_data.concept)

                    chunk = KnowledgeChunk(
                        id=chunk_id,
                        content=chunk_data.content,
                        metadata=metadata
                    )
                    chunks.append(chunk)

                return chunks

            except Exception as e:
                print(f"Custom chunking failed: {e}")
                return self.chunking_engine._chunk_by_units(content, source)

        # Temporarily replace the method
        self.chunking_engine.chunk_by_structured_output = custom_chunk_method

        try:
            total_valid = 0
            total_chunks = 0
            all_failures = []

            for test_case in test_cases:
                chunks = self.chunking_engine.chunk_by_structured_output(test_case['content'], source="test")

                for i, chunk in enumerate(chunks):
                    validation = self.chunking_engine.validate_chunk_completeness(chunk)
                    if validation.is_complete:
                        total_valid += 1
                    else:
                        all_failures.append({
                            'test_case': test_case['name'],
                            'chunk_index': i,
                            'reason': validation.reason,
                            'issues': validation.issues,
                            'confidence': validation.confidence,
                            'content_preview': chunk.content[:200]
                        })
                    total_chunks += 1

            completion_rate = (total_valid / total_chunks) * 100 if total_chunks > 0 else 0
            return completion_rate, all_failures

        finally:
            # Restore original method
            self.chunking_engine.chunk_by_structured_output = original_method

    def iterative_improvement(self, test_cases: List[Dict[str, str]], target_rate: float = 90.0, max_iterations: int = 5) -> Dict[str, Any]:
        """Iteratively improve chunking prompts until target completion rate is achieved."""

        # Get the current prompt from the chunking engine
        current_prompt = """You are an expert knowledge curator following CRAFT principles for chunking content into domain-coherent units.

Your task is to analyze the given content and break it into complete, semantically coherent knowledge chunks. Each chunk should:

1. Contain a complete concept or idea that can stand alone
2. Be semantically bounded to prevent knowledge interference
3. Include all necessary context to be understood independently
4. Follow domain separation principles (technical, business, design, etc.)

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

CRITICAL: Do not try to preserve exact character positions or markdown formatting. Focus on semantic completeness and conceptual coherence. Each chunk should be a complete, self-contained unit of knowledge.

Output complete chunks directly - do not emit metadata about character positions or line numbers."""

        print("🔄 Starting iterative improvement process...")
        print(f"Target completion rate: {target_rate}%")
        print(f"Maximum iterations: {max_iterations}")
        print("=" * 70)

        for iteration in range(max_iterations):
            print(f"\n🧪 Iteration {iteration + 1}")
            print("-" * 50)

            # Test current prompt
            completion_rate, failures = self.test_prompt_iteration(test_cases, current_prompt)

            print(f"📊 Completion rate: {completion_rate:.1f}%")
            print(f"❌ Failures: {len(failures)}")

            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'prompt': current_prompt,
                'completion_rate': completion_rate,
                'failures': failures
            }
            self.iteration_history.append(iteration_result)

            # Check if target is achieved
            if completion_rate >= target_rate:
                print(f"✅ Target achieved! Completion rate: {completion_rate:.1f}%")
                break

            # Analyze failures and improve prompt
            if failures:
                print("🔍 Analyzing failure patterns...")
                failure_analysis = self.analyze_failure_patterns(failures)

                print("📈 Failure breakdown:")
                for category, data in failure_analysis['categories'].items():
                    print(f"  {category}: {data['count']} ({data['percentage']:.1f}%)")

                print("🛠️  Generating improved prompt...")
                improved_prompt = self.generate_prompt_improvements(failure_analysis, current_prompt)

                if improved_prompt != current_prompt:
                    current_prompt = improved_prompt
                    print("✅ Prompt updated")
                else:
                    print("⚠️  No prompt improvements generated")
                    break
            else:
                print("🎉 No failures detected!")
                break

        # Final results
        final_result = {
            'final_completion_rate': completion_rate,
            'iterations_completed': len(self.iteration_history),
            'target_achieved': completion_rate >= target_rate,
            'final_prompt': current_prompt,
            'iteration_history': self.iteration_history
        }

        print(f"\n{'='*70}")
        print(f"🏁 FINAL RESULTS:")
        print(f"Final completion rate: {completion_rate:.1f}%")
        print(f"Target achieved: {'✅ Yes' if completion_rate >= target_rate else '❌ No'}")
        print(f"Iterations completed: {len(self.iteration_history)}")

        return final_result


def main():
    """Run the feedback system."""

    # Initialize LLM gateway
    print("Initializing feedback system...")
    llm_gateway = LLMGateway(model="qwen3:32b")
    feedback_system = ChunkingFeedbackSystem(llm_gateway)

    # Test cases
    test_cases = [
        {
            "name": "Well-formed Markdown",
            "content": """# Project Setup Guide

## Prerequisites
Before starting, ensure you have Python 3.8+ installed.

## Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`

## Configuration
Create a `.env` file with your API keys:
```
API_KEY=your_key_here
DEBUG=true
```

## Troubleshooting
If you encounter issues, check the logs in `/var/log/app.log`."""
        },
        {
            "name": "Malformed Markdown",
            "content": """Project Setup
This is some text without proper headers
Prerequisites: Python 3.8+
Installation:
Clone repo
Install deps
Run tests
Config file should have API_KEY=your_key DEBUG=true
Check logs if problems"""
        },
        {
            "name": "Plain Text",
            "content": """This is a plain text document about software architecture principles. 
The first principle is separation of concerns which means each module should have a single 
responsibility.
The second principle is dependency inversion where high level modules should not depend on low 
level modules.
Both should depend on abstractions. The third principle is open closed principle meaning software 
entities should be open for extension but closed for modification."""
        }
    ]

    # Run iterative improvement
    results = feedback_system.iterative_improvement(test_cases, target_rate=90.0, max_iterations=5)

    # Save results
    results_file = Path(__file__).parent / "feedback_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
