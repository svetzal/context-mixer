#!/usr/bin/env python3
"""
Test an improved chunking prompt based on the failure analysis.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.gateways.llm import LLMGateway
from mojentic.llm import LLMMessage, MessageRole


def check_improved_prompt():
    """Test an improved prompt that addresses the identified failure patterns."""

    # Initialize LLM gateway
    print("Initializing LLM gateway...")
    llm_gateway = LLMGateway(model="qwen3:32b")
    chunking_engine = ChunkingEngine(llm_gateway)

    # Improved prompt based on failure analysis
    improved_prompt = """You are an expert knowledge curator following CRAFT principles for chunking content into domain-coherent units.

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

Output complete chunks directly - do not emit metadata about character positions or line numbers.

ADDITIONAL CHUNKING RULES:

AVOID TITLE-ONLY CHUNKS: Never create chunks that contain only titles, headers, or single phrases without explanatory content. Always combine titles with their associated content to form complete, meaningful chunks.

MINIMUM CHUNK SIZE: Each chunk must contain at least 2-3 complete sentences and provide sufficient detail to be understood independently. Avoid single-sentence or phrase-only chunks.

CONTEXT REQUIREMENT: Every chunk must include enough context to be understood without referring to other chunks. Include relevant background information, definitions, and explanations within each chunk.

COMPLETENESS CHECK: Ensure each chunk contains a complete explanation of its concept. Don't create chunks that introduce topics without explaining them or that end abruptly without conclusion.

CHUNK SIZE GUIDELINES:
- Minimum: 50-100 words per chunk (except for very specific technical details)
- Optimal: 100-300 words per chunk
- Maximum: 500 words per chunk
- Always prioritize semantic completeness over size constraints"""

    # Test cases (all original test cases)
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

    print("Testing improved chunking prompt...")
    print("=" * 70)

    total_valid = 0
    total_chunks = 0
    all_failures = []

    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 50)

        try:
            # Use the improved prompt
            messages = [
                LLMMessage(role=MessageRole.System, content=improved_prompt),
                LLMMessage(
                    role=MessageRole.User,
                    content=f"Please analyze this content and break it into complete, semantically coherent knowledge chunks:\n\n{test_case['content']}"
                )
            ]

            from context_mixer.domain.chunking_engine import StructuredChunkOutput
            structured_output = llm_gateway.generate_object(messages, StructuredChunkOutput)

            if not structured_output.chunks:
                print("❌ No chunks generated")
                continue

            print(f"Generated {len(structured_output.chunks)} chunks")

            # Convert to KnowledgeChunk objects and validate
            valid_chunks = 0

            for i, chunk_data in enumerate(structured_output.chunks):
                from context_mixer.domain.knowledge import (
                    KnowledgeChunk, ChunkMetadata, AuthorityLevel, 
                    GranularityLevel, TemporalScope, ProvenanceInfo
                )
                from datetime import datetime

                # Create a simplified chunk for validation
                chunk = KnowledgeChunk(
                    id=f"test_chunk_{i}",
                    content=chunk_data.content,
                    metadata=ChunkMetadata(
                        domains=chunk_data.domains,
                        authority=AuthorityLevel.CONVENTIONAL,
                        scope=chunk_data.scope,
                        granularity=GranularityLevel.DETAILED,
                        temporal=TemporalScope.CURRENT,
                        dependencies=chunk_data.dependencies,
                        conflicts=chunk_data.conflicts,
                        tags=chunk_data.tags,
                        provenance=ProvenanceInfo(
                            source="test",
                            created_at=datetime.now().isoformat(),
                            author="test"
                        )
                    )
                )

                print(f"\nChunk {i+1}: {chunk_data.concept}")
                print(f"Content length: {len(chunk.content)} chars")
                print(f"Content preview: {chunk.content[:100]}...")

                validation = chunking_engine.validate_chunk_completeness(chunk)
                if validation.is_complete:
                    valid_chunks += 1
                    total_valid += 1
                    print(f"✅ Valid (confidence: {validation.confidence:.2f})")
                else:
                    print(f"❌ Invalid: {validation.reason}")
                    print(f"   Issues: {validation.issues}")
                    all_failures.append({
                        'test_case': test_case['name'],
                        'chunk_index': i,
                        'reason': validation.reason,
                        'issues': validation.issues,
                        'confidence': validation.confidence,
                        'content_preview': chunk.content[:200]
                    })

                total_chunks += 1

            completion_rate = (valid_chunks / len(structured_output.chunks)) * 100
            print(f"\n📊 {test_case['name']} completion rate: {valid_chunks}/{len(structured_output.chunks)} ({completion_rate:.1f}%)")

        except Exception as e:
            print(f"❌ {test_case['name']} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    # Calculate overall results
    overall_completion_rate = (total_valid / total_chunks) * 100 if total_chunks > 0 else 0

    print(f"\n{'='*70}")
    print(f"📊 OVERALL RESULTS:")
    print(f"Total completion rate: {total_valid}/{total_chunks} ({overall_completion_rate:.1f}%)")

    if all_failures:
        print(f"\n🔍 REMAINING FAILURES ({len(all_failures)} failures):")
        for failure in all_failures:
            test_case = failure.get('test_case', 'Unknown')
            chunk_idx = failure.get('chunk_index', 'N/A')
            print(f"  [{test_case}] Chunk {chunk_idx}: {failure['reason']}")
            print(f"    Issues: {failure['issues']}")
            print(f"    Preview: {failure.get('content_preview', '')[:100]}...")
            print()

    if overall_completion_rate >= 90:
        print("✅ SUCCESS: Achieved 90%+ completion rate!")
    else:
        print(f"❌ NEEDS IMPROVEMENT: {overall_completion_rate:.1f}% < 90% target")

    return overall_completion_rate, all_failures


if __name__ == "__main__":
    completion_rate, failures = check_improved_prompt()
    print(f"\nFinal result: {completion_rate:.1f}% completion rate")
