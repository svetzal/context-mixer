#!/usr/bin/env python3
"""
Simple test to understand chunking failures without timeout issues.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.gateways.llm import LLMGateway


def run_simple_test():
    """Run a simple test with multiple content types to identify failure patterns."""

    # Initialize LLM gateway (using a local model for testing)
    print("Initializing LLM gateway...")
    llm_gateway = LLMGateway(model="qwen3:32b")
    chunking_engine = ChunkingEngine(llm_gateway)

    # Test cases with different content types (from original experiment)
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

    print("Testing structured output chunking on multiple content types...")
    print("=" * 70)

    all_results = []
    total_valid = 0
    total_chunks = 0

    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 50)

        try:
            # Test structured output approach
            chunks = chunking_engine.chunk_by_structured_output(test_case['content'], source="test")
            print(f"Generated {len(chunks)} chunks")

            # Validate each chunk and collect detailed failure info
            valid_chunks = 0
            failure_reasons = []

            for i, chunk in enumerate(chunks):
                print(f"\nChunk {i+1}:")
                print(f"Content length: {len(chunk.content)} chars")
                print(f"Content preview: {chunk.content[:100]}...")

                validation = chunking_engine.validate_chunk_completeness(chunk)
                if validation.is_complete:
                    valid_chunks += 1
                    print(f"✅ Valid (confidence: {validation.confidence:.2f})")
                else:
                    print(f"❌ Invalid: {validation.reason}")
                    print(f"   Issues: {validation.issues}")
                    print(f"   Confidence: {validation.confidence:.2f}")
                    failure_reasons.append({
                        'test_case': test_case['name'],
                        'chunk_index': i,
                        'reason': validation.reason,
                        'issues': validation.issues,
                        'confidence': validation.confidence,
                        'content_preview': chunk.content[:200]
                    })

            completion_rate = (valid_chunks / len(chunks)) * 100 if chunks else 0
            print(f"\n📊 {test_case['name']} completion rate: {valid_chunks}/{len(chunks)} ({completion_rate:.1f}%)")

            all_results.append({
                'test_case': test_case['name'],
                'completion_rate': completion_rate,
                'valid_chunks': valid_chunks,
                'total_chunks': len(chunks),
                'failures': failure_reasons
            })

            total_valid += valid_chunks
            total_chunks += len(chunks)

        except Exception as e:
            print(f"❌ {test_case['name']} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'test_case': test_case['name'],
                'completion_rate': 0,
                'valid_chunks': 0,
                'total_chunks': 0,
                'failures': [{'reason': str(e), 'issues': ['exception'], 'confidence': 0}]
            })

    # Calculate overall completion rate
    overall_completion_rate = (total_valid / total_chunks) * 100 if total_chunks > 0 else 0

    print(f"\n{'='*70}")
    print(f"📊 OVERALL RESULTS:")
    print(f"Total completion rate: {total_valid}/{total_chunks} ({overall_completion_rate:.1f}%)")

    # Show results by test case
    for result in all_results:
        print(f"  {result['test_case']}: {result['completion_rate']:.1f}%")

    # Collect all failures for analysis
    all_failures = []
    for result in all_results:
        all_failures.extend(result['failures'])

    if all_failures:
        print(f"\n🔍 FAILURE ANALYSIS ({len(all_failures)} failures):")
        for failure in all_failures:
            test_case = failure.get('test_case', 'Unknown')
            chunk_idx = failure.get('chunk_index', 'N/A')
            print(f"  [{test_case}] Chunk {chunk_idx}: {failure['reason']}")
            print(f"    Issues: {failure['issues']}")
            print(f"    Preview: {failure.get('content_preview', '')[:100]}...")
            print()

    return overall_completion_rate, all_failures


if __name__ == "__main__":
    completion_rate, failures = run_simple_test()
    print(f"\nFinal completion rate: {completion_rate:.1f}%")
    if completion_rate < 90:
        print("❌ Below 90% target - needs improvement")
    else:
        print("✅ Meets 90% target")
