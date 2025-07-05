#!/usr/bin/env python3
"""
Experiment script to compare character-based chunking vs structured output chunking.

This script experiments with both approaches on various types of content to verify that
the structured output approach achieves better completion rates.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.gateways.llm import LLMGateway


def check_chunking_approaches():
    """Check both chunking approaches on various content types."""

    # Initialize LLM gateway (using a local model for testing)
    llm_gateway = LLMGateway(model="qwen3:32b")
    chunking_engine = ChunkingEngine(llm_gateway)

    # Read the copilot instructions file content
    copilot_instructions_path = Path(__file__).parent.parent.parent / ".github" / "copilot-instructions.md"
    try:
        with open(copilot_instructions_path, 'r', encoding='utf-8') as f:
            copilot_instructions_content = f.read()
    except FileNotFoundError:
        copilot_instructions_content = None
        print(f"Warning: Could not find {copilot_instructions_path}")

    # Test cases with different content types
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
        },
        {
            "name": "Mixed Content",
            "content": """API Documentation
The user authentication endpoint accepts POST requests
URL: /api/auth/login
Parameters: username, password
Returns: JWT token
Example: curl -X POST /api/auth/login -d '{"username":"test","password":"secret"}'
Error codes:
401 - Invalid credentials
429 - Rate limit exceeded
500 - Server error
Security considerations: Always use HTTPS in production"""
        }
    ]

    # Add the copilot instructions as a test case if the file was found
    if copilot_instructions_content:
        test_cases.append({
            "name": "Copilot Instructions (Real Documentation)",
            "content": copilot_instructions_content
        })

    print("Testing Chunking Approaches")
    print("=" * 50)

    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 30)

        # Test character-based approach
        print("Character-based chunking:")
        try:
            old_chunks = chunking_engine.chunk_by_concepts(test_case['content'], source="test")
            old_valid = 0
            for chunk in old_chunks:
                validation = chunking_engine.validate_chunk_completeness(chunk)
                if validation.is_complete:
                    old_valid += 1
                else:
                    print(f"  ❌ Incomplete chunk: {validation.reason}")

            old_completion_rate = (old_valid / len(old_chunks)) * 100 if old_chunks else 0
            print(f"  📊 {old_valid}/{len(old_chunks)} chunks complete ("
                  f"{old_completion_rate:.1f}%)")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            old_completion_rate = 0
            old_chunks = []

        # Test structured output approach
        print("Structured output chunking:")
        try:
            new_chunks = chunking_engine.chunk_by_structured_output(test_case['content'],
                                                                    source="test")
            new_valid = 0
            for chunk in new_chunks:
                validation = chunking_engine.validate_chunk_completeness(chunk)
                if validation.is_complete:
                    new_valid += 1
                else:
                    print(f"  ❌ Incomplete chunk: {validation.reason}")

            new_completion_rate = (new_valid / len(new_chunks)) * 100 if new_chunks else 0
            print(f"  📊 {new_valid}/{len(new_chunks)} chunks comp"
                  f"lete ({new_completion_rate:.1f}%)")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            new_completion_rate = 0
            new_chunks = []

        # Compare results
        if new_completion_rate > old_completion_rate:
            print(
                f"  ✅ Structured output improv"
                f"ed by {new_completion_rate - old_completion_rate:.1f}%")
        elif new_completion_rate == old_completion_rate:
            print(f"  ➡️  Both approaches achieved {new_completion_rate:.1f}%")
        else:
            print(
                f"  ⚠️  Structured output decreased by "
                f"{old_completion_rate - new_completion_rate:.1f}%")

    print("\n" + "=" * 50)
    print("Test completed. The structured output approach should show")
    print("better completion rates, especially for malformed content.")


if __name__ == "__main__":
    check_chunking_approaches()
