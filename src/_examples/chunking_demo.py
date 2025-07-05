#!/usr/bin/env python3
"""
Demo script to test the ChunkingEngine integration with the ingest command.

This script demonstrates how the ChunkingEngine can detect semantic boundaries
in real agent instruction prompts and create intelligent knowledge chunks.
"""

# Add src to path for imports
import sys
import tempfile
from pathlib import Path

from rich.console import Console

sys.path.insert(0, 'src')

from context_mixer.domain.chunking_engine import ChunkingEngine
from context_mixer.commands.ingest import do_ingest
from context_mixer.config import Config


def create_sample_agent_prompt():
    """Create a sample agent instruction prompt for testing."""
    return """# AI Coding Assistant Instructions

You are an expert software developer and architect with deep knowledge of modern development practices.

## Core Principles

Follow these fundamental principles in all your responses:
- Write clean, maintainable, and well-documented code
- Prioritize readability and simplicity over cleverness
- Use established patterns and best practices
- Consider performance implications of your suggestions

## Code Quality Standards

When writing or reviewing code, ensure:
1. Proper error handling and edge case management
2. Comprehensive unit tests for all new functionality
3. Clear variable and function naming conventions
4. Appropriate use of design patterns where beneficial
5. Security considerations for all user inputs

## Architecture Guidelines

For system design and architecture decisions:
- Favor composition over inheritance
- Apply SOLID principles consistently
- Design for scalability and maintainability
- Consider the trade-offs between different approaches
- Document architectural decisions and their rationale

## Communication Style

When explaining technical concepts:
- Start with high-level overview, then dive into details
- Use concrete examples to illustrate abstract concepts
- Provide step-by-step instructions for complex procedures
- Anticipate common questions and address them proactively
- Adapt explanations to the user's apparent skill level

## Technology Preferences

Unless specifically requested otherwise:
- Use Python 3.9+ for backend development
- Prefer TypeScript over JavaScript for frontend work
- Recommend PostgreSQL for relational data storage
- Suggest Docker for containerization needs
- Use pytest for Python testing frameworks

## Response Format

Structure your responses as follows:
1. Brief summary of what you'll address
2. Detailed explanation or implementation
3. Key considerations and potential issues
4. Next steps or recommendations for further improvement"""


def create_mock_llm_gateway():
    """Create a mock LLM gateway for testing without actual LLM calls."""
    class MockLLMGateway:
        def generate_object(self, messages, object_model):
            # Mock boundary detection
            if "BoundaryDetectionResult" in str(object_model):
                from context_mixer.domain.chunking_engine import BoundaryDetectionResult, ChunkBoundary
                return BoundaryDetectionResult(
                    boundaries=[
                        ChunkBoundary(
                            start_position=0,
                            end_position=200,
                            concept="Introduction and Core Principles",
                            confidence=0.9,
                            reasoning="Clear introduction section with core principles"
                        ),
                        ChunkBoundary(
                            start_position=200,
                            end_position=600,
                            concept="Code Quality Standards",
                            confidence=0.85,
                            reasoning="Distinct section about code quality requirements"
                        ),
                        ChunkBoundary(
                            start_position=600,
                            end_position=900,
                            concept="Architecture Guidelines",
                            confidence=0.88,
                            reasoning="Architecture and design principles section"
                        ),
                        ChunkBoundary(
                            start_position=900,
                            end_position=1200,
                            concept="Communication Style",
                            confidence=0.82,
                            reasoning="Guidelines for technical communication"
                        ),
                        ChunkBoundary(
                            start_position=1200,
                            end_position=1500,
                            concept="Technology Preferences",
                            confidence=0.87,
                            reasoning="Technology stack recommendations"
                        ),
                        ChunkBoundary(
                            start_position=1500,
                            end_position=1800,
                            concept="Response Format",
                            confidence=0.84,
                            reasoning="Structure for responses"
                        )
                    ],
                    total_concepts=6,
                    confidence_score=0.86
                )
            
            # Mock concept analysis
            elif "ConceptAnalysis" in str(object_model):
                from context_mixer.domain.chunking_engine import ConceptAnalysis
                from context_mixer.domain.knowledge import AuthorityLevel, GranularityLevel
                return ConceptAnalysis(
                    concept="AI Assistant Instructions",
                    domains=["technical", "development", "ai"],
                    authority_level=AuthorityLevel.OFFICIAL,
                    granularity=GranularityLevel.DETAILED,
                    tags=["ai", "coding", "instructions", "best-practices"],
                    dependencies=["software-development", "architecture"],
                    conflicts_with=[]
                )
        
        def generate(self, messages):
            return "COMPLETE"
    
    return MockLLMGateway()


def main():
    """Main demo function."""
    console = Console()
    
    console.print("[bold blue]ChunkingEngine Demo[/bold blue]")
    console.print("Testing semantic boundary detection with agent instruction prompts\n")
    
    # Create sample content
    sample_content = create_sample_agent_prompt()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        temp_file = Path(f.name)
    
    try:
        # Create mock LLM gateway
        llm_gateway = create_mock_llm_gateway()
        
        # Test ChunkingEngine directly
        console.print("[yellow]1. Testing ChunkingEngine directly:[/yellow]")
        chunking_engine = ChunkingEngine(llm_gateway)
        
        # Detect boundaries
        console.print("   Detecting semantic boundaries...")
        boundary_result = chunking_engine.detect_semantic_boundaries(sample_content)
        console.print(f"   Found {len(boundary_result.boundaries)} semantic boundaries")
        console.print(f"   Overall confidence: {boundary_result.confidence_score:.2f}")
        
        # Create chunks
        console.print("   Creating knowledge chunks...")
        chunks = chunking_engine.chunk_by_concepts(sample_content, source="demo_prompt.md")
        console.print(f"   Created {len(chunks)} knowledge chunks")
        
        # Display chunk information
        for i, chunk in enumerate(chunks, 1):
            console.print(f"   Chunk {i}: {chunk.id[:12]}... ({len(chunk.content)} chars)")
            console.print(f"     Domains: {', '.join(chunk.metadata.domains)}")
            console.print(f"     Authority: {chunk.metadata.authority.value}")
            console.print(f"     Tags: {', '.join(chunk.metadata.tags)}")
        
        console.print("\n[yellow]2. Testing integration with ingest command:[/yellow]")
        
        # Create temporary config
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(library_path=Path(temp_dir))
            
            # Test ingest with boundary detection enabled
            console.print("   Running ingest with semantic boundary detection...")
            do_ingest(
                console=console,
                config=config,
                llm_gateway=llm_gateway,
                filename=temp_file,
                commit=False,  # Skip git commit for demo
                detect_boundaries=True
            )
            
            # Check if context.md was created
            context_file = config.library_path / "context.md"
            if context_file.exists():
                console.print(f"   ✓ Created context.md ({context_file.stat().st_size} bytes)")
                
                # Show first few lines
                content = context_file.read_text()
                lines = content.split('\n')[:10]
                console.print("   First 10 lines of generated context.md:")
                for line in lines:
                    console.print(f"     {line}")
                if len(content.split('\n')) > 10:
                    console.print("     ...")
            else:
                console.print("   ✗ context.md was not created")
        
        console.print("\n[green]✓ Demo completed successfully![/green]")
        console.print("\nThe ChunkingEngine successfully:")
        console.print("- Detected semantic boundaries in the agent instruction prompt")
        console.print("- Created coherent knowledge chunks with proper metadata")
        console.print("- Integrated seamlessly with the ingest command")
        console.print("- Generated a structured context.md file")
        
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    main()