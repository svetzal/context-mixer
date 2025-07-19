import time
import os
import logging
from typing import List

# Set logging level before importing mojentic
logging.basicConfig(level=logging.WARN)

from mojentic.llm.gateways import OpenAIGateway
from context_mixer.domain.chunking_engine import ChunkingEngine, ValidationResult
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, ProvenanceInfo, AuthorityLevel, GranularityLevel, TemporalScope
from context_mixer.gateways.llm import LLMGateway


def create_sample_chunks(count: int = 10) -> List[KnowledgeChunk]:
    """Create sample chunks for testing parallel validation."""
    chunks = []

    metadata = ChunkMetadata(
        domains=["technical"],
        authority=AuthorityLevel.CONVENTIONAL,
        scope=["general"],
        granularity=GranularityLevel.DETAILED,
        temporal=TemporalScope.CURRENT,
        tags=["test"],
        provenance=ProvenanceInfo(
            source="demo.py",
            project_id=None,
            project_name=None,
            project_path=None,
            created_at="2024-01-01T00:00:00",
            updated_at=None,
            author="demo"
        )
    )

    for i in range(count):
        chunk = KnowledgeChunk(
            id=f"chunk_{i+1}",
            content=f"This is sample chunk number {i+1} with meaningful content for validation testing. "
                   f"It contains enough text to pass basic validation checks and demonstrates the "
                   f"parallel validation functionality of the ChunkingEngine.",
            metadata=metadata
        )
        chunks.append(chunk)

    return chunks


def demo_parallel_validation():
    """Demonstrate parallel validation with performance comparison."""
    print("=== ChunkingEngine Parallel Validation Demo ===\n")

    # Create a real LLM gateway using OpenAI's o4-mini model
    # This will make actual API calls to validate chunk completeness
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return

    openai_gateway = OpenAIGateway(api_key=api_key)
    llm_gateway = LLMGateway(model="o4-mini", gateway=openai_gateway)

    # Create chunking engine
    engine = ChunkingEngine(llm_gateway)

    # Create sample chunks
    chunk_counts = [5, 10, 20]

    for count in chunk_counts:
        print(f"Testing with {count} chunks:")
        chunks = create_sample_chunks(count)

        # Sequential validation
        start_time = time.time()
        sequential_results = [engine.validate_chunk_completeness(chunk) for chunk in chunks]
        sequential_time = time.time() - start_time

        # Parallel validation
        start_time = time.time()
        parallel_results = engine.validate_chunks_parallel(chunks, max_workers=4)
        parallel_time = time.time() - start_time

        # Verify results are identical
        results_match = True
        for seq, par in zip(sequential_results, parallel_results):
            if (seq.is_complete != par.is_complete or 
                seq.reason != par.reason or 
                seq.confidence != par.confidence or 
                seq.issues != par.issues):
                results_match = False
                break

        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        print(f"  Sequential time: {sequential_time:.3f}s")
        print(f"  Parallel time:   {parallel_time:.3f}s")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"  Results match:   {results_match}")
        print(f"  All chunks valid: {all(r.is_complete for r in parallel_results)}")
        print()

    # Test configurable concurrency
    print("Testing configurable concurrency levels:")
    chunks = create_sample_chunks(10)

    for workers in [1, 2, 4, 8]:
        start_time = time.time()
        results = engine.validate_chunks_parallel(chunks, max_workers=workers)
        elapsed_time = time.time() - start_time

        print(f"  {workers} workers: {elapsed_time:.3f}s ({len(results)} chunks validated)")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_parallel_validation()
