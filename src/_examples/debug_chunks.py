import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.WARN)

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mojentic.llm.gateways import OpenAIGateway
from context_mixer.gateways.llm import LLMGateway
from context_mixer.domain.chunking_engine import ChunkingEngine

def debug_chunking():
    """Debug the chunking process to see what content is being created."""

    # Read the actual file content
    content = Path("../../.github/copilot-instructions.md").read_text()

    # Create LLM gateway and chunking engine
    openai_gateway = OpenAIGateway(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_gateway = LLMGateway(model="o4-mini", gateway=openai_gateway)
    chunking_engine = ChunkingEngine(llm_gateway)

    print("Creating chunks from copilot-instructions.md...")
    chunks = chunking_engine.chunk_by_concepts(content, source=".github/copilot-instructions.md")

    print(f"\nCreated {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (ID: {chunk.id[:12]}...) ---")
        print(f"Concept: {chunk.metadata.tags[0] if chunk.metadata.tags else 'Unknown'}")
        print(f"Length: {len(chunk.content)} characters")
        print(f"Content preview: {repr(chunk.content[:100])}...")

        # Test validation
        validation_result = chunking_engine.validate_chunk_completeness(chunk)
        print(f"Validation result: {validation_result.is_complete}")
        print(f"Reason: {validation_result.reason}")
        print(f"Confidence: {validation_result.confidence:.2f}")
        if validation_result.issues:
            print(f"Issues: {', '.join(validation_result.issues)}")

        # Show the full content for the first few chunks to understand the issue
        # if i <= 3:
        print(f"Full content: {repr(chunk.content)}")

if __name__ == "__main__":
    debug_chunking()
