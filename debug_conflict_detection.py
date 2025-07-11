import asyncio
import tempfile
from pathlib import Path

from context_mixer.config import Config
from context_mixer.gateways.llm import LLMGateway
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory
from context_mixer.domain.knowledge import KnowledgeChunk, ChunkMetadata, AuthorityLevel, GranularityLevel, TemporalScope, ProvenanceInfo
from datetime import datetime

# Import OpenAI gateway
from mojentic.llm.gateways import OpenAIGateway

async def debug_conflict_detection():
    """Debug why conflict detection isn't working between files."""

    # Setup LLM
    api_key = "sk-proj-fake-key"  # This will be overridden by environment
    openai_gateway = OpenAIGateway(api_key=api_key)
    llm_gateway = LLMGateway(model="gpt-4o-mini", gateway=openai_gateway)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        vector_store_path = temp_path / "vector_store"

        # Create knowledge store
        knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path)

        # Create first chunk (4 spaces indentation)
        chunk1_content = "Use exactly 4 spaces per indentation level. Do not use tabs under any circumstances."
        chunk1_metadata = ChunkMetadata(
            domains=["code_style"],
            authority=AuthorityLevel.CONVENTIONAL,
            scope=["project"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["indentation"],
            provenance=ProvenanceInfo(
                source="code1.md",
                project_id="test",
                project_name="Test",
                project_path="/test",
                created_at=datetime.now().isoformat()
            )
        )
        chunk1 = KnowledgeChunk(
            id="chunk1_indentation_4spaces",
            content=chunk1_content,
            metadata=chunk1_metadata
        )

        # Store first chunk
        await knowledge_store.store_chunks([chunk1])
        print(f"Stored first chunk: {chunk1.id}")
        print(f"Content: {chunk1.content}")

        # Create second chunk (2 spaces indentation)
        chunk2_content = "Use 2 spaces for indentation in all source files to maintain uniform code structure and readability."
        chunk2_metadata = ChunkMetadata(
            domains=["code_style"],
            authority=AuthorityLevel.CONVENTIONAL,
            scope=["project"],
            granularity=GranularityLevel.DETAILED,
            temporal=TemporalScope.CURRENT,
            tags=["indentation"],
            provenance=ProvenanceInfo(
                source="code2.md",
                project_id="test",
                project_name="Test",
                project_path="/test",
                created_at=datetime.now().isoformat()
            )
        )
        chunk2 = KnowledgeChunk(
            id="chunk2_indentation_2spaces",
            content=chunk2_content,
            metadata=chunk2_metadata
        )

        print(f"\nCreated second chunk: {chunk2.id}")
        print(f"Content: {chunk2.content}")

        # Test conflict detection
        print(f"\nTesting conflict detection...")
        conflicting_chunks = await knowledge_store.detect_conflicts(chunk2)

        if conflicting_chunks:
            print(f"✓ Found {len(conflicting_chunks)} conflicting chunks:")
            for chunk in conflicting_chunks:
                print(f"  - {chunk.id}: {chunk.content[:50]}...")
        else:
            print("✗ No conflicts detected")

        # Test direct LLM conflict detection
        print(f"\nTesting direct LLM conflict detection...")
        from context_mixer.commands.operations.merge import detect_conflicts
        conflicts = detect_conflicts(chunk1.content, chunk2.content, llm_gateway)

        if conflicts.list:
            print(f"✓ Direct LLM detected {len(conflicts.list)} conflicts:")
            for conflict in conflicts.list:
                print(f"  - {conflict.description}")
        else:
            print("✗ Direct LLM detected no conflicts")

if __name__ == "__main__":
    asyncio.run(debug_conflict_detection())
