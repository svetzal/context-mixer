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

async def debug_simple():
    """Simple debug to test conflict detection without LLM calls."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        vector_store_path = temp_path / "vector_store"
        
        # Create knowledge store
        knowledge_store = KnowledgeStoreFactory.create_vector_store(vector_store_path)
        
        # Create first chunk (4 spaces indentation)
        chunk1_content = "Use 4 spaces for indentation"
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
        print(f"Domains: {chunk1.metadata.domains}")
        
        # Create second chunk (2 spaces indentation)
        chunk2_content = "Use 2 spaces for indentation"
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
        print(f"Domains: {chunk2.metadata.domains}")
        
        # Test domain search first
        print(f"\nTesting domain search for 'code_style'...")
        from context_mixer.domain.knowledge import SearchQuery
        domain_query = SearchQuery(
            text="*",
            domains=["code_style"],
            max_results=100
        )
        domain_results = await knowledge_store.search(domain_query)
        print(f"Found {len(domain_results.results)} chunks in code_style domain:")
        for result in domain_results.results:
            print(f"  - {result.chunk.id}: {result.chunk.content}")
        
        # Test conflict detection (this will fail due to LLM call, but we can see if it gets that far)
        print(f"\nTesting conflict detection...")
        try:
            conflicting_chunks = await knowledge_store.detect_conflicts(chunk2)
            print(f"Found {len(conflicting_chunks)} conflicting chunks")
        except Exception as e:
            print(f"Conflict detection failed (expected due to LLM): {e}")

if __name__ == "__main__":
    asyncio.run(debug_simple())