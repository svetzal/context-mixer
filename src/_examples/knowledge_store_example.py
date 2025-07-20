"""
Example demonstrating the KnowledgeStore interface with vector backend.

This example shows how to use the new KnowledgeStore interface to store,
search, and manage knowledge chunks following CRAFT principles.
"""

import asyncio
import tempfile
from pathlib import Path

from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    ProvenanceInfo,
    SearchQuery,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)
from context_mixer.domain.knowledge_store import KnowledgeStoreFactory


async def main():
    """Demonstrate KnowledgeStore usage."""
    print("🧠 KnowledgeStore Example - CRAFT Principles in Action")
    print("=" * 60)
    
    # Create a temporary database for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir)
        
        # Create a vector-based knowledge store
        print("📦 Creating vector knowledge store...")
        store = KnowledgeStoreFactory.create_vector_store(db_path)
        
        # Create some sample knowledge chunks following CRAFT principles
        print("\n📝 Creating knowledge chunks...")
        
        # Chunk 1: React Hooks (Current, Official)
        react_hooks_chunk = KnowledgeChunk(
            id="react-hooks-guide",
            content="Use React hooks for state management in functional components. "
                   "useState for local state, useEffect for side effects, and useContext for shared state.",
            metadata=ChunkMetadata(
                domains=["technical", "frontend"],
                authority=AuthorityLevel.OFFICIAL,
                scope=["enterprise", "production"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                tags=["react", "hooks", "javascript", "frontend"],
                provenance=ProvenanceInfo(
                    source="React Official Documentation",
                    created_at="2024-01-15T10:00:00Z",
                    author="frontend-team"
                )
            )
        )
        
        # Chunk 2: React Class Components (Deprecated)
        react_class_chunk = KnowledgeChunk(
            id="react-class-components",
            content="Use React class components with this.state and this.setState for state management. "
                   "Lifecycle methods like componentDidMount for side effects.",
            metadata=ChunkMetadata(
                domains=["technical", "frontend"],
                authority=AuthorityLevel.DEPRECATED,
                scope=["legacy"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.DEPRECATED,
                tags=["react", "class-components", "javascript", "legacy"],
                conflicts=["react-hooks-guide"],  # Explicit conflict
                provenance=ProvenanceInfo(
                    source="React Legacy Documentation",
                    created_at="2020-01-15T10:00:00Z",
                    author="frontend-team"
                )
            )
        )
        
        # Chunk 3: API Design Principles (High-level)
        api_design_chunk = KnowledgeChunk(
            id="api-design-principles",
            content="Design APIs to be RESTful, consistent, and intuitive. "
                   "Use proper HTTP methods, status codes, and resource naming conventions.",
            metadata=ChunkMetadata(
                domains=["technical", "backend", "architecture"],
                authority=AuthorityLevel.FOUNDATIONAL,
                scope=["enterprise", "production"],
                granularity=GranularityLevel.OVERVIEW,
                temporal=TemporalScope.CURRENT,
                tags=["api", "rest", "design", "architecture"],
                provenance=ProvenanceInfo(
                    source="Architecture Team Guidelines",
                    created_at="2024-01-10T14:00:00Z",
                    author="architecture-team"
                )
            )
        )
        
        # Store the chunks
        print("💾 Storing knowledge chunks...")
        await store.store_chunks([react_hooks_chunk, react_class_chunk, api_design_chunk])
        
        # Get store statistics
        stats = await store.get_stats()
        print(f"📊 Store Statistics: {stats['total_chunks']} chunks stored")
        
        # Demonstrate search functionality
        print("\n🔍 Searching for React knowledge...")
        search_query = SearchQuery(
            text="React state management",
            domains=["technical", "frontend"],
            max_results=5
        )
        results = await store.search(search_query)
        
        print(f"Found {len(results.results)} results:")
        for i, result in enumerate(results.results, 1):
            chunk = result.chunk
            print(f"  {i}. {chunk.id} (Authority: {chunk.metadata.authority.value}, "
                  f"Score: {result.relevance_score:.3f})")
            print(f"     {chunk.content[:100]}...")
        
        # Demonstrate filtering by authority
        print("\n🏛️ Getting current (non-deprecated) knowledge...")
        current_chunks = await store.get_chunks_by_authority([
            AuthorityLevel.FOUNDATIONAL,
            AuthorityLevel.OFFICIAL,
            AuthorityLevel.CONVENTIONAL
        ])
        
        print(f"Found {len(current_chunks)} current chunks:")
        for chunk in current_chunks:
            print(f"  - {chunk.id} ({chunk.metadata.authority.value})")
        
        # Demonstrate conflict detection
        print("\n⚠️  Detecting conflicts...")
        conflicts = await store.detect_conflicts(react_hooks_chunk)
        
        if conflicts:
            print(f"Found {len(conflicts)} conflicting chunks:")
            for conflict in conflicts:
                print(f"  - {conflict.id} ({conflict.metadata.authority.value}, "
                      f"{conflict.metadata.temporal.value})")
        else:
            print("No conflicts detected.")
        
        # Demonstrate dependency validation
        print("\n🔗 Validating dependencies...")
        missing_deps = await store.validate_dependencies(react_hooks_chunk)
        
        if missing_deps:
            print(f"Missing dependencies: {missing_deps}")
        else:
            print("All dependencies satisfied.")
        
        # Demonstrate domain filtering
        print("\n🏗️ Getting architecture knowledge...")
        arch_chunks = await store.get_chunks_by_domain(["architecture"])
        
        print(f"Found {len(arch_chunks)} architecture chunks:")
        for chunk in arch_chunks:
            print(f"  - {chunk.id}")
            print(f"    Domains: {', '.join(chunk.metadata.domains)}")
        
        print("\n✅ KnowledgeStore example completed successfully!")
        print("\nKey CRAFT Principles Demonstrated:")
        print("  🧩 Chunk: Knowledge separated into domain-coherent units")
        print("  🛡️  Resist: Conflict detection and authority hierarchies")
        print("  🎯 Adapt: Different granularity levels for different needs")
        print("  🔧 Fit: Domain and authority filtering for relevant knowledge")
        print("  🚀 Transcend: Storage-agnostic interface with vector backend")


if __name__ == "__main__":
    asyncio.run(main())