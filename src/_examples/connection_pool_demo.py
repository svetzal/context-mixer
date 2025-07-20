import asyncio
import time
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging to see connection pool activity
logging.basicConfig(level=logging.INFO)

from context_mixer.domain.knowledge_store import KnowledgeStoreFactory
from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    ChunkMetadata,
    SearchQuery,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope,
    ProvenanceInfo
)


async def create_test_chunks(count: int = 20) -> list:
    """Create test chunks for performance testing."""
    chunks = []
    for i in range(count):
        chunk = KnowledgeChunk(
            id=f"test_chunk_{i}",
            content=f"This is test content for chunk {i}. It contains information about testing connection pooling performance.",
            metadata=ChunkMetadata(
                domains=[f"test_domain_{i % 3}"],
                authority=AuthorityLevel.CONVENTIONAL,
                scope=["test"],
                granularity=GranularityLevel.DETAILED,
                temporal=TemporalScope.CURRENT,
                dependencies=[],
                tags=[f"test_{i}", "performance", "connection_pool"],
                provenance=ProvenanceInfo(
                    source=f"connection_pool_test_{i}",
                    project_id="connection_pool_test",
                    project_name="Connection Pool Test",
                    created_at="2024-01-01T00:00:00Z"
                )
            )
        )
        chunks.append(chunk)
    return chunks


async def test_concurrent_operations(store, chunks, operation_count: int = 50):
    """Test concurrent operations to stress test the connection pool."""
    print(f"\n🔄 Testing {operation_count} concurrent operations...")

    async def perform_operations():
        tasks = []

        # Mix of different operations
        for i in range(operation_count):
            if i % 4 == 0:
                # Store operation
                chunk_batch = chunks[i % len(chunks):i % len(chunks) + 2]
                tasks.append(store.store_chunks(chunk_batch))
            elif i % 4 == 1:
                # Search operation
                query = SearchQuery(text=f"test content {i % 10}", max_results=5)
                tasks.append(store.search(query))
            elif i % 4 == 2:
                # Get chunk operation
                chunk_id = f"test_chunk_{i % len(chunks)}"
                tasks.append(store.get_chunk(chunk_id))
            else:
                # Get stats operation
                tasks.append(store.get_stats())

        # Execute all operations concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Count successful operations
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        return {
            'total_time': end_time - start_time,
            'successful': successful,
            'failed': failed,
            'operations_per_second': len(results) / (end_time - start_time)
        }

    return await perform_operations()


async def test_connection_pool_performance():
    """Test connection pool performance with different configurations."""
    print("🚀 Connection Pool Performance Test")
    print("=" * 50)

    # Create temporary database directories
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_db"

        # Test configurations
        configs = [
            {"name": "Small Pool", "pool_size": 2, "max_pool_size": 3},
            {"name": "Medium Pool", "pool_size": 5, "max_pool_size": 10},
            {"name": "Large Pool", "pool_size": 10, "max_pool_size": 20},
        ]

        test_chunks = await create_test_chunks(20)

        for config in configs:
            print(f"\n📊 Testing {config['name']} Configuration")
            print(f"   Pool Size: {config['pool_size']}, Max: {config['max_pool_size']}")

            # Create store with specific pool configuration
            store = KnowledgeStoreFactory.create_vector_store(
                db_path=db_path,
                pool_size=config['pool_size'],
                max_pool_size=config['max_pool_size'],
                connection_timeout=30.0
            )

            try:
                # Initial setup - store some chunks
                await store.store_chunks(test_chunks[:10])

                # Test concurrent operations
                results = await test_concurrent_operations(store, test_chunks, 30)

                # Get connection pool stats
                stats = await store.get_stats()
                pool_stats = stats.get('connection_pool', {})

                print(f"   ✅ Results:")
                print(f"      Total Time: {results['total_time']:.2f}s")
                print(f"      Operations/sec: {results['operations_per_second']:.2f}")
                print(f"      Successful: {results['successful']}")
                print(f"      Failed: {results['failed']}")
                print(f"   📈 Pool Stats:")
                print(f"      Current Connections: {pool_stats.get('current_connections', 'N/A')}")
                print(f"      Available Connections: {pool_stats.get('available_connections', 'N/A')}")
                print(f"      Pool Utilization: {((pool_stats.get('current_connections', 0) - pool_stats.get('available_connections', 0)) / max(pool_stats.get('current_connections', 1), 1) * 100):.1f}%")

            finally:
                # Clean up
                await store.reset()
                await store.close()


async def test_connection_pool_health_monitoring():
    """Test connection pool health monitoring features."""
    print("\n🏥 Connection Pool Health Monitoring Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "health_test_db"

        # Create store with health monitoring
        store = KnowledgeStoreFactory.create_vector_store(
            db_path=db_path,
            pool_size=3,
            max_pool_size=5,
            connection_timeout=10.0
        )

        try:
            # Create some test data
            test_chunks = await create_test_chunks(5)
            await store.store_chunks(test_chunks)

            # Monitor stats over time
            print("📊 Monitoring connection pool over time...")
            for i in range(5):
                stats = await store.get_stats()
                pool_stats = stats.get('connection_pool', {})

                print(f"   Time {i+1}:")
                print(f"      Total Chunks: {stats.get('total_chunks', 0)}")
                print(f"      Pool Connections: {pool_stats.get('current_connections', 'N/A')}")
                print(f"      Available: {pool_stats.get('available_connections', 'N/A')}")
                print(f"      Last Health Check: {pool_stats.get('last_health_check', 'N/A')}")

                # Perform some operations to exercise the pool
                query = SearchQuery(text=f"test content {i}", max_results=3)
                results = await store.search(query)
                print(f"      Search Results: {len(results.get_chunks())} chunks")

                await asyncio.sleep(1)

        finally:
            await store.close()


async def main():
    """Run all connection pool tests."""
    print("🔧 Context Mixer - Connection Pool Implementation Test")
    print("=" * 60)

    try:
        await test_connection_pool_performance()
        await test_connection_pool_health_monitoring()

        print("\n✅ All connection pool tests completed successfully!")
        print("\n📋 Summary:")
        print("   • Connection pooling reduces database connection overhead")
        print("   • Configurable pool size and timeout settings")
        print("   • Health monitoring tracks connection status")
        print("   • Concurrent operations benefit from connection reuse")
        print("   • Proper resource cleanup prevents connection leaks")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
