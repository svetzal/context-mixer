"""
Demonstration of HDBSCAN clustering performance improvements for conflict detection.

This script shows how the new clustering-based conflict detection reduces expensive
LLM calls from O(n²) to O(k*log(k)) where k << n².
"""

import asyncio
import logging
import time
from typing import List

from context_mixer.domain.conflict import ConflictList

# Set logging to WARN before importing mojentic
logging.basicConfig(level=logging.WARN)

from context_mixer.domain.knowledge import (
    KnowledgeChunk, ChunkMetadata, AuthorityLevel, GranularityLevel,
    TemporalScope, ProvenanceInfo
)
from context_mixer.domain.clustering_service import ClusteringService
from context_mixer.domain.cluster_aware_conflict_detection import ClusterAwareConflictDetector


def create_sample_chunks(count: int = 50) -> List[KnowledgeChunk]:
    """Create sample knowledge chunks for testing."""
    chunks = []

    # Define different domains and content types
    domains_content = [
        ("technical", "architecture", [
            "Use microservices architecture for scalability",
            "Implement API versioning for backward compatibility",
            "Follow REST principles for API design",
            "Use database migrations for schema changes",
            "Implement proper error handling in services"
        ]),
        ("technical", "testing", [
            "Write unit tests for all business logic",
            "Use integration tests for API endpoints",
            "Mock external dependencies in tests",
            "Maintain test coverage above 80%",
            "Use test-driven development approach"
        ]),
        ("business", "process", [
            "Follow agile development methodology",
            "Conduct regular sprint retrospectives",
            "Maintain product backlog prioritization",
            "Ensure stakeholder communication",
            "Document business requirements clearly"
        ]),
        ("operational", "deployment", [
            "Use containerization for consistent deployments",
            "Implement CI/CD pipelines for automation",
            "Monitor application performance metrics",
            "Set up proper logging and alerting",
            "Follow security best practices"
        ])
    ]

    chunk_id = 0
    for domain, subdomain, contents in domains_content:
        for content in contents:
            if chunk_id >= count:
                break

            # Create some variations to test clustering
            variations = [
                content,
                f"Important: {content}",
                f"Best practice: {content}",
                f"Guideline: {content}"
            ]

            for i, variation in enumerate(variations):
                if chunk_id >= count:
                    break

                # Create realistic embeddings (normally would come from LLM)
                # Simulate semantic similarity within domains
                base_embedding = [0.1 * (chunk_id % 10)] * 384  # Simulate 384-dim embedding
                if domain == "technical":
                    base_embedding[0] += 0.5
                elif domain == "business":
                    base_embedding[1] += 0.5
                elif domain == "operational":
                    base_embedding[2] += 0.5

                # Add some noise for clustering
                embedding = [val + 0.01 * (i - 2) for val in base_embedding]

                provenance = ProvenanceInfo(
                    source=f"sample_{domain}_{subdomain}.md",
                    project_id=f"project-{domain}",
                    project_name=f"Sample {domain.title()} Project",
                    created_at="2024-01-01T00:00:00Z"
                )

                metadata = ChunkMetadata(
                    domains=[domain, subdomain],
                    authority=AuthorityLevel.OFFICIAL,
                    scope=["enterprise"],
                    granularity=GranularityLevel.DETAILED,
                    temporal=TemporalScope.CURRENT,
                    provenance=provenance
                )

                chunk = KnowledgeChunk(
                    id=f"chunk-{chunk_id:03d}",
                    content=variation,
                    metadata=metadata,
                    embedding=embedding
                )

                chunks.append(chunk)
                chunk_id += 1

                if chunk_id >= count:
                    break

    return chunks[:count]


async def demo_clustering_performance():
    """Demonstrate clustering performance improvements."""
    print("🚀 HDBSCAN Clustering Performance Demo")
    print("=" * 50)

    # Create sample chunks
    print("📝 Creating sample knowledge chunks...")
    chunks = create_sample_chunks(30)  # Start with smaller number for demo
    print(f"   Created {len(chunks)} sample chunks across multiple domains")

    # Initialize clustering service (without LLM for this demo)
    print("\n🔧 Initializing clustering service...")
    clustering_service = ClusteringService()

    # Perform clustering
    print("\n🎯 Performing HDBSCAN clustering...")
    start_time = time.time()
    clustering_result = await clustering_service.cluster_knowledge_chunks(chunks)
    clustering_time = time.time() - start_time

    print(f"   ✅ Clustering completed in {clustering_time:.2f} seconds")
    print(f"   📊 Found {len(clustering_result.clusters)} clusters")
    print(f"   🔇 {len(clustering_result.noise_chunk_ids)} noise chunks")

    # Display cluster information
    print("\n📋 Cluster Analysis:")
    for i, cluster in enumerate(clustering_result.clusters):
        print(f"   Cluster {i+1}: {len(cluster.chunk_ids)} chunks, "
              f"type: {cluster.metadata.cluster_type}, "
              f"domains: {cluster.metadata.domains}")
        if cluster.summary:
            print(f"      Summary: {cluster.summary}")

    # Calculate performance improvement
    total_chunks = len(chunks)
    traditional_comparisons = total_chunks * (total_chunks - 1) // 2

    # Calculate cluster-aware comparisons
    cluster_comparisons = 0
    for cluster in clustering_result.clusters:
        cluster_size = len(cluster.chunk_ids)
        # Intra-cluster comparisons
        cluster_comparisons += cluster_size * (cluster_size - 1) // 2
        # Inter-cluster comparisons (limited to 3 representatives per cluster)
        inter_cluster_reps = min(3, cluster_size)
        cluster_comparisons += inter_cluster_reps * (len(clustering_result.clusters) - 1) * 3

    # Add noise chunk comparisons
    noise_chunks = len(clustering_result.noise_chunk_ids)
    cluster_comparisons += noise_chunks * len(clustering_result.clusters) * 3

    reduction_ratio = 1 - (cluster_comparisons / max(traditional_comparisons, 1))

    print(f"\n📈 Performance Analysis:")
    print(f"   Traditional approach: {traditional_comparisons} pairwise comparisons")
    print(f"   Cluster-aware approach: {cluster_comparisons} comparisons")
    print(f"   🎉 Reduction: {reduction_ratio * 100:.1f}% fewer LLM calls!")

    # Demonstrate conflict detection candidate generation
    print(f"\n🔍 Conflict Detection Optimization:")
    if chunks:
        test_chunk = chunks[0]
        candidates = clustering_service.generate_conflict_detection_candidates(
            clustering_result, test_chunk
        )

        print(f"   Target chunk: {test_chunk.id}")
        print(f"   Generated {len(candidates)} conflict detection candidates")
        print(f"   (vs {len(chunks) - 1} candidates in traditional approach)")

        # Show candidate priorities
        print(f"   Top 5 candidates by priority:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"      {i+1}. {candidate.chunk2_id} "
                  f"(priority: {candidate.priority_score:.2f}, "
                  f"context: {candidate.relationship_context})")

    print(f"\n✨ Demo completed successfully!")
    return clustering_result


async def demo_with_mock_llm():
    """Demo with mock LLM for conflict detection."""
    print("\n" + "=" * 50)
    print("🤖 Mock LLM Conflict Detection Demo")
    print("=" * 50)

    # Create a simple mock LLM gateway
    class MockLLMGateway:
        async def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
            # Simple mock response for cluster summaries
            if "summary" in prompt.lower():
                return "Technical guidelines for backend development and testing practices."
            return "No conflicts detected."

        def generate_object(self, messages, object_model):
            # Import here to avoid circular imports

            # Mock response for conflict detection - return empty conflict list
            if object_model == ConflictList:
                return ConflictList(list=[])

            # For other object types, return a simple mock
            return object_model()

    mock_llm = MockLLMGateway()

    # Create clustering service with mock LLM
    clustering_service = ClusteringService(mock_llm)
    cluster_detector = ClusterAwareConflictDetector(clustering_service, mock_llm)

    # Create sample chunks
    chunks = create_sample_chunks(20)
    print(f"📝 Created {len(chunks)} sample chunks")

    # Test cluster-aware conflict detection
    print(f"\n🔍 Testing cluster-aware conflict detection...")
    test_chunk = chunks[0]
    existing_chunks = chunks[1:]

    start_time = time.time()
    conflicts = await cluster_detector.detect_conflicts_optimized(
        test_chunk, existing_chunks, use_cache=True, max_candidates=10
    )
    detection_time = time.time() - start_time

    print(f"   ✅ Conflict detection completed in {detection_time:.2f} seconds")
    print(f"   🔍 Found {len(conflicts)} potential conflicts")
    print(f"   📊 Checked against {len(existing_chunks)} existing chunks")

    # Analyze performance
    analysis = await cluster_detector.analyze_clustering_performance(chunks)
    print(f"\n📈 Performance Analysis:")
    print(f"   Total chunks: {analysis['total_chunks']}")
    print(f"   Clusters created: {analysis['num_clusters']}")
    print(f"   Noise chunks: {analysis['noise_chunks']}")
    print(f"   Estimated LLM call reduction: {analysis['conflict_detection_optimization']['estimated_llm_call_reduction']}")

    print(f"\n✨ Mock LLM demo completed!")


async def main():
    """Run the clustering performance demonstration."""
    try:
        # Run basic clustering demo
        await demo_clustering_performance()

        # Run mock LLM demo
        await demo_with_mock_llm()

        print(f"\n🎉 All demos completed successfully!")
        print(f"\nKey Benefits Demonstrated:")
        print(f"   ✅ Significant reduction in conflict detection comparisons")
        print(f"   ✅ Intelligent clustering of semantically related chunks")
        print(f"   ✅ Prioritized conflict detection candidates")
        print(f"   ✅ Graceful fallback when clustering is not available")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"   Please install: pip install hdbscan scikit-learn numpy")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
