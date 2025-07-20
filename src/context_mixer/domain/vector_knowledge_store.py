"""
Vector-based knowledge store implementation using ChromaDB.

This module provides a concrete implementation of the KnowledgeStore interface
using ChromaDB as the vector database backend.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

from .knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResults,
    AuthorityLevel,
    TemporalScope
)
from .knowledge_store import KnowledgeStore, StorageError
from ..gateways.chroma import ChromaGateway


class VectorKnowledgeStore(KnowledgeStore):
    """
    Vector-based knowledge store implementation using ChromaDB.

    This implementation provides semantic search capabilities through vector
    embeddings while maintaining the storage-agnostic interface defined
    by the KnowledgeStore abstract class.
    """

    def __init__(
        self, 
        db_path: Path, 
        llm_gateway=None,
        pool_size: int = 5,
        max_pool_size: int = 10,
        connection_timeout: float = 30.0
    ):
        """
        Initialize the vector knowledge store.

        Args:
            db_path: Path to the ChromaDB database directory
            llm_gateway: Optional LLM gateway for conflict detection
            pool_size: Initial number of connections in the connection pool
            max_pool_size: Maximum number of connections in the connection pool
            connection_timeout: Timeout in seconds for getting a connection from the pool
        """
        self.db_path = db_path
        self._gateway: Optional[ChromaGateway] = None
        self._llm_gateway = llm_gateway
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.connection_timeout = connection_timeout

    def _get_gateway(self) -> ChromaGateway:
        """Get or create the ChromaDB gateway instance."""
        if self._gateway is None:
            self._gateway = ChromaGateway(
                db_dir=self.db_path,
                pool_size=self.pool_size,
                max_pool_size=self.max_pool_size,
                connection_timeout=self.connection_timeout
            )
        return self._gateway

    async def store_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """
        Store or update knowledge chunks in the vector database.

        Args:
            chunks: List of KnowledgeChunk objects to store

        Raises:
            StorageError: If storage operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, gateway.store_knowledge_chunks, chunks
            )
        except Exception as e:
            raise StorageError(f"Failed to store chunks: {str(e)}", e)

    async def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """
        Retrieve a specific knowledge chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk

        Returns:
            KnowledgeChunk if found, None otherwise

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, gateway.get_knowledge_chunk, chunk_id
            )
        except Exception as e:
            raise StorageError(f"Failed to retrieve chunk {chunk_id}: {str(e)}", e)

    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Search for knowledge chunks using vector similarity.

        Args:
            query: SearchQuery specifying search criteria and filters

        Returns:
            SearchResults containing matching chunks with relevance scores

        Raises:
            StorageError: If search operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, gateway.search_knowledge, query
            )
        except Exception as e:
            raise StorageError(f"Failed to search knowledge: {str(e)}", e)

    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a knowledge chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk to delete

        Returns:
            True if chunk was deleted, False if not found

        Raises:
            StorageError: If deletion operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, gateway.delete_knowledge_chunk, chunk_id
            )
        except Exception as e:
            raise StorageError(f"Failed to delete chunk {chunk_id}: {str(e)}", e)

    async def get_chunks_by_domain(self, domains: List[str]) -> List[KnowledgeChunk]:
        """
        Retrieve all chunks belonging to specific domains.

        Args:
            domains: List of domain names to filter by

        Returns:
            List of KnowledgeChunk objects in the specified domains

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            # Create a search query that filters by domains
            query = SearchQuery(
                text="*",  # Match all content
                domains=domains,
                max_results=1000  # Large number to get all matching chunks
            )
            results = await self.search(query)
            return results.get_chunks()
        except Exception as e:
            raise StorageError(f"Failed to retrieve chunks by domain: {str(e)}", e)

    async def get_chunks_by_authority(self, authority_levels: List[AuthorityLevel]) -> List[KnowledgeChunk]:
        """
        Retrieve all chunks with specific authority levels.

        Args:
            authority_levels: List of authority levels to filter by

        Returns:
            List of KnowledgeChunk objects with specified authority levels

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            # Create a search query that filters by authority levels
            query = SearchQuery(
                text="*",  # Match all content
                authority_levels=authority_levels,
                max_results=1000  # Large number to get all matching chunks
            )
            results = await self.search(query)
            return results.get_chunks()
        except Exception as e:
            raise StorageError(f"Failed to retrieve chunks by authority: {str(e)}", e)

    async def get_chunks_by_project(self, project_ids: List[str]) -> List[KnowledgeChunk]:
        """
        Retrieve all chunks belonging to specific projects.

        Args:
            project_ids: List of project IDs to filter by

        Returns:
            List of KnowledgeChunk objects belonging to the specified projects

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            # Create a search query that filters by project IDs
            query = SearchQuery(
                text="*",  # Match all content
                project_ids=project_ids,
                max_results=1000  # Large number to get all matching chunks
            )
            results = await self.search(query)
            return results.get_chunks()
        except Exception as e:
            raise StorageError(f"Failed to retrieve chunks by project: {str(e)}", e)

    async def get_chunks_excluding_projects(self, exclude_project_ids: List[str]) -> List[KnowledgeChunk]:
        """
        Retrieve all chunks excluding specific projects.

        Args:
            exclude_project_ids: List of project IDs to exclude

        Returns:
            List of KnowledgeChunk objects not belonging to the excluded projects

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            # Create a search query that excludes specific projects
            query = SearchQuery(
                text="*",  # Match all content
                exclude_projects=exclude_project_ids,
                max_results=1000  # Large number to get all matching chunks
            )
            results = await self.search(query)
            return results.get_chunks()
        except Exception as e:
            raise StorageError(f"Failed to retrieve chunks excluding projects: {str(e)}", e)

    async def detect_conflicts(self, chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
        """
        Detect potential conflicts with existing knowledge using comprehensive analysis.

        This implementation checks all chunks in the same domains, not just semantically
        similar ones, to ensure we catch conflicts that might use different wording
        but address the same topics.

        Args:
            chunk: KnowledgeChunk to check for conflicts

        Returns:
            List of potentially conflicting chunks

        Raises:
            StorageError: If conflict detection fails
        """
        try:
            conflicts = []

            # Get all chunks in the same domains as the input chunk
            # This ensures we check for conflicts across all related content,
            # not just semantically similar content
            for domain in chunk.metadata.domains:
                domain_query = SearchQuery(
                    text="*",  # Match all content
                    domains=[domain],
                    max_results=100  # Get more results to be comprehensive
                )
                domain_results = await self.search(domain_query)

                for result in domain_results.results:
                    candidate = result.chunk

                    # Skip the chunk itself
                    if candidate.id == chunk.id:
                        continue

                    # Skip if we've already checked this candidate
                    if candidate in conflicts:
                        continue

                    # Use LLM-based conflict detection for accurate analysis
                    if await self._llm_detect_conflict(chunk, candidate):
                        conflicts.append(candidate)

            return conflicts
        except Exception as e:
            raise StorageError(f"Failed to detect conflicts: {str(e)}", e)

    async def find_similar_chunks(self, chunk: KnowledgeChunk, similarity_threshold: float = 0.7) -> List[KnowledgeChunk]:
        """
        Find semantically similar chunks for deduplication purposes.

        This method finds chunks with similar content regardless of metadata differences,
        which is useful for deduplication during assembly.

        Args:
            chunk: KnowledgeChunk to find similar chunks for
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of semantically similar chunks

        Raises:
            StorageError: If similarity detection fails
        """
        try:
            # Search for semantically similar chunks
            query = SearchQuery(
                text=chunk.content,
                max_results=20,
                min_relevance_score=similarity_threshold
            )
            results = await self.search(query)

            similar_chunks = []
            for result in results.results:
                candidate = result.chunk

                # Skip the chunk itself
                if candidate.id == chunk.id:
                    continue

                # For deduplication, we consider chunks similar if they have high semantic similarity
                # regardless of metadata differences (unlike conflict detection)
                similar_chunks.append(candidate)

            return similar_chunks
        except Exception as e:
            raise StorageError(f"Failed to find similar chunks: {str(e)}", e)

    def _is_potential_conflict(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
        """
        Check if two chunks are potentially conflicting based on metadata.

        This is a fast, rule-based check that doesn't require LLM inference.

        Args:
            chunk1: First chunk to compare
            chunk2: Second chunk to compare

        Returns:
            True if chunks are potentially conflicting
        """
        # Check if chunks are in different domains - no conflict if so
        common_domains = set(chunk1.metadata.domains) & set(chunk2.metadata.domains)
        if not common_domains:
            return False

        # Check for explicit conflicts in metadata
        if chunk1.id in chunk2.metadata.conflicts or chunk2.id in chunk1.metadata.conflicts:
            return True

        # Check for authority conflicts (different authority levels)
        if chunk1.metadata.authority != chunk2.metadata.authority:
            return True

        # Check for temporal conflicts (current vs deprecated)
        if (chunk1.metadata.temporal == TemporalScope.CURRENT and 
            chunk2.metadata.temporal == TemporalScope.DEPRECATED) or \
           (chunk1.metadata.temporal == TemporalScope.DEPRECATED and 
            chunk2.metadata.temporal == TemporalScope.CURRENT):
            return True

        return False

    async def _llm_detect_conflict(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
        """
        Use LLM to determine if two chunks are conflicting.

        This method uses the same LLM-based approach as the merge operations,
        providing a simple, extensible way to detect conflicts without
        hardcoded rules.

        Args:
            chunk1: First chunk to compare
            chunk2: Second chunk to compare

        Returns:
            True if chunks are conflicting
        """
        try:
            # Check if chunks are in the same domains (quick filter)
            common_domains = set(chunk1.metadata.domains) & set(chunk2.metadata.domains)
            if not common_domains:
                return False

            # Check for explicit conflicts in metadata
            if chunk1.id in chunk2.metadata.conflicts or chunk2.id in chunk1.metadata.conflicts:
                return True

            # Use LLM to detect semantic conflicts if gateway is available
            if self._llm_gateway:
                from context_mixer.commands.operations.merge import detect_conflicts
                conflicts = detect_conflicts(chunk1.content, chunk2.content, self._llm_gateway)
                return len(conflicts.list) > 0

            # If no LLM gateway available, fall back to basic checks
            # Check for temporal conflicts
            if (chunk1.metadata.temporal == TemporalScope.CURRENT and 
                chunk2.metadata.temporal == TemporalScope.DEPRECATED):
                return True

            return False

        except Exception:
            # If LLM detection fails, fall back to basic checks
            # Check for temporal conflicts
            if (chunk1.metadata.temporal == TemporalScope.CURRENT and 
                chunk2.metadata.temporal == TemporalScope.DEPRECATED):
                return True

            return False

    async def validate_dependencies(self, chunk: KnowledgeChunk) -> List[str]:
        """
        Validate that all dependencies for a chunk exist.

        Args:
            chunk: KnowledgeChunk to validate dependencies for

        Returns:
            List of missing dependency IDs (empty if all dependencies exist)

        Raises:
            StorageError: If validation operation fails
        """
        try:
            missing_deps = []

            for dep_id in chunk.metadata.dependencies:
                dep_chunk = await self.get_chunk(dep_id)
                if dep_chunk is None:
                    missing_deps.append(dep_id)

            return missing_deps
        except Exception as e:
            raise StorageError(f"Failed to validate dependencies: {str(e)}", e)

    async def get_all_chunks(self) -> List[KnowledgeChunk]:
        """
        Retrieve all chunks from the knowledge store.

        Returns:
            List of all KnowledgeChunk objects in the store

        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            all_chunks = await asyncio.get_event_loop().run_in_executor(
                None, gateway.get_all_chunks
            )
            return all_chunks
        except Exception as e:
            raise StorageError(f"Failed to retrieve all chunks: {str(e)}", e)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge store.

        Returns:
            Dictionary containing store statistics

        Raises:
            StorageError: If stats retrieval fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            chroma_stats = await asyncio.get_event_loop().run_in_executor(
                None, gateway.get_collection_stats
            )

            # Get connection pool statistics
            pool_stats = await asyncio.get_event_loop().run_in_executor(
                None, gateway.get_pool_stats
            )

            # Enhance with additional statistics
            stats = {
                "total_chunks": chroma_stats["total_chunks"],
                "collection_name": chroma_stats["collection_name"],
                "storage_type": "vector",
                "backend": "chromadb",
                "db_path": str(self.db_path),
                "connection_pool": pool_stats
            }

            # Add domain and authority distribution if we have chunks
            if stats["total_chunks"] > 0:
                try:
                    # Get a sample of chunks to analyze distribution
                    sample_query = SearchQuery(text="*", max_results=100)
                    sample_results = await self.search(sample_query)

                    domains = set()
                    authorities = set()
                    for chunk in sample_results.get_chunks():
                        domains.update(chunk.metadata.domains)
                        authorities.add(chunk.metadata.authority.value)

                    stats["domains"] = sorted(list(domains))
                    stats["authority_levels"] = sorted(list(authorities))
                except Exception:
                    # If sampling fails, just continue without these stats
                    pass

            return stats
        except Exception as e:
            raise StorageError(f"Failed to get stats: {str(e)}", e)

    async def reset(self) -> None:
        """
        Reset the knowledge store, removing all data.

        Warning: This operation is irreversible.

        Raises:
            StorageError: If reset operation fails
        """
        try:
            gateway = self._get_gateway()
            # Run the synchronous gateway method in a thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, gateway.reset_knowledge_store
            )
        except Exception as e:
            raise StorageError(f"Failed to reset knowledge store: {str(e)}", e)

    async def close(self) -> None:
        """
        Close the knowledge store and clean up resources.

        This should be called when the store is no longer needed to properly
        close the connection pool and free resources.
        """
        if self._gateway:
            # Run the synchronous gateway close method in a thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, self._gateway.close
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
