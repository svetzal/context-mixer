"""
Vector-based knowledge store implementation using ChromaDB.

This module provides a concrete implementation of the KnowledgeStore interface
using ChromaDB as the vector database backend.
"""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from .knowledge_store import KnowledgeStore, StorageError
from .knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResults,
    SearchResult,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)
from ..gateways.chroma import ChromaGateway


class VectorKnowledgeStore(KnowledgeStore):
    """
    Vector-based knowledge store implementation using ChromaDB.
    
    This implementation provides semantic search capabilities through vector
    embeddings while maintaining the storage-agnostic interface defined
    by the KnowledgeStore abstract class.
    """

    def __init__(self, db_path: Path):
        """
        Initialize the vector knowledge store.
        
        Args:
            db_path: Path to the ChromaDB database directory
        """
        self.db_path = db_path
        self._gateway: Optional[ChromaGateway] = None

    def _get_gateway(self) -> ChromaGateway:
        """Get or create the ChromaDB gateway instance."""
        if self._gateway is None:
            self._gateway = ChromaGateway(self.db_path)
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

    async def detect_conflicts(self, chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
        """
        Detect potential conflicts with existing knowledge using semantic similarity.
        
        This implementation uses vector similarity to find chunks with similar
        content but potentially conflicting metadata (different authority levels,
        temporal scopes, etc.).
        
        Args:
            chunk: KnowledgeChunk to check for conflicts
            
        Returns:
            List of potentially conflicting chunks
            
        Raises:
            StorageError: If conflict detection fails
        """
        try:
            # Search for semantically similar chunks
            query = SearchQuery(
                text=chunk.content,
                max_results=20,
                min_relevance_score=0.7  # High similarity threshold
            )
            results = await self.search(query)
            
            conflicts = []
            for result in results.results:
                candidate = result.chunk
                
                # Skip the chunk itself
                if candidate.id == chunk.id:
                    continue
                
                # Check for potential conflicts
                if self._is_potential_conflict(chunk, candidate):
                    conflicts.append(candidate)
            
            return conflicts
        except Exception as e:
            raise StorageError(f"Failed to detect conflicts: {str(e)}", e)

    def _is_potential_conflict(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
        """
        Determine if two chunks are potentially conflicting.
        
        Args:
            chunk1: First chunk to compare
            chunk2: Second chunk to compare
            
        Returns:
            True if chunks are potentially conflicting
        """
        # Check if chunks are in the same domains
        common_domains = set(chunk1.metadata.domains) & set(chunk2.metadata.domains)
        if not common_domains:
            return False
        
        # Check for explicit conflicts
        if chunk1.id in chunk2.metadata.conflicts or chunk2.id in chunk1.metadata.conflicts:
            return True
        
        # Check for authority conflicts (different authority levels for similar content)
        if chunk1.metadata.authority != chunk2.metadata.authority:
            return True
        
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
            
            # Enhance with additional statistics
            stats = {
                "total_chunks": chroma_stats["total_chunks"],
                "collection_name": chroma_stats["collection_name"],
                "storage_type": "vector",
                "backend": "chromadb",
                "db_path": str(self.db_path)
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