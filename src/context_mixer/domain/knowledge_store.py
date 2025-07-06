"""
Knowledge Store interface and implementations following CRAFT principles.

This module provides storage-agnostic interfaces for knowledge management,
implementing the CRAFT principle of Transcendence by abstracting storage
details behind domain-focused interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .vector_knowledge_store import VectorKnowledgeStore
    from .file_knowledge_store import FileKnowledgeStore
    from .hybrid_knowledge_store import HybridKnowledgeStore

from .knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResults,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)


class KnowledgeStore(ABC):
    """
    Abstract interface for knowledge storage and retrieval.

    This interface implements the CRAFT principle of Transcendence by providing
    storage-agnostic operations that work regardless of the underlying storage
    mechanism (vector database, graph database, file system, etc.).
    """

    @abstractmethod
    async def store_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """
        Store or update knowledge chunks.

        Args:
            chunks: List of KnowledgeChunk objects to store

        Raises:
            StorageError: If storage operation fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Search for knowledge chunks based on query criteria.

        Args:
            query: SearchQuery specifying search criteria and filters

        Returns:
            SearchResults containing matching chunks with relevance scores

        Raises:
            StorageError: If search operation fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def detect_conflicts(self, chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
        """
        Detect potential conflicts with existing knowledge.

        Args:
            chunk: KnowledgeChunk to check for conflicts

        Returns:
            List of potentially conflicting chunks

        Raises:
            StorageError: If conflict detection fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge store.

        Returns:
            Dictionary containing store statistics (chunk count, domains, etc.)

        Raises:
            StorageError: If stats retrieval fails
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """
        Reset the knowledge store, removing all data.

        Warning: This operation is irreversible.

        Raises:
            StorageError: If reset operation fails
        """
        pass


class StorageError(Exception):
    """Exception raised when storage operations fail."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class KnowledgeStoreFactory:
    """
    Factory for creating KnowledgeStore instances.

    This factory implements the CRAFT principle of Transcendence by allowing
    different storage backends to be created through a unified interface.
    """

    @staticmethod
    def create_vector_store(db_path: Path) -> "VectorKnowledgeStore":
        """
        Create a vector-based knowledge store using ChromaDB.

        Args:
            db_path: Path to the database directory

        Returns:
            VectorKnowledgeStore instance
        """
        from .vector_knowledge_store import VectorKnowledgeStore
        return VectorKnowledgeStore(db_path)

    @staticmethod
    def create_file_store(storage_path: Path) -> "FileKnowledgeStore":
        """
        Create a file-based knowledge store.

        Args:
            storage_path: Path to the storage directory

        Returns:
            FileKnowledgeStore instance
        """
        from .file_knowledge_store import FileKnowledgeStore
        return FileKnowledgeStore(storage_path)

    @staticmethod
    def create_hybrid_store(
        vector_db_path: Path,
        file_storage_path: Path
    ) -> "HybridKnowledgeStore":
        """
        Create a hybrid knowledge store combining vector and file storage.

        Args:
            vector_db_path: Path to the vector database directory
            file_storage_path: Path to the file storage directory

        Returns:
            HybridKnowledgeStore instance
        """
        from .hybrid_knowledge_store import HybridKnowledgeStore
        return HybridKnowledgeStore(vector_db_path, file_storage_path)
