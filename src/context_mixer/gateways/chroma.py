from pathlib import Path
from typing import List, Optional

from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResults
)
from context_mixer.gateways.adapters.chroma_adapter import ChromaAdapter
from context_mixer.gateways.chroma_connection_pool import ChromaConnectionPool


class ChromaGateway:
    """
    Gateway for knowledge storage and retrieval using ChromaDB.

    This gateway implements the CRAFT principle of Transcendence by providing
    a domain-focused interface that hides ChromaDB implementation details.
    All interactions use domain objects rather than exposing storage specifics.
    """

    def __init__(
        self, 
        db_dir: Path, 
        connection_pool: Optional[ChromaConnectionPool] = None,
        pool_size: int = 5,
        max_pool_size: int = 10,
        connection_timeout: float = 30.0
    ):
        """
        Initialize the ChromaGateway with a database directory.

        Args:
            db_dir: Path to the directory where ChromaDB will store data
            connection_pool: Optional existing connection pool to use
            pool_size: Initial number of connections in the pool (if creating new pool)
            max_pool_size: Maximum number of connections in the pool (if creating new pool)
            connection_timeout: Timeout in seconds for getting a connection (if creating new pool)
        """
        self.db_dir = db_dir
        self.adapter = ChromaAdapter()

        # Use provided connection pool or create a new one
        if connection_pool is not None:
            self.connection_pool = connection_pool
            self._owns_pool = False
        else:
            self.connection_pool = ChromaConnectionPool(
                db_dir=db_dir,
                pool_size=pool_size,
                max_pool_size=max_pool_size,
                connection_timeout=connection_timeout
            )
            self._owns_pool = True

        # Keep legacy client for backward compatibility (deprecated)
        self.chroma_client = None

    def _get_collection(self, client):
        """Get or create the default collection for knowledge storage."""
        return client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def store_knowledge_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """
        Store or update knowledge chunks in the database.

        Args:
            chunks: List of KnowledgeChunk domain objects to store
        """
        if not chunks:
            return

        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)
            chroma_data = self.adapter.chunks_to_chroma_format(chunks)

            # Use upsert to handle both new and updated chunks
            collection.upsert(
                ids=chroma_data["ids"],
                documents=chroma_data["documents"],
                metadatas=chroma_data["metadatas"],
                embeddings=chroma_data["embeddings"]
            )

    def search_knowledge(self, query: SearchQuery) -> SearchResults:
        """
        Search for knowledge chunks based on a query.

        Args:
            query: SearchQuery domain object specifying search criteria

        Returns:
            SearchResults containing matching knowledge chunks
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)

            # Convert domain query to ChromaDB parameters
            chroma_params = self.adapter.search_query_to_chroma_params(query)

            # Perform the search
            if hasattr(query, 'embedding') and query.embedding:
                # Vector similarity search with embedding
                chroma_results = collection.query(
                    query_embeddings=[query.embedding],
                    **chroma_params
                )
            else:
                # Text-based search (ChromaDB will generate embeddings)
                # Don't include embeddings in results to avoid dimension mismatches
                chroma_results = collection.query(
                    query_texts=[query.text],
                    include=["metadatas", "documents", "distances"],
                    **chroma_params
                )

            # Convert results back to domain objects
            return self.adapter.chroma_results_to_search_results(chroma_results, query)

    def get_knowledge_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """
        Retrieve a specific knowledge chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk

        Returns:
            KnowledgeChunk if found, None otherwise
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)

            results = collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])

            if not results["ids"]:
                return None

            # Convert ChromaDB result to domain object
            embedding = None
            if results["embeddings"] is not None and len(results["embeddings"]) > 0 and results["embeddings"][0] is not None:
                embedding = results["embeddings"][0]

            # Extract concept from metadata
            chroma_metadata = results["metadatas"][0]
            concept = chroma_metadata.get("concept", "") if isinstance(chroma_metadata, dict) else ""

            chunk_data = {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "concept": concept,
                "metadata": self.adapter._chroma_dict_to_metadata(chroma_metadata),
                "embedding": embedding
            }

            return KnowledgeChunk(**chunk_data)

    def get_all_chunks(self) -> List[KnowledgeChunk]:
        """
        Retrieve all knowledge chunks from the database.

        Returns:
            List of all KnowledgeChunk objects in the store
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)

            # Get all chunks from the collection
            results = collection.get(include=["documents", "metadatas", "embeddings"])

            if not results["ids"]:
                return []

            chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                embedding = None
                if (results["embeddings"] is not None and 
                    len(results["embeddings"]) > i and 
                    results["embeddings"][i] is not None):
                    embedding = results["embeddings"][i]

                # Extract concept from metadata
                chroma_metadata = results["metadatas"][i]
                concept = chroma_metadata.get("concept", "") if isinstance(chroma_metadata, dict) else ""

                chunk_data = {
                    "id": chunk_id,
                    "content": results["documents"][i],
                    "concept": concept,
                    "metadata": self.adapter._chroma_dict_to_metadata(chroma_metadata),
                    "embedding": embedding
                }

                chunks.append(KnowledgeChunk(**chunk_data))

            return chunks

    def delete_knowledge_chunk(self, chunk_id: str) -> bool:
        """
        Delete a knowledge chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk to delete

        Returns:
            True if chunk was deleted, False if not found
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)

            try:
                # Check if chunk exists first
                existing = collection.get(ids=[chunk_id])
                if not existing["ids"]:
                    return False

                collection.delete(ids=[chunk_id])
                return True

            except Exception:
                return False

    def reset_knowledge_store(self) -> None:
        """
        Reset the entire knowledge store, removing all data.

        Warning: This operation is irreversible and will delete all stored knowledge.
        """
        with self.connection_pool.get_connection() as connection:
            connection.client.reset()

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the knowledge collection.

        Returns:
            Dictionary with collection statistics
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)
            count = collection.count()

            return {
                "total_chunks": count,
                "collection_name": collection.name
            }

    def get_pool_stats(self) -> dict:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with connection pool statistics
        """
        return self.connection_pool.get_stats()

    def close(self) -> None:
        """
        Close the connection pool and clean up resources.

        This should be called when the gateway is no longer needed.
        """
        if self._owns_pool and self.connection_pool:
            self.connection_pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
