from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import numpy as np

from context_mixer.domain.knowledge import (
    KnowledgeChunk, 
    SearchQuery, 
    SearchResults
)
from context_mixer.gateways.adapters.chroma_adapter import ChromaAdapter
from context_mixer.gateways.chroma_connection_pool import ChromaConnectionPool


class SimpleLocalEmbeddingFunction(EmbeddingFunction):
    """
    Simple local embedding function that doesn't require downloads.
    Uses basic text hashing for embeddings - suitable for testing and offline use.
    """
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate simple hash-based embeddings for the input documents."""
        embeddings = []
        for doc in input:
            # Create a simple hash-based embedding
            # This is not semantically meaningful but allows offline operation
            hash_val = hash(doc)
            # Create a 384-dimensional vector (same as many embedding models)
            # Use the hash to seed a random number generator for consistency
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(384).astype(np.float32)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        return embeddings


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
            embedding_function=SimpleLocalEmbeddingFunction()
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

            chunk_data = {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": self.adapter._chroma_dict_to_metadata(results["metadatas"][0]),
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

                chunk_data = {
                    "id": chunk_id,
                    "content": results["documents"][i],
                    "metadata": self.adapter._chroma_dict_to_metadata(results["metadatas"][i]),
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

    def _get_embedding_for_chunk(self, chunk: KnowledgeChunk) -> Optional[List[float]]:
        """
        Get embedding for a single chunk.
        
        Args:
            chunk: KnowledgeChunk to get embedding for
            
        Returns:
            Embedding vector as list of floats, or None if not found
        """
        with self.connection_pool.get_connection() as connection:
            collection = self._get_collection(connection.client)
            
            # First check if chunk already has an embedding
            if chunk.embedding:
                return chunk.embedding
            
            # Try to retrieve existing embedding from ChromaDB
            results = collection.get(ids=[chunk.id], include=["embeddings"])
            
            if (results["ids"] and results["embeddings"] and 
                len(results["embeddings"]) > 0 and results["embeddings"][0] is not None):
                return results["embeddings"][0]
            
            # If no existing embedding, generate one by adding the chunk temporarily
            try:
                chroma_data = self.adapter.chunks_to_chroma_format([chunk])
                collection.upsert(
                    ids=chroma_data["ids"],
                    documents=chroma_data["documents"], 
                    metadatas=chroma_data["metadatas"],
                    embeddings=chroma_data["embeddings"]  # Will be generated if None
                )
                
                # Retrieve the generated embedding
                results = collection.get(ids=[chunk.id], include=["embeddings"])
                if (results["embeddings"] and len(results["embeddings"]) > 0 and 
                    results["embeddings"][0] is not None):
                    return results["embeddings"][0]
            
            except Exception:
                pass
            
            return None

    def _get_embeddings_for_chunks(self, chunk_ids: List[str]) -> Optional[List[List[float]]]:
        """
        Get embeddings for multiple chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to get embeddings for
            
        Returns:
            List of embedding vectors, or None if retrieval fails
        """
        if not chunk_ids:
            return []
        
        try:
            with self.connection_pool.get_connection() as connection:
                collection = self._get_collection(connection.client)
                
                results = collection.get(ids=chunk_ids, include=["embeddings"])
                
                if results["embeddings"] is None or len(results["embeddings"]) == 0:
                    return None
                
                # Filter out None embeddings and maintain order
                embeddings = []
                for i, embedding in enumerate(results["embeddings"]):
                    if embedding is not None:
                        embeddings.append(embedding)
                
                return embeddings if embeddings else None
                
        except Exception:
            return None

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
