from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb import Settings

from context_mixer.domain.knowledge import (
    KnowledgeChunk, 
    SearchQuery, 
    SearchResults
)
from context_mixer.gateways.adapters.chroma_adapter import ChromaAdapter


class ChromaGateway:
    """
    Gateway for knowledge storage and retrieval using ChromaDB.

    This gateway implements the CRAFT principle of Transcendence by providing
    a domain-focused interface that hides ChromaDB implementation details.
    All interactions use domain objects rather than exposing storage specifics.
    """

    def __init__(self, db_dir: Path):
        """
        Initialize the ChromaGateway with a database directory.

        Args:
            db_dir: Path to the directory where ChromaDB will store data
        """
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(allow_reset=True),
        )
        self.adapter = ChromaAdapter()

    def _get_collection(self):
        """Get or create the default collection for knowledge storage."""
        return self.chroma_client.get_or_create_collection(
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

        collection = self._get_collection()
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
        collection = self._get_collection()

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
        collection = self._get_collection()

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

    def delete_knowledge_chunk(self, chunk_id: str) -> bool:
        """
        Delete a knowledge chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk to delete

        Returns:
            True if chunk was deleted, False if not found
        """
        collection = self._get_collection()

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
        self.chroma_client.reset()

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the knowledge collection.

        Returns:
            Dictionary with collection statistics
        """
        collection = self._get_collection()
        count = collection.count()

        return {
            "total_chunks": count,
            "collection_name": collection.name
        }
