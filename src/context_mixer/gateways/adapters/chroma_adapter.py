"""
Adapter for converting between ChromaDB structures and Context Mixer domain objects.

This adapter implements the CRAFT principle of Transcendence by isolating
ChromaDB implementation details from the domain layer.
"""

from typing import List, Dict, Any

from context_mixer.domain.knowledge import (
    KnowledgeChunk,
    SearchQuery,
    SearchResult,
    SearchResults,
    ChunkMetadata,
    ProvenanceInfo,
    AuthorityLevel,
    GranularityLevel,
    TemporalScope
)


class ChromaAdapter:
    """
    Adapter for converting between ChromaDB structures and domain objects.

    This class handles the conversion between ChromaDB's raw data structures
    and our domain-specific knowledge objects, following the adapter pattern
    to isolate storage implementation details.
    """

    def chunks_to_chroma_format(self, chunks: List[KnowledgeChunk]) -> Dict[str, List[Any]]:
        """
        Convert domain KnowledgeChunk objects to ChromaDB format.

        Args:
            chunks: List of KnowledgeChunk domain objects

        Returns:
            Dictionary with ChromaDB format: {ids, documents, metadatas, embeddings}
        """
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append(self._metadata_to_chroma_dict(chunk.metadata, chunk.concept))

            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                # ChromaDB will generate embeddings if not provided
                embeddings.append(None)

        # Filter out None embeddings if all are None
        if all(emb is None for emb in embeddings):
            embeddings = None

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings
        }

    def chroma_results_to_search_results(self, 
                                       chroma_results: Dict[str, Any], 
                                       query: SearchQuery) -> SearchResults:
        """
        Convert ChromaDB query results to domain SearchResults.

        Args:
            chroma_results: Raw results from ChromaDB query
            query: The original search query

        Returns:
            SearchResults domain object
        """
        results = []

        # ChromaDB returns results in lists indexed by position
        ids = chroma_results.get('ids', [[]])[0]  # ChromaDB wraps in extra list
        documents = chroma_results.get('documents', [[]])[0]
        metadatas = chroma_results.get('metadatas', [[]])[0]
        distances = chroma_results.get('distances', [[]])[0]

        # Handle embeddings safely - they might be None or not included
        embeddings = []
        if 'embeddings' in chroma_results and chroma_results['embeddings'] is not None:
            embeddings = chroma_results['embeddings'][0]
        else:
            embeddings = [None] * len(ids)

        for i, chunk_id in enumerate(ids):
            # Convert distance to relevance score (ChromaDB uses cosine distance)
            # Lower distance = higher relevance, so we invert it
            distance = distances[i] if i < len(distances) else 1.0
            relevance_score = max(0.0, 1.0 - distance)

            # Skip results below minimum relevance threshold
            if relevance_score < query.min_relevance_score:
                continue

            # Extract concept from ChromaDB metadata
            chunk_metadata = metadatas[i] if i < len(metadatas) else {}
            concept = chunk_metadata.get("concept", "") if isinstance(chunk_metadata, dict) else ""

            # Reconstruct KnowledgeChunk from ChromaDB data
            chunk = KnowledgeChunk(
                id=chunk_id,
                content=documents[i] if i < len(documents) else "",
                concept=concept,
                metadata=self._chroma_dict_to_metadata(chunk_metadata),
                embedding=embeddings[i] if embeddings[i] is not None else None
            )

            # Apply domain filtering at application level
            if query.domains:
                chunk_domains = chunk.metadata.domains
                if not any(domain in chunk_domains for domain in query.domains):
                    continue

            # Apply scope filtering at application level
            if query.scopes:
                chunk_scopes = chunk.metadata.scope
                if not any(scope in chunk_scopes for scope in query.scopes):
                    continue

            result = SearchResult(
                chunk=chunk,
                relevance_score=relevance_score,
                match_explanation=f"Semantic similarity score: {relevance_score:.3f}"
            )

            results.append(result)

        return SearchResults(
            query=query,
            results=results,
            total_found=len(results)
        )

    def _metadata_to_chroma_dict(self, metadata: ChunkMetadata, concept: str = "") -> Dict[str, Any]:
        """Convert ChunkMetadata to ChromaDB metadata dictionary."""
        return {
            # Store concept for conflict detection context
            "concept": concept,
            # Convert lists to comma-separated strings for ChromaDB compatibility
            "domains": ",".join(metadata.domains) if metadata.domains else "",
            "authority": metadata.authority.value,
            "scope": ",".join(metadata.scope) if metadata.scope else "",
            "granularity": metadata.granularity.value,
            "temporal": metadata.temporal.value,
            "dependencies": ",".join(metadata.dependencies) if metadata.dependencies else "",
            "conflicts": ",".join(metadata.conflicts) if metadata.conflicts else "",
            "tags": ",".join(metadata.tags) if metadata.tags else "",
            "source": metadata.provenance.source,
            "project_id": metadata.provenance.project_id or "",
            "project_name": metadata.provenance.project_name or "",
            "project_path": metadata.provenance.project_path or "",
            "created_at": metadata.provenance.created_at,
            "updated_at": metadata.provenance.updated_at or "",
            "author": metadata.provenance.author or ""
        }

    def _chroma_dict_to_metadata(self, chroma_dict: Dict[str, Any]) -> ChunkMetadata:
        """Convert ChromaDB metadata dictionary to ChunkMetadata."""
        # Helper function to convert comma-separated strings back to lists
        def _string_to_list(value: Any) -> List[str]:
            if isinstance(value, str) and value.strip():
                return [item.strip() for item in value.split(",") if item.strip()]
            return []

        # Provide defaults for missing fields to handle legacy data
        provenance = ProvenanceInfo(
            source=chroma_dict.get("source", "unknown"),
            project_id=chroma_dict.get("project_id") or None,
            project_name=chroma_dict.get("project_name") or None,
            project_path=chroma_dict.get("project_path") or None,
            created_at=chroma_dict.get("created_at", "unknown"),
            updated_at=chroma_dict.get("updated_at") or None,
            author=chroma_dict.get("author") or None
        )

        return ChunkMetadata(
            domains=_string_to_list(chroma_dict.get("domains", "")),
            authority=AuthorityLevel(chroma_dict.get("authority", AuthorityLevel.CONVENTIONAL.value)),
            scope=_string_to_list(chroma_dict.get("scope", "")),
            granularity=GranularityLevel(chroma_dict.get("granularity", GranularityLevel.DETAILED.value)),
            temporal=TemporalScope(chroma_dict.get("temporal", TemporalScope.CURRENT.value)),
            dependencies=_string_to_list(chroma_dict.get("dependencies", "")),
            conflicts=_string_to_list(chroma_dict.get("conflicts", "")),
            tags=_string_to_list(chroma_dict.get("tags", "")),
            provenance=provenance
        )

    def search_query_to_chroma_params(self, query: SearchQuery) -> Dict[str, Any]:
        """
        Convert SearchQuery to ChromaDB query parameters.

        Args:
            query: Domain SearchQuery object

        Returns:
            Dictionary of ChromaDB query parameters
        """
        params = {
            "n_results": query.max_results
        }

        # Build where clause for metadata filtering
        # Note: For now, we'll implement simple filtering and do domain/scope filtering
        # at the application level since ChromaDB doesn't support string contains operations
        where_conditions = {}

        if query.authority_levels:
            authority_values = [level.value for level in query.authority_levels]
            where_conditions["authority"] = {"$in": authority_values}

        if query.granularity:
            where_conditions["granularity"] = query.granularity.value

        # Add project filtering support
        if query.project_ids:
            # Include only chunks from specified projects
            where_conditions["project_id"] = {"$in": query.project_ids}

        if query.exclude_projects:
            # Exclude chunks from specified projects
            where_conditions["project_id"] = {"$nin": query.exclude_projects}

        # Only add where clause if we have conditions
        if where_conditions:
            params["where"] = where_conditions

        return params
