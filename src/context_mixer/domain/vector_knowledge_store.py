"""
Vector-based knowledge store implementation using ChromaDB.

This module provides a concrete implementation of the KnowledgeStore interface
using ChromaDB as the vector database backend with HDBSCAN clustering optimization.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

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
from .clustering import KnowledgeClusterer, ClusteringConfig
from ..gateways.chroma import ChromaGateway


class VectorKnowledgeStore(KnowledgeStore):
    """
    Vector-based knowledge store implementation using ChromaDB.

    This implementation provides semantic search capabilities through vector
    embeddings while maintaining the storage-agnostic interface defined
    by the KnowledgeStore abstract class. It includes HDBSCAN clustering
    optimization for conflict detection.
    """

    def __init__(
        self, 
        db_path: Path, 
        llm_gateway=None,
        pool_size: int = 5,
        max_pool_size: int = 10,
        connection_timeout: float = 30.0,
        clustering_config: Optional[ClusteringConfig] = None,
        enable_clustering: bool = True
    ):
        """
        Initialize the vector knowledge store.

        Args:
            db_path: Path to the ChromaDB database directory
            llm_gateway: Optional LLM gateway for conflict detection
            pool_size: Initial number of connections in the connection pool
            max_pool_size: Maximum number of connections in the connection pool
            connection_timeout: Timeout in seconds for getting a connection from the pool
            clustering_config: Configuration for HDBSCAN clustering
            enable_clustering: Whether to enable clustering optimization
        """
        self.db_path = db_path
        self._gateway: Optional[ChromaGateway] = None
        self._llm_gateway = llm_gateway
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.connection_timeout = connection_timeout
        
        # Clustering components
        self.enable_clustering = enable_clustering
        self.clustering_config = clustering_config or ClusteringConfig()
        self._clusterer: Optional[KnowledgeClusterer] = None
        self._clusters_dirty = True  # Flag to track if clusters need rebuilding
        
        if self.enable_clustering:
            try:
                self._clusterer = KnowledgeClusterer(self.clustering_config)
                logging.info("HDBSCAN clustering enabled for conflict detection optimization")
            except ImportError:
                logging.warning("HDBSCAN not available. Clustering disabled.")
                self.enable_clustering = False

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
            # Mark clusters as dirty since we added new chunks
            self._clusters_dirty = True
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
        Detect potential conflicts with existing knowledge using clustering optimization.

        This implementation uses HDBSCAN clustering to reduce the number of expensive
        LLM-based conflict checks by only comparing chunks within the same or nearby clusters.

        Args:
            chunk: KnowledgeChunk to check for conflicts

        Returns:
            List of potentially conflicting chunks

        Raises:
            StorageError: If conflict detection fails
        """
        try:
            conflicts = []

            if self.enable_clustering and self._clusterer:
                # Use cluster-based conflict detection
                conflicts = await self._cluster_based_conflict_detection(chunk)
            else:
                # Fall back to the original domain-based approach
                conflicts = await self._domain_based_conflict_detection(chunk)

            return conflicts
        except Exception as e:
            raise StorageError(f"Failed to detect conflicts: {str(e)}", e)

    async def _cluster_based_conflict_detection(self, chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
        """
        Cluster-optimized conflict detection using HDBSCAN.
        
        This method significantly reduces the number of conflict checks by:
        1. Ensuring clusters are up to date
        2. Predicting which cluster the new chunk belongs to
        3. Only checking conflicts within the same cluster and nearby clusters
        
        Args:
            chunk: KnowledgeChunk to check for conflicts
            
        Returns:
            List of potentially conflicting chunks
        """
        conflicts = []
        
        # Ensure clusters are up to date
        await self._ensure_clusters_updated()
        
        if not self._clusterer or not self._clusterer._fitted:
            # If clustering failed, fall back to domain-based detection
            logging.warning("Clustering not available, falling back to domain-based conflict detection")
            return await self._domain_based_conflict_detection(chunk)
        
        try:
            # Get embedding for the new chunk
            gateway = self._get_gateway()
            chunk_embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gateway._get_embedding_for_chunk(chunk)
            )
            
            if chunk_embedding is None:
                # If we can't get embedding, fall back to domain-based detection
                return await self._domain_based_conflict_detection(chunk)
            
            # Predict cluster for the new chunk
            predicted_cluster, confidence = self._clusterer.predict_cluster(chunk_embedding)
            
            # Get chunks to check based on cluster prediction
            candidate_chunks = []
            
            if predicted_cluster != -1:
                # Check chunks in the same cluster
                same_cluster_chunk_ids = self._clusterer.get_chunks_in_cluster(predicted_cluster)
                same_cluster_chunks = await self._get_chunks_by_ids(list(same_cluster_chunk_ids))
                candidate_chunks.extend(same_cluster_chunks)
                
                # Check chunks in nearby clusters (for boundary cases)
                nearby_clusters = self._clusterer.get_nearby_clusters(predicted_cluster)
                for nearby_cluster in nearby_clusters[:3]:  # Limit to 3 nearby clusters
                    nearby_chunk_ids = self._clusterer.get_chunks_in_cluster(nearby_cluster)
                    nearby_chunks = await self._get_chunks_by_ids(list(nearby_chunk_ids))
                    candidate_chunks.extend(nearby_chunks)
            else:
                # If predicted as noise, check against a sample of existing chunks
                # to avoid missing potential conflicts
                all_chunks = await self.get_all_chunks()
                # Sample up to 50 chunks for conflict checking (much better than checking all)
                candidate_chunks = all_chunks[:50] if len(all_chunks) > 50 else all_chunks
            
            # Now perform LLM-based conflict detection only on candidate chunks
            for candidate in candidate_chunks:
                # Skip the chunk itself
                if candidate.id == chunk.id:
                    continue
                
                # Skip if we've already checked this candidate
                if candidate in conflicts:
                    continue
                
                # Use LLM-based conflict detection for accurate analysis
                if await self._llm_detect_conflict(chunk, candidate):
                    conflicts.append(candidate)
            
            # Log performance improvement
            total_chunks = len(await self.get_all_chunks())
            checked_chunks = len(candidate_chunks)
            if total_chunks > 0:
                reduction = (1 - checked_chunks / total_chunks) * 100
                logging.info(f"Clustering optimization: checked {checked_chunks}/{total_chunks} chunks "
                           f"({reduction:.1f}% reduction in conflict checks)")
            
            return conflicts
            
        except Exception as e:
            logging.warning(f"Cluster-based conflict detection failed: {e}. "
                          f"Falling back to domain-based detection.")
            return await self._domain_based_conflict_detection(chunk)

    async def _domain_based_conflict_detection(self, chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
        """
        Original domain-based conflict detection (fallback method).
        
        This method checks all chunks in the same domains, which can be expensive
        for large knowledge bases but provides comprehensive conflict detection.
        
        Args:
            chunk: KnowledgeChunk to check for conflicts
            
        Returns:
            List of potentially conflicting chunks
        """
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

    async def _ensure_clusters_updated(self) -> None:
        """
        Ensure that clusters are up to date with the current knowledge base.
        
        This method rebuilds clusters if they are marked as dirty (e.g., after
        new chunks are added to the knowledge store).
        """
        if not self.enable_clustering or not self._clusterer:
            return
        
        if not self._clusters_dirty and self._clusterer._fitted:
            return  # Clusters are up to date
        
        try:
            # Get all chunks and their embeddings
            all_chunks = await self.get_all_chunks()
            
            if len(all_chunks) < self.clustering_config.min_cluster_size:
                logging.info(f"Not enough chunks ({len(all_chunks)}) for clustering. "
                           f"Minimum required: {self.clustering_config.min_cluster_size}")
                return
            
            # Get embeddings for all chunks
            gateway = self._get_gateway()
            chunk_ids = [chunk.id for chunk in all_chunks]
            embeddings_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gateway._get_embeddings_for_chunks(chunk_ids)
            )
            
            if embeddings_data is None or len(embeddings_data) == 0:
                logging.warning("Could not retrieve embeddings for clustering")
                return
            
            embeddings = np.array(embeddings_data)
            
            # Fit the clusterer
            logging.info(f"Rebuilding clusters for {len(all_chunks)} chunks")
            cluster_metadata = self._clusterer.fit(embeddings, chunk_ids)
            
            # Log clustering results
            stats = self._clusterer.get_cluster_stats()
            logging.info(f"Clustering complete: {stats}")
            
            self._clusters_dirty = False
            
        except Exception as e:
            logging.error(f"Failed to update clusters: {e}")
            # Don't raise - clustering is an optimization, not a requirement

    async def _get_chunks_by_ids(self, chunk_ids: List[str]) -> List[KnowledgeChunk]:
        """
        Retrieve multiple chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of KnowledgeChunk objects (may be shorter than input if some IDs not found)
        """
        chunks = []
        for chunk_id in chunk_ids:
            chunk = await self.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    async def rebuild_clusters(self) -> Dict[str, Any]:
        """
        Manually rebuild clusters and return clustering statistics.
        
        This method can be called to force a cluster rebuild, which might be
        useful after significant changes to the knowledge base.
        
        Returns:
            Dictionary with clustering statistics
        """
        if not self.enable_clustering:
            return {"error": "Clustering is disabled"}
        
        self._clusters_dirty = True
        await self._ensure_clusters_updated()
        
        if self._clusterer:
            return self._clusterer.get_cluster_stats()
        else:
            return {"error": "Clusterer not available"}

    async def get_cluster_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cluster information for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to get cluster info for
            
        Returns:
            Dictionary with cluster information or None if not available
        """
        if not self.enable_clustering or not self._clusterer:
            return None
        
        await self._ensure_clusters_updated()
        
        if not self._clusterer._fitted:
            return None
        
        cluster_id = self._clusterer.get_cluster_for_chunk(chunk_id)
        cluster_chunks = self._clusterer.get_chunks_in_cluster(cluster_id)
        
        return {
            "chunk_id": chunk_id,
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_chunks),
            "is_noise": cluster_id == -1,
            "cluster_chunks": list(cluster_chunks)
        }

    def _fast_conflict_check(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
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
        Get statistics about the knowledge store including clustering information.

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
                "connection_pool": pool_stats,
                "clustering_enabled": self.enable_clustering
            }

            # Add clustering statistics if available
            if self.enable_clustering and self._clusterer:
                await self._ensure_clusters_updated()
                cluster_stats = self._clusterer.get_cluster_stats()
                stats["clustering"] = cluster_stats
            
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

    async def detect_conflicts_batch_with_clustering(self, chunks: List[KnowledgeChunk]) -> List[tuple]:
        """
        Detect conflicts between chunks in a batch using clustering optimization.
        
        This method creates temporary clusters for the batch of chunks and only
        performs conflict detection between chunks in the same or nearby clusters,
        significantly reducing the number of expensive LLM calls.
        
        Args:
            chunks: List of KnowledgeChunk objects to check for conflicts
            
        Returns:
            List of tuples (chunk1, chunk2, has_conflict) where has_conflict is boolean
        """
        if not chunks or len(chunks) < 2:
            return []
        
        if not self.enable_clustering or not self._clusterer:
            # Fall back to checking all pairs if clustering is disabled
            return await self._detect_conflicts_batch_pairwise(chunks)
        
        try:
            # Get embeddings for all chunks
            gateway = self._get_gateway()
            chunk_embeddings = []
            chunk_ids = []
            
            for chunk in chunks:
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, lambda c=chunk: gateway._get_embedding_for_chunk(c)
                )
                if embedding is not None:
                    chunk_embeddings.append(embedding)
                    chunk_ids.append(chunk.id)
            
            if len(chunk_embeddings) < self.clustering_config.min_cluster_size:
                # Not enough data for clustering, fall back to pairwise
                logging.info(f"Not enough chunks for clustering ({len(chunk_embeddings)}), "
                           f"falling back to pairwise conflict detection")
                return await self._detect_conflicts_batch_pairwise(chunks)
            
            # Create a temporary clusterer for this batch
            from .clustering import KnowledgeClusterer
            temp_clusterer = KnowledgeClusterer(self.clustering_config)
            
            # Fit the clusterer on the batch
            embeddings_array = np.array(chunk_embeddings)
            temp_clusterer.fit(embeddings_array, chunk_ids)
            
            # Group chunks by cluster
            cluster_groups = {}
            chunk_lookup = {chunk.id: chunk for chunk in chunks}
            
            for chunk_id in chunk_ids:
                if chunk_id in chunk_lookup:
                    cluster_id = temp_clusterer.get_cluster_for_chunk(chunk_id)
                    if cluster_id not in cluster_groups:
                        cluster_groups[cluster_id] = []
                    cluster_groups[cluster_id].append(chunk_lookup[chunk_id])
            
            # Perform conflict detection within each cluster
            conflicts = []
            total_comparisons = 0
            optimized_comparisons = 0
            
            for cluster_id, cluster_chunks in cluster_groups.items():
                # Check all pairs within this cluster
                for i, chunk1 in enumerate(cluster_chunks):
                    for chunk2 in cluster_chunks[i+1:]:
                        optimized_comparisons += 1
                        has_conflict = await self._llm_detect_conflict(chunk1, chunk2)
                        conflicts.append((chunk1, chunk2, has_conflict))
                
                # Also check against nearby clusters (for boundary cases)
                if cluster_id != -1:  # Skip noise cluster for nearby checks
                    nearby_clusters = temp_clusterer.get_nearby_clusters(cluster_id)
                    for nearby_cluster_id in nearby_clusters[:2]:  # Limit to 2 nearby clusters
                        if nearby_cluster_id in cluster_groups:
                            nearby_chunks = cluster_groups[nearby_cluster_id]
                            for chunk1 in cluster_chunks:
                                for chunk2 in nearby_chunks:
                                    optimized_comparisons += 1
                                    has_conflict = await self._llm_detect_conflict(chunk1, chunk2)
                                    conflicts.append((chunk1, chunk2, has_conflict))
            
            # Calculate statistics
            total_comparisons = len(chunks) * (len(chunks) - 1) // 2
            reduction = (1 - optimized_comparisons / total_comparisons) * 100 if total_comparisons > 0 else 0
            
            logging.info(f"Batch clustering optimization: {optimized_comparisons}/{total_comparisons} "
                       f"conflict checks ({reduction:.1f}% reduction)")
            
            return conflicts
            
        except Exception as e:
            logging.warning(f"Batch clustering failed: {e}. Falling back to pairwise detection.")
            return await self._detect_conflicts_batch_pairwise(chunks)
    
    async def _detect_conflicts_batch_pairwise(self, chunks: List[KnowledgeChunk]) -> List[tuple]:
        """
        Fallback method for pairwise conflict detection without clustering.
        
        Args:
            chunks: List of KnowledgeChunk objects to check for conflicts
            
        Returns:
            List of tuples (chunk1, chunk2, has_conflict) where has_conflict is boolean
        """
        conflicts = []
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                has_conflict = await self._llm_detect_conflict(chunk1, chunk2)
                conflicts.append((chunk1, chunk2, has_conflict))
        
        return conflicts

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
