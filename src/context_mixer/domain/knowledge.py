"""
Domain models for knowledge management.

This module contains the core domain objects for representing knowledge chunks,
search queries, and search results in the Context Mixer application.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AuthorityLevel(str, Enum):
    """Authority levels for knowledge chunks following CRAFT principles."""
    FOUNDATIONAL = "foundational"  # Company mission, legal requirements
    OFFICIAL = "official"          # Approved standards, architectural decisions
    CONVENTIONAL = "conventional"  # Team conventions, best practices
    EXPERIMENTAL = "experimental"  # Trials, proof-of-concepts
    DEPRECATED = "deprecated"      # Outdated but maintained


class GranularityLevel(str, Enum):
    """Granularity levels for knowledge detail."""
    SUMMARY = "summary"           # 50-100 tokens
    OVERVIEW = "overview"         # 200-300 tokens
    DETAILED = "detailed"         # 500-800 tokens
    COMPREHENSIVE = "comprehensive"  # 1000+ tokens


class TemporalScope(str, Enum):
    """Temporal validity of knowledge."""
    CURRENT = "current"
    DEPRECATED = "deprecated"
    FUTURE = "future"


class ProvenanceInfo(BaseModel):
    """Information about the source and history of knowledge."""
    source: str = Field(..., description="Original source file path")
    project_id: Optional[str] = Field(None, description="Project identifier")
    project_name: Optional[str] = Field(None, description="Human-readable project name")
    project_path: Optional[str] = Field(None, description="Root path of the source project")
    created_at: str = Field(..., description="When this knowledge was created")
    updated_at: Optional[str] = Field(None, description="When this knowledge was last updated")
    author: Optional[str] = Field(None, description="Who created or last modified this knowledge")


class ChunkMetadata(BaseModel):
    """Metadata for knowledge chunks following CRAFT principles."""
    domains: List[str] = Field(..., description="Knowledge domains (technical, business, design)")
    authority: AuthorityLevel = Field(..., description="Authority level of this knowledge")
    scope: List[str] = Field(..., description="Applicable scopes (enterprise, prototype, etc.)")
    granularity: GranularityLevel = Field(..., description="Detail level")
    temporal: TemporalScope = Field(..., description="When this knowledge is valid")
    dependencies: List[str] = Field(default_factory=list, description="Required prerequisite chunks")
    conflicts: List[str] = Field(default_factory=list, description="Chunks this conflicts with")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    provenance: ProvenanceInfo = Field(..., description="Source and history tracking")


class KnowledgeChunk(BaseModel):
    """
    A knowledge chunk representing a domain-coherent unit of information.

    This follows the CRAFT principle of chunking knowledge into atomic,
    semantically bounded units that prevent knowledge interference.
    """
    id: str = Field(..., description="Unique identifier for this chunk")
    content: str = Field(..., description="The actual knowledge content")
    concept: str = Field(default="", description="The main concept or topic this chunk covers")
    metadata: ChunkMetadata = Field(..., description="Metadata following CRAFT principles")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic search")

    def get_domains(self) -> List[str]:
        """Get the domains this knowledge chunk belongs to."""
        return self.metadata.domains

    def get_authority_level(self) -> AuthorityLevel:
        """Get the authority level of this knowledge."""
        return self.metadata.authority

    def is_current(self) -> bool:
        """Check if this knowledge is currently valid."""
        return self.metadata.temporal == TemporalScope.CURRENT

    def get_project_id(self) -> Optional[str]:
        """Get the project identifier this knowledge belongs to."""
        return self.metadata.provenance.project_id

    def get_project_name(self) -> Optional[str]:
        """Get the human-readable project name this knowledge belongs to."""
        return self.metadata.provenance.project_name

    def belongs_to_project(self, project_id: str) -> bool:
        """Check if this knowledge belongs to a specific project."""
        return self.metadata.provenance.project_id == project_id

    def belongs_to_any_project(self, project_ids: List[str]) -> bool:
        """Check if this knowledge belongs to any of the specified projects."""
        chunk_project_id = self.metadata.provenance.project_id
        return chunk_project_id is not None and chunk_project_id in project_ids


class SearchQuery(BaseModel):
    """A search query for knowledge retrieval."""
    text: str = Field(..., description="The search query text")
    domains: Optional[List[str]] = Field(None, description="Filter by specific domains")
    authority_levels: Optional[List[AuthorityLevel]] = Field(None, description="Filter by authority levels")
    scopes: Optional[List[str]] = Field(None, description="Filter by applicable scopes")
    granularity: Optional[GranularityLevel] = Field(None, description="Preferred granularity level")
    project_ids: Optional[List[str]] = Field(None, description="Filter by specific projects")
    exclude_projects: Optional[List[str]] = Field(None, description="Exclude specific projects")
    max_results: int = Field(10, description="Maximum number of results to return")
    min_relevance_score: float = Field(0.0, description="Minimum relevance score threshold")


class SearchResult(BaseModel):
    """A search result containing a knowledge chunk and relevance information."""
    chunk: KnowledgeChunk = Field(..., description="The knowledge chunk")
    relevance_score: float = Field(..., description="Relevance score for this result")
    match_explanation: Optional[str] = Field(None, description="Explanation of why this chunk matched")


class SearchResults(BaseModel):
    """Collection of search results."""
    query: SearchQuery = Field(..., description="The original search query")
    results: List[SearchResult] = Field(..., description="List of search results")
    total_found: int = Field(..., description="Total number of matching chunks found")

    def get_chunks(self) -> List[KnowledgeChunk]:
        """Extract just the knowledge chunks from the results."""
        return [result.chunk for result in self.results]

    def get_top_result(self) -> Optional[SearchResult]:
        """Get the highest-scoring result."""
        return self.results[0] if self.results else None
