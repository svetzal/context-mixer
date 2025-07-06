"""
Knowledge Quarantine System for Context Mixer.

This module implements the quarantine system for isolating conflicting knowledge chunks
that cannot be automatically resolved. It provides mechanisms to quarantine, review,
and resolve conflicts in knowledge management following CRAFT principles.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

from .knowledge import KnowledgeChunk, AuthorityLevel
from .conflict import Conflict


class QuarantineReason(str, Enum):
    """Reasons why a knowledge chunk was quarantined."""
    SEMANTIC_CONFLICT = "semantic_conflict"      # Conflicts with existing knowledge semantically
    AUTHORITY_CONFLICT = "authority_conflict"    # Conflicts with higher authority knowledge
    TEMPORAL_CONFLICT = "temporal_conflict"      # Conflicts with current/deprecated status
    DEPENDENCY_VIOLATION = "dependency_violation" # Missing required dependencies
    VALIDATION_FAILURE = "validation_failure"    # Failed chunk completeness validation
    MANUAL_QUARANTINE = "manual_quarantine"      # Manually quarantined by user


class ResolutionAction(str, Enum):
    """Actions that can be taken to resolve quarantined chunks."""
    ACCEPT = "accept"                # Accept the quarantined chunk, overriding conflicts
    REJECT = "reject"                # Reject the quarantined chunk permanently
    MERGE = "merge"                  # Merge with existing conflicting knowledge
    MODIFY = "modify"                # Modify the chunk to resolve conflicts
    DEFER = "defer"                  # Defer resolution to later time
    ESCALATE = "escalate"            # Escalate to higher authority for resolution


class Resolution(BaseModel):
    """Resolution information for a quarantined chunk."""
    action: ResolutionAction = Field(..., description="The resolution action taken")
    reason: str = Field(..., description="Explanation for the resolution")
    resolved_by: Optional[str] = Field(None, description="Who resolved the quarantine")
    resolved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), 
                            description="When the quarantine was resolved")
    modified_chunk: Optional[KnowledgeChunk] = Field(None, 
                                                   description="Modified chunk if action is MODIFY")
    notes: Optional[str] = Field(None, description="Additional notes about the resolution")


class QuarantinedChunk(BaseModel):
    """A knowledge chunk that has been quarantined due to conflicts or validation issues."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                   description="Unique identifier for the quarantined chunk")
    chunk: KnowledgeChunk = Field(..., description="The quarantined knowledge chunk")
    reason: QuarantineReason = Field(..., description="Why this chunk was quarantined")
    description: str = Field(..., description="Detailed description of the quarantine reason")
    conflicting_chunks: List[str] = Field(default_factory=list, 
                                        description="IDs of chunks this conflicts with")
    quarantined_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(),
                               description="When this chunk was quarantined")
    quarantined_by: Optional[str] = Field(None, description="Who quarantined this chunk")
    priority: int = Field(1, description="Priority for resolution (1=high, 5=low)")
    resolution: Optional[Resolution] = Field(None, description="Resolution if resolved")
    
    def is_resolved(self) -> bool:
        """Check if this quarantined chunk has been resolved."""
        return self.resolution is not None
    
    def get_age_days(self) -> int:
        """Get the age of this quarantine in days."""
        quarantined = datetime.fromisoformat(self.quarantined_at.replace('Z', '+00:00'))
        now = datetime.utcnow()
        return (now - quarantined).days


class KnowledgeQuarantine:
    """
    Knowledge Quarantine System for managing conflicting knowledge chunks.
    
    This system isolates knowledge chunks that cannot be automatically resolved
    due to conflicts, validation failures, or other issues. It provides mechanisms
    for reviewing and resolving quarantined chunks.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the knowledge quarantine system.
        
        Args:
            storage_path: Optional path for persistent storage of quarantined chunks
        """
        self._quarantined_chunks: Dict[str, QuarantinedChunk] = {}
        self._storage_path = storage_path
        
    def quarantine_chunk(self, 
                        chunk: KnowledgeChunk, 
                        reason: QuarantineReason,
                        description: str,
                        conflicting_chunks: Optional[List[str]] = None,
                        quarantined_by: Optional[str] = None,
                        priority: int = 1) -> str:
        """
        Quarantine a knowledge chunk due to conflicts or validation issues.
        
        Args:
            chunk: The knowledge chunk to quarantine
            reason: The reason for quarantine
            description: Detailed description of why the chunk was quarantined
            conflicting_chunks: List of chunk IDs that this chunk conflicts with
            quarantined_by: Who quarantined this chunk
            priority: Priority for resolution (1=high, 5=low)
            
        Returns:
            The ID of the quarantined chunk
        """
        quarantined_chunk = QuarantinedChunk(
            chunk=chunk,
            reason=reason,
            description=description,
            conflicting_chunks=conflicting_chunks or [],
            quarantined_by=quarantined_by,
            priority=priority
        )
        
        self._quarantined_chunks[quarantined_chunk.id] = quarantined_chunk
        return quarantined_chunk.id
    
    def review_quarantined_chunks(self, 
                                 reason_filter: Optional[QuarantineReason] = None,
                                 resolved_filter: Optional[bool] = None,
                                 priority_filter: Optional[int] = None,
                                 project_filter: Optional[str] = None) -> List[QuarantinedChunk]:
        """
        Review quarantined chunks with optional filtering.
        
        Args:
            reason_filter: Filter by quarantine reason
            resolved_filter: Filter by resolution status (True=resolved, False=unresolved, None=all)
            priority_filter: Filter by priority level
            project_filter: Filter by project ID
            
        Returns:
            List of quarantined chunks matching the filters
        """
        chunks = list(self._quarantined_chunks.values())
        
        if reason_filter is not None:
            chunks = [c for c in chunks if c.reason == reason_filter]
            
        if resolved_filter is not None:
            chunks = [c for c in chunks if c.is_resolved() == resolved_filter]
            
        if priority_filter is not None:
            chunks = [c for c in chunks if c.priority == priority_filter]
            
        if project_filter is not None:
            chunks = [c for c in chunks if c.chunk.get_project_id() == project_filter]
        
        # Sort by priority (high to low) then by quarantine date (oldest first)
        chunks.sort(key=lambda c: (c.priority, c.quarantined_at))
        
        return chunks
    
    def resolve_quarantine(self, 
                          chunk_id: str, 
                          resolution: Resolution) -> bool:
        """
        Resolve a quarantined chunk with the specified resolution.
        
        Args:
            chunk_id: ID of the quarantined chunk to resolve
            resolution: The resolution to apply
            
        Returns:
            True if the quarantine was successfully resolved, False if chunk not found
        """
        if chunk_id not in self._quarantined_chunks:
            return False
            
        quarantined_chunk = self._quarantined_chunks[chunk_id]
        quarantined_chunk.resolution = resolution
        
        return True
    
    def get_quarantined_chunk(self, chunk_id: str) -> Optional[QuarantinedChunk]:
        """
        Get a specific quarantined chunk by ID.
        
        Args:
            chunk_id: ID of the quarantined chunk
            
        Returns:
            The quarantined chunk if found, None otherwise
        """
        return self._quarantined_chunks.get(chunk_id)
    
    def remove_quarantined_chunk(self, chunk_id: str) -> bool:
        """
        Remove a quarantined chunk from the quarantine system.
        
        This should typically only be done after the chunk has been resolved
        and any necessary actions have been taken.
        
        Args:
            chunk_id: ID of the quarantined chunk to remove
            
        Returns:
            True if the chunk was removed, False if not found
        """
        if chunk_id in self._quarantined_chunks:
            del self._quarantined_chunks[chunk_id]
            return True
        return False
    
    def get_quarantine_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the quarantine system.
        
        Returns:
            Dictionary containing quarantine statistics
        """
        chunks = list(self._quarantined_chunks.values())
        resolved_chunks = [c for c in chunks if c.is_resolved()]
        unresolved_chunks = [c for c in chunks if not c.is_resolved()]
        
        # Count by reason
        reason_counts = {}
        for chunk in chunks:
            reason = chunk.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Count by priority
        priority_counts = {}
        for chunk in unresolved_chunks:
            priority = chunk.priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Age statistics for unresolved chunks
        ages = [c.get_age_days() for c in unresolved_chunks]
        avg_age = sum(ages) / len(ages) if ages else 0
        max_age = max(ages) if ages else 0
        
        return {
            "total_quarantined": len(chunks),
            "resolved": len(resolved_chunks),
            "unresolved": len(unresolved_chunks),
            "reason_breakdown": reason_counts,
            "priority_breakdown": priority_counts,
            "average_age_days": round(avg_age, 1),
            "oldest_quarantine_days": max_age
        }
    
    def get_high_priority_unresolved(self) -> List[QuarantinedChunk]:
        """
        Get high-priority unresolved quarantined chunks.
        
        Returns:
            List of unresolved chunks with priority 1 or 2, sorted by age
        """
        high_priority = [
            c for c in self._quarantined_chunks.values() 
            if not c.is_resolved() and c.priority <= 2
        ]
        
        # Sort by age (oldest first)
        high_priority.sort(key=lambda c: c.quarantined_at)
        
        return high_priority
    
    def clear_resolved_chunks(self) -> int:
        """
        Remove all resolved quarantined chunks from the system.
        
        Returns:
            Number of chunks that were removed
        """
        resolved_ids = [
            chunk_id for chunk_id, chunk in self._quarantined_chunks.items()
            if chunk.is_resolved()
        ]
        
        for chunk_id in resolved_ids:
            del self._quarantined_chunks[chunk_id]
            
        return len(resolved_ids)