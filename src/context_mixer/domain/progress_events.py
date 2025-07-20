from dataclasses import dataclass
from typing import Optional, Dict, Any
from .events import Event


@dataclass
class ProgressStartedEvent(Event):
    """
    Event published when a progress operation starts.
    """
    operation_id: str
    operation_name: str
    total: int
    project_id: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "progress_started"
        super().__post_init__()


@dataclass
class ProgressUpdatedEvent(Event):
    """
    Event published when progress is updated for an operation.
    """
    operation_id: str
    operation_name: str
    current: int
    total: int
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "progress_updated"
        super().__post_init__()
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)


@dataclass
class ProgressCompletedEvent(Event):
    """
    Event published when a progress operation completes successfully.
    """
    operation_id: str
    operation_name: str
    project_id: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "progress_completed"
        super().__post_init__()


@dataclass
class ProgressFailedEvent(Event):
    """
    Event published when a progress operation fails.
    """
    operation_id: str
    operation_name: str
    error: str
    project_id: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = "progress_failed"
        super().__post_init__()