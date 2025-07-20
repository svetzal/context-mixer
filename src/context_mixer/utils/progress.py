from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProgressStatus(Enum):
    """Status of a progress operation."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Represents a progress update."""
    operation_id: str
    operation_name: str
    current: int
    total: int
    status: ProgressStatus
    message: Optional[str] = None
    metadata: Optional[dict] = None

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)


class ProgressObserver(ABC):
    """Abstract base class for progress observers."""
    
    @abstractmethod
    def on_progress_update(self, update: ProgressUpdate) -> None:
        """Called when progress is updated."""
        pass
    
    @abstractmethod
    def on_operation_start(self, operation_id: str, operation_name: str, total: int) -> None:
        """Called when an operation starts."""
        pass
    
    @abstractmethod
    def on_operation_complete(self, operation_id: str, operation_name: str) -> None:
        """Called when an operation completes."""
        pass
    
    @abstractmethod
    def on_operation_failed(self, operation_id: str, operation_name: str, error: str) -> None:
        """Called when an operation fails."""
        pass


class NoOpProgressObserver(ProgressObserver):
    """A no-op progress observer that does nothing."""
    
    def on_progress_update(self, update: ProgressUpdate) -> None:
        pass
    
    def on_operation_start(self, operation_id: str, operation_name: str, total: int) -> None:
        pass
    
    def on_operation_complete(self, operation_id: str, operation_name: str) -> None:
        pass
    
    def on_operation_failed(self, operation_id: str, operation_name: str, error: str) -> None:
        pass


class ProgressTracker:
    """Tracks progress for long-running operations."""
    
    def __init__(self, observer: Optional[ProgressObserver] = None):
        self.observer = observer or NoOpProgressObserver()
        self._operations = {}
    
    def start_operation(self, operation_id: str, operation_name: str, total: int) -> None:
        """Start tracking a new operation."""
        self._operations[operation_id] = {
            'name': operation_name,
            'current': 0,
            'total': total,
            'status': ProgressStatus.STARTED
        }
        self.observer.on_operation_start(operation_id, operation_name, total)
    
    def update_progress(self, operation_id: str, current: int, message: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        """Update progress for an operation."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['current'] = current
        op['status'] = ProgressStatus.IN_PROGRESS
        
        update = ProgressUpdate(
            operation_id=operation_id,
            operation_name=op['name'],
            current=current,
            total=op['total'],
            status=op['status'],
            message=message,
            metadata=metadata
        )
        self.observer.on_progress_update(update)
    
    def complete_operation(self, operation_id: str) -> None:
        """Mark an operation as completed."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['status'] = ProgressStatus.COMPLETED
        op['current'] = op['total']
        
        self.observer.on_operation_complete(operation_id, op['name'])
        del self._operations[operation_id]
    
    def fail_operation(self, operation_id: str, error: str) -> None:
        """Mark an operation as failed."""
        if operation_id not in self._operations:
            return
        
        op = self._operations[operation_id]
        op['status'] = ProgressStatus.FAILED
        
        self.observer.on_operation_failed(operation_id, op['name'], error)
        del self._operations[operation_id]