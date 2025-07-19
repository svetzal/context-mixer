from typing import Dict, Optional
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

from .progress import ProgressObserver, ProgressUpdate


class CLIProgressObserver(ProgressObserver):
    """CLI progress observer that displays progress bars using Rich."""
    
    def __init__(self, console: Console):
        self.console = console
        self.progress: Optional[Progress] = None
        self.tasks: Dict[str, TaskID] = {}
        self._active_operations = 0
    
    def _ensure_progress_started(self) -> None:
        """Ensure the progress display is started."""
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False
            )
            self.progress.start()
    
    def _cleanup_progress_if_done(self) -> None:
        """Clean up progress display if no operations are active."""
        if self._active_operations == 0 and self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()
    
    def on_operation_start(self, operation_id: str, operation_name: str, total: int) -> None:
        """Called when an operation starts."""
        self._ensure_progress_started()
        self._active_operations += 1
        
        task_id = self.progress.add_task(
            description=operation_name,
            total=total
        )
        self.tasks[operation_id] = task_id
    
    def on_progress_update(self, update: ProgressUpdate) -> None:
        """Called when progress is updated."""
        if self.progress is None or update.operation_id not in self.tasks:
            return
        
        task_id = self.tasks[update.operation_id]
        description = update.operation_name
        if update.message:
            description = f"{update.operation_name}: {update.message}"
        
        self.progress.update(
            task_id,
            completed=update.current,
            description=description
        )
    
    def on_operation_complete(self, operation_id: str, operation_name: str) -> None:
        """Called when an operation completes."""
        if self.progress is None or operation_id not in self.tasks:
            return
        
        task_id = self.tasks[operation_id]
        self.progress.update(task_id, description=f"{operation_name}: Complete")
        
        # Remove the task after a brief moment to show completion
        self.progress.remove_task(task_id)
        del self.tasks[operation_id]
        self._active_operations -= 1
        
        self._cleanup_progress_if_done()
    
    def on_operation_failed(self, operation_id: str, operation_name: str, error: str) -> None:
        """Called when an operation fails."""
        if self.progress is None or operation_id not in self.tasks:
            return
        
        task_id = self.tasks[operation_id]
        self.progress.update(task_id, description=f"{operation_name}: Failed - {error}")
        
        # Remove the task
        self.progress.remove_task(task_id)
        del self.tasks[operation_id]
        self._active_operations -= 1
        
        self._cleanup_progress_if_done()