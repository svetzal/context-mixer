"""
Timing utilities for measuring operation performance.

This module provides clean, testable timing functionality that can be used
throughout the application to measure key operation durations.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Result of a timing operation."""
    operation_name: str
    duration_seconds: float
    start_time: float
    end_time: float


@dataclass
class TimingCollector:
    """Collects timing results for multiple operations."""
    results: List[TimingResult] = field(default_factory=list)
    
    def add_result(self, result: TimingResult) -> None:
        """Add a timing result to the collection."""
        self.results.append(result)
    
    def get_total_duration(self) -> float:
        """Get the total duration of all operations."""
        return sum(result.duration_seconds for result in self.results)
    
    def get_operation_duration(self, operation_name: str) -> Optional[float]:
        """Get the duration of a specific operation."""
        for result in self.results:
            if result.operation_name == operation_name:
                return result.duration_seconds
        return None
    
    def get_summary(self) -> Dict[str, float]:
        """Get a summary of all operation durations."""
        return {result.operation_name: result.duration_seconds for result in self.results}


@contextmanager
def time_operation(operation_name: str, collector: Optional[TimingCollector] = None):
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        collector: Optional collector to store the timing result
        
    Yields:
        TimingResult: The timing result (available after the operation completes)
        
    Example:
        collector = TimingCollector()
        with time_operation("file_reading", collector) as timing:
            # Do some work
            pass
        print(f"Operation took {timing.duration_seconds:.2f} seconds")
    """
    start_time = time.time()
    result = TimingResult(
        operation_name=operation_name,
        duration_seconds=0.0,
        start_time=start_time,
        end_time=0.0
    )
    
    try:
        yield result
    finally:
        end_time = time.time()
        result.end_time = end_time
        result.duration_seconds = end_time - start_time
        
        if collector is not None:
            collector.add_result(result)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"