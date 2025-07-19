"""
Configuration module for Context Mixer.

This module provides centralized configuration settings for the application.
"""

from pathlib import Path

DEFAULT_ROOT_CONTEXT_FILENAME = "context.md"

class Config:
    """Configuration settings for Context Mixer."""

    def __init__(self, library_path: Path = None, conflict_detection_batch_size: int = 5):
        """
        Initialize a Config object with a library path and batch processing settings.

        Args:
            library_path: Path to the context library. If None, uses the default path.
            conflict_detection_batch_size: Number of conflict detections to process concurrently (default: 5).
        """
        self._library_path = library_path or Path.home() / ".context-mixer"
        self._conflict_detection_batch_size = conflict_detection_batch_size

    @property
    def library_path(self) -> Path:
        """
        Get the library path.

        Returns:
            Path: The library path.
        """
        return self._library_path

    @property
    def conflict_detection_batch_size(self) -> int:
        """
        Get the conflict detection batch size.

        Returns:
            int: The number of conflict detections to process concurrently.
        """
        return self._conflict_detection_batch_size
