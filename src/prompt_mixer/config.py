"""
Configuration module for Prompt Mixer.

This module provides centralized configuration settings for the application.
"""

from pathlib import Path

DEFAULT_ROOT_CONTEXT_FILENAME = "context.md"

class Config:
    """Configuration settings for Prompt Mixer."""

    def __init__(self, library_path: Path = None):
        """
        Initialize a Config object with a library path.

        Args:
            library_path: Path to the prompt library. If None, uses the default path.
        """
        self._library_path = library_path or Path.home() / ".prompt-mixer"

    @property
    def library_path(self) -> Path:
        """
        Get the library path.

        Returns:
            Path: The library path.
        """
        return self._library_path
