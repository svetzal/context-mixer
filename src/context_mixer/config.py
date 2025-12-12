"""
Configuration module for Context Mixer.

This module provides centralized configuration settings for the application.
"""

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_ROOT_CONTEXT_FILENAME = "context.md"

class Config:
    """Configuration settings for Context Mixer."""

    def __init__(self,
                 library_path: Path = None,
                 conflict_detection_batch_size: int = 5,
                 clustering_enabled: bool = True,
                 min_cluster_size: int = 3,
                 clustering_fallback: bool = True,
                 llm_provider: str = "openai",
                 llm_model: str = "o4-mini",
                 llm_api_key: Optional[str] = None,
                 conflict_embedding_similarity_threshold: float = 0.70,
                 conflict_pairs_per_llm_batch: int = 10,
                 conflict_detection_metrics_enabled: bool = True):
        """
        Initialize a Config object with library path, batch processing, clustering, and LLM settings.

        Args:
            library_path: Path to the context library. If None, uses the default path.
            conflict_detection_batch_size: Number of conflict detections to process concurrently (default: 5).
            clustering_enabled: Enable HDBSCAN clustering for conflict detection optimization (default: True).
            min_cluster_size: Minimum size for clusters in HDBSCAN (default: 3).
            clustering_fallback: Fall back to traditional O(n²) detection if clustering fails (default: True).
            llm_provider: LLM provider to use ("openai" or "ollama") (default: "openai").
            llm_model: Model name to use with the LLM provider (default: "o4-mini").
            llm_api_key: API key for providers that require it (default: None).
            conflict_embedding_similarity_threshold: Minimum cosine similarity for conflict detection (default: 0.70).
            conflict_pairs_per_llm_batch: Number of pairs to analyze per LLM call (default: 10).
            conflict_detection_metrics_enabled: Enable metrics collection for conflict detection (default: True).
        """
        self._library_path = library_path or Path.home() / ".context-mixer"
        self._conflict_detection_batch_size = conflict_detection_batch_size
        self._clustering_enabled = clustering_enabled
        self._min_cluster_size = min_cluster_size
        self._clustering_fallback = clustering_fallback
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._conflict_embedding_similarity_threshold = conflict_embedding_similarity_threshold
        self._conflict_pairs_per_llm_batch = conflict_pairs_per_llm_batch
        self._conflict_detection_metrics_enabled = conflict_detection_metrics_enabled

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

    @property
    def clustering_enabled(self) -> bool:
        """
        Get whether HDBSCAN clustering is enabled for conflict detection optimization.

        Returns:
            bool: True if clustering is enabled, False otherwise.
        """
        return self._clustering_enabled

    @property
    def min_cluster_size(self) -> int:
        """
        Get the minimum cluster size for HDBSCAN clustering.

        Returns:
            int: The minimum number of chunks required to form a cluster.
        """
        return self._min_cluster_size

    @property
    def clustering_fallback(self) -> bool:
        """
        Get whether to fall back to traditional conflict detection if clustering fails.

        Returns:
            bool: True if fallback is enabled, False otherwise.
        """
        return self._clustering_fallback

    @property
    def llm_provider(self) -> str:
        """
        Get the LLM provider.

        Returns:
            str: The LLM provider ("openai" or "ollama").
        """
        return self._llm_provider

    @property
    def llm_model(self) -> str:
        """
        Get the LLM model.

        Returns:
            str: The model name to use with the LLM provider.
        """
        return self._llm_model

    @property
    def llm_api_key(self) -> Optional[str]:
        """
        Get the LLM API key.

        Returns:
            Optional[str]: The API key for providers that require it.
        """
        return self._llm_api_key

    @property
    def conflict_embedding_similarity_threshold(self) -> float:
        """
        Get the embedding similarity threshold for conflict detection.

        Returns:
            float: Minimum cosine similarity to consider for conflicts.
        """
        return self._conflict_embedding_similarity_threshold

    @property
    def conflict_pairs_per_llm_batch(self) -> int:
        """
        Get the number of pairs to analyze per LLM call.

        Returns:
            int: Number of pairs per batch.
        """
        return self._conflict_pairs_per_llm_batch

    @property
    def conflict_detection_metrics_enabled(self) -> bool:
        """
        Get whether conflict detection metrics are enabled.

        Returns:
            bool: True if metrics collection is enabled.
        """
        return self._conflict_detection_metrics_enabled

    @property
    def config_path(self) -> Path:
        """
        Get the path to the config file.

        Returns:
            Path: The path to the config.json file.
        """
        return self.library_path / "config.json"

    def to_dict(self) -> dict:
        """
        Convert the config to a dictionary for JSON serialization.

        Returns:
            dict: Configuration as a dictionary.
        """
        return {
            "library_path": str(self._library_path),
            "conflict_detection_batch_size": self._conflict_detection_batch_size,
            "clustering_enabled": self._clustering_enabled,
            "min_cluster_size": self._min_cluster_size,
            "clustering_fallback": self._clustering_fallback,
            "llm_provider": self._llm_provider,
            "llm_model": self._llm_model,
            "llm_api_key": self._llm_api_key,
            "conflict_embedding_similarity_threshold": self._conflict_embedding_similarity_threshold,
            "conflict_pairs_per_llm_batch": self._conflict_pairs_per_llm_batch,
            "conflict_detection_metrics_enabled": self._conflict_detection_metrics_enabled
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """
        Create a Config object from a dictionary.

        Args:
            data: Configuration data as a dictionary.

        Returns:
            Config: A new Config object.
        """
        library_path = Path(data.get("library_path", Path.home() / ".context-mixer"))
        return cls(
            library_path=library_path,
            conflict_detection_batch_size=data.get("conflict_detection_batch_size", 5),
            clustering_enabled=data.get("clustering_enabled", True),
            min_cluster_size=data.get("min_cluster_size", 3),
            clustering_fallback=data.get("clustering_fallback", True),
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "o4-mini"),
            llm_api_key=data.get("llm_api_key"),
            conflict_embedding_similarity_threshold=data.get("conflict_embedding_similarity_threshold", 0.70),
            conflict_pairs_per_llm_batch=data.get("conflict_pairs_per_llm_batch", 10),
            conflict_detection_metrics_enabled=data.get("conflict_detection_metrics_enabled", True)
        )

    def save(self) -> None:
        """
        Save the configuration to the config file.
        """
        # Ensure the config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """
        Load configuration from a file, or create default if it doesn't exist.

        Args:
            config_path: Optional path to config file. If None, uses default location.

        Returns:
            Config: The loaded or default configuration.
        """
        if config_path is None:
            config_path = Path.home() / ".context-mixer" / "config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # If config file is corrupted, fall back to defaults
                pass

        # Return default config if file doesn't exist or is corrupted
        return cls()
