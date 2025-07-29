"""
Common classes and utilities for workbench scenarios.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class ConflictExpectation(BaseModel):
    """Expected conflict details."""
    description_contains: str
    conflicting_guidance_count: int
    should_detect: bool = True


class ScenarioDefinition(BaseModel):
    """Definition of a test scenario."""
    name: str
    description: str
    input_files: Dict[str, str]  # filename -> content
    expected_conflicts: List[ConflictExpectation]
    expected_resolution: str
    validation_checks: List[str]  # List of strings that should/shouldn't be in final output
    expected_chunk_counts: Optional[Dict[str, int]] = None  # filename -> expected chunk count
