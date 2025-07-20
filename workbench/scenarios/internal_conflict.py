"""
Internal Conflict Scenario

This scenario tests detection of conflicts within a single file:
- A file that specifies both 80 and 100 character line lengths

This should be detected as an internal conflict.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConflictExpectation:
    """Expected conflict details."""
    description_contains: str
    conflicting_guidance_count: int
    should_detect: bool = True


@dataclass
class ScenarioDefinition:
    """Definition of a test scenario."""
    name: str
    description: str
    input_files: Dict[str, str]  # filename -> content
    expected_conflicts: List[ConflictExpectation]
    expected_resolution: str
    validation_checks: List[str]  # List of strings that should/shouldn't be in final output


def get_scenario() -> ScenarioDefinition:
    """Get the internal conflict scenario definition."""

    internal_conflict_content = """# Coding Standards

## Style Guide

- Use 2 spaces for indentation
- Maximum line length is 80 characters
- Use camelCase for variable names
- Use PascalCase for class names
- Use snake_case for function names
- Maximum line length is 100 characters
"""

    return ScenarioDefinition(
        name="internal_conflict",
        description="Tests detection of conflicts within a single file",
        input_files={
            "code_internal_conflict.md": internal_conflict_content
        },
        expected_conflicts=[
            ConflictExpectation(
                description_contains="line length",
                conflicting_guidance_count=2,
                should_detect=True
            )
        ],
        expected_resolution="Maximum line length is 80 characters",  # Default resolution
        validation_checks=[
            "should_contain:80 characters",
            "should_not_contain:100 characters",
            "should_contain:2 spaces for indentation",  # Non-conflicting content should remain
            "should_contain:camelCase",  # Variable naming convention should remain
        ]
    )
