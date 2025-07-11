"""
False Positive Naming Convention Scenario

This scenario tests that different naming conventions for different contexts
should NOT be detected as conflicts:
- camelCase for variables
- PascalCase for classes  
- snake_case for functions

These are complementary rules, not conflicting ones.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


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
    """Get the false positive naming convention scenario definition."""

    naming1_content = """# Naming Conventions

## Variables
- Use camelCase for variable names

## Classes  
- Use PascalCase for class names
"""

    naming2_content = """# Code Style Guide

## Functions
- Use snake_case for function names

## Variables
- Use camelCase for variable names
"""

    return ScenarioDefinition(
        name="false_positive_naming",
        description="Tests that different naming conventions for different contexts are NOT detected as conflicts",
        input_files={
            "naming1.md": naming1_content,
            "naming2.md": naming2_content
        },
        expected_conflicts=[
            # This scenario should NOT detect any conflicts
        ],
        expected_resolution="",  # No resolution needed since no conflicts expected
        validation_checks=[
            "should_contain:camelCase",
            "should_contain:PascalCase", 
            "should_contain:snake_case",
            # All naming conventions should be preserved since they're not conflicting
        ]
    )
