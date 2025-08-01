"""
Indentation Conflict Scenario

This scenario tests the basic indentation conflict that we've encountered:
- code1.md specifies 4 spaces for indentation
- code2.md specifies 2 spaces for indentation

This should be detected as a conflict and resolved automatically.
"""

from .common import ConflictExpectation, ScenarioDefinition


def get_scenario() -> ScenarioDefinition:
    """Get the indentation conflict scenario definition."""

    code1_content = """# Project Guidelines

## Code Style

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Use camelCase for variable names
- Use PascalCase for class names
- Use snake_case for function names
"""

    code2_content = """# Coding Standards

## Style Guide

- Use 2 spaces for indentation
- Maximum line length is 80 characters
- Use camelCase for variable names
- Use PascalCase for class names
- Use snake_case for function names
"""

    return ScenarioDefinition(
        name="indentation_conflict",
        description="Tests detection and resolution of conflicting indentation guidance",
        input_files={
            "code1.md": code1_content,
            "code2.md": code2_content
        },
        expected_conflicts=[
            ConflictExpectation(
                description_contains="indentation",
                conflicting_guidance_count=2,
                should_detect=True
            ),
            ConflictExpectation(
                description_contains="line length",
                conflicting_guidance_count=2,
                should_detect=True
            )
        ],
        expected_resolution="Use 4 spaces for indentation",  # Default resolution
        validation_checks=[
            "should_contain:4 spaces for indentation",
            "should_not_contain:2 spaces for indentation",
            "should_contain:camelCase for variable names",  # Non-conflicting content should remain
        ],
        expected_chunk_counts={
            "code1.md": 5,  # Expect 5 chunks of knowledge from code1.md
            "code2.md": 5   # Expect 5 chunks of knowledge from code2.md
        }
    )
