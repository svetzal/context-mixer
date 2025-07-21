"""
Architectural Scope False Positive Scenario

This scenario tests that architectural pattern-specific rules should NOT be 
detected as conflicts with general coding guidelines:
- Gateway-specific rule: "Gateways should not be tested"
- General testing rule: "Write tests for any new functionality"

These rules apply to different architectural scopes and should be complementary,
not conflicting. This scenario validates the fix for the false positive
conflict detection issue.
"""

from .common import ConflictExpectation, ScenarioDefinition


def get_scenario() -> ScenarioDefinition:
    """Get the architectural scope false positive scenario definition."""

    gateway_content = """# Gateway Pattern Guidelines

Use the Gateway pattern to isolate I/O (network, disk, etc.) from business logic.

- Gateways should contain minimal to no logic, simply delegate to the OS or libraries that perform I/O.
- Gateways should not be tested, to avoid mocking systems you don't own.
"""

    general_testing_content = """# General Development Guidelines

- Follow the existing project structure.
- Write tests for any new functionality and co-locate them.
- Document using numpy-style docstrings.
- Keep code complexity low (max complexity 10).
- Use type hints for all functions and methods.
- Use pydantic v2 (not dataclasses) for data objects.
- Favor list/dict comprehensions over explicit loops.
- Favor declarative styles over imperative.
"""

    return ScenarioDefinition(
        name="architectural_scope_false_positive",
        description="Tests that architectural pattern-specific rules are NOT detected as conflicts with general guidelines",
        input_files={
            "gateway_guidelines.md": gateway_content,
            "general_guidelines.md": general_testing_content
        },
        expected_conflicts=[
            # This scenario should NOT detect any conflicts
            # The gateway-specific "don't test" rule and general "write tests" rule
            # apply to different architectural scopes and should coexist
        ],
        expected_resolution="",  # No resolution needed since no conflicts expected
        validation_checks=[
            "should_contain:Gateway pattern",
            "should_contain:Gateways should not be tested",
            "should_contain:Write tests for any new functionality",
            "should_contain:Use type hints",
            "should_contain:pydantic v2",
            # All content should be preserved since they're not conflicting
        ]
    )