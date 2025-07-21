# Context Mixer Conflict Detection Workbench

## Overview

The Context Mixer Workbench is a comprehensive testing environment for conflict detection and resolution scenarios in the Context Mixer system. It provides automated integration testing that captures real-world problematic scenarios and ensures the conflict detection system works correctly.

## Purpose and Design Philosophy

The workbench serves as a **pure consumer** of the Context Mixer framework, maintaining complete separation from the main system to ensure flexibility for expansion and extension. Key design principles:

- **Integration Testing Focus**: Tests the complete conflict detection and resolution pipeline end-to-end
- **Regression Prevention**: Ensures previously fixed issues don't reoccur
- **Real-World Scenario Accumulation**: Collects actual problematic scenarios encountered during development and production use
- **Automated Execution**: Runs without user input for CI/CD integration
- **Easy Scenario Creation**: Supports rapid addition of new scenarios from real conflict data

## Prerequisites

Before using the workbench, ensure you have:

1. **Context Mixer installed**: The workbench depends on the main Context Mixer package
2. **OpenAI API access**: An OpenAI API key is required for LLM operations
3. **Python 3.8+**: The workbench requires modern Python with async support

## Configuration

### Required Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Optional Configuration

Configure the AI model to use (default is o4-mini):

```bash
export WORKBENCH_MODEL="O4_MINI"  # Available: O4_MINI, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO
```

## CLI Usage

The workbench provides a comprehensive command-line interface with multiple subcommands:

### Getting Help

```bash
# Show all available commands
python workbench/workbench_cli.py --help

# Get help for specific command
python workbench/workbench_cli.py run --help
```

### Running Scenarios

```bash
# Run all scenarios (recommended for regression testing)
python workbench/workbench_cli.py run

# Run a specific scenario
python workbench/workbench_cli.py run --scenario indentation_conflict

# Run scenario with different model
WORKBENCH_MODEL=GPT_4_1 python workbench/workbench_cli.py run --scenario internal_conflict
```

### Managing Scenarios

```bash
# List all available scenarios with descriptions
python workbench/workbench_cli.py list-scenarios

# Validate a scenario definition without running it
python workbench/workbench_cli.py validate-scenario indentation_conflict

# Add a new scenario from YAML definition
python workbench/workbench_cli.py add-scenario --from-yaml my_scenario.yaml
```

## Understanding Scenario Output

When running scenarios, you'll see detailed output including:

- **Execution timing**: How long each scenario takes to run
- **Conflict detection results**: What conflicts were found
- **Validation outcomes**: Whether the scenario passed or failed
- **Error details**: Specific issues if scenarios fail

### Example Output

```
🚀 Starting Conflict Detection Workbench
Using OpenAI model: OpenAIModels.O4_MINI

Running scenario: indentation_conflict
Description: Tests detection and resolution of conflicting indentation guidance
⏱️  Scenario completed in 3.45 seconds
✅ indentation_conflict PASSED

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Scenario               ┃ Status   ┃ Validations ┃ Time (s) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
┃ indentation_conflict   ┃ PASSED   ┃ 3/3         ┃ 3.45     ┃
└────────────────────────┴──────────┴─────────────┴──────────┘
```

## Creating New Scenarios

### Quick Start: From Real Conflicts

When you encounter a conflict detection issue during development or production use:

1. **Copy the conflicting content** to a YAML file
2. **Define expected behavior** 
3. **Generate the scenario** using the CLI
4. **Run and validate** the scenario

### Method 1: YAML Definition (Recommended)

Create a YAML file describing your scenario:

```yaml
# example_scenario.yaml
name: "my_conflict_scenario"
description: "Tests detection of indentation conflicts in coding standards"

input_files:
  "standards_v1.md": |
    # Coding Standards v1
    
    ## Formatting
    - Use 4 spaces for indentation
    - Maximum line length is 100 characters
    - Use semicolons in JavaScript
    
  "standards_v2.md": |
    # Updated Standards
    
    ## Formatting Rules
    - Use 2 spaces for indentation  
    - Maximum line length is 80 characters
    - Use semicolons in JavaScript

expected_conflicts:
  - description_contains: "indentation"
    conflicting_guidance_count: 2
    should_detect: true
  - description_contains: "line length"
    conflicting_guidance_count: 2
    should_detect: true

expected_resolution: "Use 4 spaces for indentation"

validation_checks:
  - "should_contain:4 spaces for indentation"
  - "should_not_contain:2 spaces for indentation"
  - "should_contain:Maximum line length is 100 characters"
  - "should_contain:semicolons in JavaScript"
```

Then generate the Python scenario:

```bash
python workbench/workbench_cli.py add-scenario --from-yaml example_scenario.yaml
```

### Method 2: Manual Python Files

For complex scenarios requiring custom logic, create a Python file directly:

```python
# workbench/scenarios/my_custom_scenario.py
"""
Custom scenario for testing specific conflict detection logic.
"""

from .common import ConflictExpectation, ScenarioDefinition


def get_scenario() -> ScenarioDefinition:
    """Get the custom scenario definition."""
    
    return ScenarioDefinition(
        name="my_custom_scenario",
        description="Tests custom conflict detection behavior",
        input_files={
            "file1.md": "Content with rules...",
            "file2.md": "Conflicting content..."
        },
        expected_conflicts=[
            ConflictExpectation(
                description_contains="specific term",
                conflicting_guidance_count=2,
                should_detect=True
            )
        ],
        expected_resolution="Expected resolution text",
        validation_checks=[
            "should_contain:expected text",
            "should_not_contain:unwanted text"
        ]
    )
```

The workbench will automatically discover this scenario on the next run.

## Scenario Format Reference

### Core Components

Every scenario must define these components:

#### 1. Metadata
- **name**: Unique identifier (used in CLI commands)
- **description**: Human-readable explanation of what the scenario tests

#### 2. Input Files
Dictionary of filename to content mappings:
```yaml
input_files:
  "filename.md": |
    Multi-line content
    goes here
```

#### 3. Expected Conflicts
List of conflicts that should be detected:
```yaml
expected_conflicts:
  - description_contains: "keyword that should appear in conflict description"
    conflicting_guidance_count: 2  # Number of conflicting guidance items
    should_detect: true  # Whether this conflict should be detected
```

#### 4. Expected Resolution
The expected outcome after conflict resolution:
```yaml
expected_resolution: "The rule that should win after resolution"
```

#### 5. Validation Checks
List of string-based validations for the final output:
```yaml
validation_checks:
  - "should_contain:text that must be in final output"
  - "should_not_contain:text that must not be in final output"
```

### Advanced YAML Features

#### Multi-line Content
Use YAML's `|` operator for multi-line file content:
```yaml
input_files:
  "complex_file.md": |
    # Multi-line content
    
    This preserves formatting
    and line breaks exactly
    as written.
```

#### Complex Scenarios
For scenarios testing edge cases:
```yaml
# Testing false positives (scenarios that should NOT detect conflicts)
expected_conflicts: []  # Empty list means no conflicts expected

# Testing internal conflicts (within single files)
input_files:
  "self_contradicting.md": |
    - Use 4 spaces for indentation
    - Use 2 spaces for indentation
```

## Built-in Scenarios

The workbench includes several important test scenarios:

### Core Conflict Detection
- **indentation_conflict**: Tests basic conflicting indentation rules
- **internal_conflict**: Tests conflicts within a single file

### False Positive Prevention  
- **false_positive_naming**: Ensures different naming conventions for different contexts aren't flagged as conflicts
- **architectural_scope_false_positive**: Verifies architectural pattern-specific rules don't conflict with general guidelines

### Adding Your Own
Use these as templates when creating new scenarios for specific issues you encounter.

## Integration with Development Workflow

### During Development
```bash
# Quick validation after making changes
python workbench/workbench_cli.py run

# Test specific area of concern
python workbench/workbench_cli.py run --scenario false_positive_naming
```

### In CI/CD Pipelines
```bash
# Add to your CI pipeline
python workbench/workbench_cli.py run
if [ $? -ne 0 ]; then
    echo "Workbench scenarios failed - conflict detection regression detected"
    exit 1
fi
```

### Capturing Production Issues
When users report conflict detection problems:

1. Create a YAML scenario file with the problematic content
2. Add it to the workbench
3. Verify it fails (reproduces the bug)
4. Fix the underlying issue
5. Verify the scenario now passes
6. The scenario prevents regression

## Architecture and Extensibility

### Pure Consumer Design
The workbench operates as a pure consumer of the Context Mixer system:
- Uses public APIs only
- No tight coupling to internal implementations  
- Can be extended without modifying core system
- Validates the system from a user perspective

### Auto-Discovery System
- Scenarios are automatically discovered from the `scenarios/` directory
- No manual registration required
- New scenarios become available immediately
- Dynamic CLI choices based on available scenarios

### Extensibility Points
- **Custom Validators**: Extend validation beyond string matching
- **Custom Resolvers**: Test different conflict resolution strategies
- **Performance Metrics**: Add timing and resource usage validation
- **External Integrations**: Connect to monitoring systems

## Troubleshooting

### Common Issues

#### API Key Problems
```
Error: OPENAI_API_KEY environment variable not set
```
**Solution**: Export your OpenAI API key as shown in Configuration section.

#### Model Configuration Issues  
```
Error: Unknown model 'INVALID_MODEL'
```
**Solution**: Use one of the supported models: O4_MINI, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO

#### Scenario Discovery Problems
```
Warning: Could not load scenario my_scenario
```
**Solution**: Ensure your scenario file:
- Is in the `workbench/scenarios/` directory
- Has a `.py` extension
- Contains a `get_scenario()` function
- Uses proper imports from `.common`

#### Validation Failures
When scenarios fail, check:
- Expected conflicts match actual conflict detection behavior
- Validation checks use correct string matching patterns
- Input files contain the expected conflicting content

### Debug Mode
For detailed debugging, check the scenario validation:
```bash
python workbench/workbench_cli.py validate-scenario problematic_scenario
```

### Getting Support
If you encounter issues:
1. Check this README for common solutions
2. Validate your scenario definitions
3. Test with built-in scenarios first
4. Review the scenario output for specific error messages

## Performance Considerations

### Execution Time
- Each scenario creates a temporary library and processes files
- LLM calls can take several seconds per scenario
- Use specific scenario execution for faster iteration during development

### Resource Usage
- Temporary directories are created and cleaned up automatically
- Memory usage scales with scenario complexity and file sizes
- Consider model selection for cost vs. accuracy trade-offs

## Contributing New Scenarios

When contributing scenarios to the project:

1. **Use descriptive names** that clearly indicate what is being tested
2. **Include comprehensive descriptions** explaining the real-world issue
3. **Add validation checks** that verify the expected behavior
4. **Test both positive and negative cases** where appropriate
5. **Document any special requirements** in the scenario file

The workbench is designed to grow over time, accumulating a comprehensive test suite that prevents regression and ensures the Context Mixer conflict detection system remains robust and reliable.