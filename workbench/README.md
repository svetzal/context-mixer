# Conflict Detection and Resolution Workbench

This workbench provides a persistent testing environment for conflict detection and resolution scenarios. It accumulates troublesome scenarios we encounter during development and ensures we can always pass them all.

## Purpose

- **Integration Testing**: Tests the complete conflict detection and resolution pipeline
- **Regression Prevention**: Ensures previously fixed issues don't reoccur
- **Scenario Accumulation**: Collects real-world conflict scenarios for comprehensive testing
- **Automated Execution**: Runs without user input for CI/CD integration

## Configuration

The workbench uses OpenAI's o4-mini model for LLM operations. Ensure you have:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run all scenarios:
```bash
python workbench/run_workbench.py
```

Run specific scenario:
```bash
python workbench/run_workbench.py --scenario indentation_conflict
python workbench/run_workbench.py --scenario false_positive_naming
python workbench/run_workbench.py --scenario internal_conflict
```

## Current Status

The workbench is fully functional and successfully:
- ✅ Runs without user interaction (automated conflict resolution)
- ✅ Uses OpenAI's gpt-4o-mini model
- ✅ Provides detailed reporting and validation
- ✅ Identifies ongoing issues with conflict detection

Current test results show that conflict detection still generates false positives, 
which is exactly what the workbench is designed to catch and help us fix.

## Adding New Scenarios

1. Create a new scenario file in `workbench/scenarios/`
2. Follow the scenario format (see existing examples)
3. Add the scenario to the workbench runner
4. Test the scenario to ensure it passes

## Scenario Format

Each scenario should define:
- **Description**: What the scenario tests
- **Input Files**: Test data files
- **Expected Conflicts**: What conflicts should be detected
- **Expected Resolution**: How conflicts should be resolved
- **Validation**: How to verify the scenario passed
