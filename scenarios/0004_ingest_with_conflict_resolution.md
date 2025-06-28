# Scenario 4: Ingest with Conflict Resolution

## Description

This scenario demonstrates the conflict detection and resolution feature of the prompt-mixer. When ingesting a file that contains guidance that conflicts with the existing library, the user is consulted to resolve the conflict.

## Prerequisites

- The prompt-mixer library already contains a context.md file with some guidance
- A new file contains guidance that conflicts with the existing guidance

## Steps

1. Create a context.md file in the library with the following content:

```markdown
# Project Guidelines

## Code Style

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Use camelCase for variable names
- Use PascalCase for class names
- Use snake_case for function names
```

2. Create a new file called conflicting_guidelines.md with the following content:

```markdown
# Coding Standards

## Style Guide

- Use 2 spaces for indentation
- Maximum line length is 80 characters
- Use camelCase for variable names
- Use PascalCase for class names
- Use snake_case for function names
```

3. Run the ingest command:

```bash
pmx ingest conflicting_guidelines.md
```

## Expected Behavior

1. The prompt-mixer should detect the conflict in indentation (4 spaces vs 2 spaces) and line length (100 characters vs 80 characters)
2. The user should be presented with the conflicting guidance and asked to choose which is correct
3. After the user makes a choice, the prompt-mixer should merge the content, incorporating the user's resolution
4. The resulting context.md file should contain the merged content with the resolved conflicts

## Example Output

```
Ingesting prompts from: conflicting_guidelines.md

Conflict Detected!
Description: There is a conflict in the indentation guidance and maximum line length.

Conflicting Guidance:

1. From existing:
- Use 4 spaces for indentation
- Maximum line length is 100 characters

2. From new:
- Use 2 spaces for indentation
- Maximum line length is 80 characters

Which guidance is correct? (Enter the number): 1

Successfully merged prompt with existing context.md
```

The resulting context.md file should contain the merged content with the resolved conflicts, using the user's choice (4 spaces for indentation and 100 characters for line length).