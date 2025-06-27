# Slicing Fragments Based on Filters

## Description
This use case covers the process of selecting and exporting specific prompt fragments from the library based on filtering criteria.

## Actors
- Developer who wants to extract specific fragments for use in a project

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized with fragments
- Fragments have appropriate tags or naming conventions for filtering

## Basic Flow
1. User wants to extract only Python testing-related fragments:
   ```bash
   pmx slice lang:python layer:testing
   ```

2. The tool:
   - Searches the library for fragments matching the specified filters
   - Collects and orders the matching fragments
   - Outputs the content to stdout (by default)

3. User can review the output and redirect it to a file if needed:
   ```bash
   pmx slice lang:python layer:testing > python-testing-guidelines.md
   ```

## Alternative Flows
- User wants to output directly to a file:
  ```bash
  pmx slice lang:python layer:testing --output python-testing-guidelines.md
  ```

- User wants to filter by facet name:
  ```bash
  pmx slice rules-python
  ```
  This would select all fragments with names starting with "rules-python".

- User wants to combine multiple filter types:
  ```bash
  pmx slice rules-python layer:testing domain:web
  ```

- User wants to see a list of matching fragment names without their content:
  ```bash
  pmx slice lang:python --list-only
  ```

## Postconditions
- Selected fragments are output to the specified destination (stdout or file)
- No changes are made to the original fragments in the library

## Notes
- Slicing is a read-only operation that doesn't modify the library
- The tool can detect uncategorized fragments and suggest new or alternative categories during slicing
- Users can accept, reject, or customize these suggestions interactively
- Slicing is useful for creating project-specific subsets of the prompt library