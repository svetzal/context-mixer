# Deriving Rules from Code

## Description
This use case covers the process of analyzing an existing codebase to automatically generate prompt fragments that capture the coding style, patterns, and conventions used in the code.

## Actors
- Developer who wants to create prompt fragments based on an exemplary codebase

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized
- An exemplary codebase exists that demonstrates desired coding practices

## Basic Flow
1. User wants to generate prompt fragments from an existing codebase:
   ```bash
   pmx derive --from-code ./src
   ```

2. The tool:
   - Scans the codebase to identify programming languages used
   - Analyzes code structure, patterns, naming conventions, and formatting
   - Uses LLM to infer coding rules and best practices from the observed patterns
   - Proposes a set of rule fragments organized by language and topic

3. User reviews the proposed fragments and confirms or modifies the suggestions

4. The tool:
   - Creates the approved fragments in the library
   - Automatically commits the changes with an LLM-generated message
   - Applies appropriate tags based on the content analysis

## Alternative Flows
- User wants to focus on a specific language:
  ```bash
  pmx derive --from-code ./src --lang python
  ```

- User wants to focus on specific aspects of the code:
  ```bash
  pmx derive --from-code ./src --aspects "naming,formatting,testing"
  ```

- User wants to preview the derived rules without adding them to the library:
  ```bash
  pmx derive --from-code ./src --preview
  ```

## Postconditions
- New rule fragments are created in the library based on the analyzed codebase
- Changes are automatically committed to the Git repository
- Fragments are properly tagged for later filtering and assembly

## Notes
- The derive process uses LLM to identify patterns and extract implicit rules from code
- The quality of derived rules depends on the consistency and quality of the input codebase
- Derived rules can be edited and refined manually after creation
- This feature is particularly useful for teams wanting to standardize on existing practices
- The tool can detect and highlight inconsistencies in the codebase for user review