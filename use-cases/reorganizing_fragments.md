# Reorganizing Fragments

## Description
This use case covers the process of analyzing existing fragments and reorganizing them to maintain a coherent facet structure in the prompt library.

## Actors
- Developer or team lead who wants to maintain organization in the prompt library

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized with fragments
- Some fragments may have inconsistent naming or categorization

## Basic Flow
1. User wants to analyze and reorganize the library structure:
   ```bash
   pmx reorganize
   ```

2. The tool:
   - Analyzes all existing fragments in the library
   - Identifies inconsistencies in naming conventions or categorization
   - Proposes a set of moves, renames, or tag changes to improve coherence
   - Shows a preview of the proposed changes

3. User reviews the proposed changes and confirms or modifies the suggestions

4. The tool:
   - Applies the approved changes to the library
   - Updates all references and dependencies
   - Automatically commits the changes with an LLM-generated message

## Alternative Flows
- User wants to focus on reorganizing a specific facet:
  ```bash
  pmx reorganize --facet rules
  ```

- User wants to automatically apply all suggested changes without confirmation:
  ```bash
  pmx reorganize --auto-apply
  ```

- User wants to export the reorganization plan without applying changes:
  ```bash
  pmx reorganize --export-plan reorganization-plan.md
  ```
  The user can review and edit this plan before applying it later.

## Postconditions
- Fragment structure is more coherent and consistent
- All changes are automatically committed to the Git repository
- References between fragments are updated to reflect the new structure

## Notes
- The reorganize process uses LLM to identify patterns and suggest improvements
- The tool maintains a history of reorganizations to prevent circular changes
- Reorganization is particularly useful after ingesting fragments from multiple sources
- The tool can detect uncategorized fragments and suggest new or alternative categories
- Users can accept, reject, or customize facet suggestions interactively