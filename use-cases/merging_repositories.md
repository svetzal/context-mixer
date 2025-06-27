# Merging Repositories

## Description
This use case covers the process of analyzing multiple repositories and building unified fragments by combining similar content from different sources.

## Actors
- Developer or team lead who wants to consolidate prompt fragments from multiple repositories

## Preconditions
- Prompt Mixer is installed
- Multiple repositories or projects with prompt artifacts exist
- User has access to all repositories to be merged

## Basic Flow
1. User wants to merge prompt fragments from multiple repositories:
   ```bash
   pmx merge repo1 repo2
   ```

2. The tool:
   - Scans each repository for prompt artifacts
   - Analyzes the content to identify similar fragments across repositories
   - Groups related fragments together
   - Proposes unified fragments that combine the best elements from each source

3. User reviews the proposed unified fragments and confirms or modifies the suggestions

4. The tool:
   - Creates the approved unified fragments in the library
   - Automatically commits the changes with an LLM-generated message
   - Applies appropriate tags based on the content analysis

## Alternative Flows
- User wants to merge only Python-related fragments:
  ```bash
  pmx merge repo1 repo2 --lang python
  ```

- User wants to specify output location for the merged fragments:
  ```bash
  pmx merge repo1 repo2 --output ./merged-fragments
  ```

- User wants to see a diff of similar fragments without merging:
  ```bash
  pmx merge repo1 repo2 --diff-only
  ```

- Tool detects conflicts between fragments:
  1. The tool shows the conflicting sections
  2. User chooses which version to keep or provides a manual resolution
  3. The tool creates the resolved fragment

## Postconditions
- New unified fragments are created in the library
- Changes are automatically committed to the Git repository
- Fragments are properly tagged for later filtering and assembly

## Notes
- The merge process uses similarity detection to identify related fragments
- LLM assistance helps create coherent unified fragments from disparate sources
- The process preserves unique information from each source while eliminating redundancy
- User confirmation is required before committing changes to ensure quality