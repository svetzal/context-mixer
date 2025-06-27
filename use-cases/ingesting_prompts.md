# Ingesting Existing Prompts

## Description
This use case covers the process of importing existing prompt artifacts from projects into the Prompt Mixer library.

## Actors
- Developer who wants to import existing prompt files into a centralized library

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized
- One or more projects with existing prompt artifacts (e.g., copilot.json, junie_guidelines.md)

## Basic Flow
1. User wants to import prompts from an existing project:
   ```bash
   pmx ingest ./my-old-repo
   ```

2. The tool:
   - Scans the specified directory for recognized prompt file types
   - Parses the content of these files
   - Analyzes the content to identify distinct concepts
   - Proposes a set of fragments to create from the content

3. User reviews the proposed fragments and confirms or modifies the suggestions

4. The tool:
   - Creates the approved fragments in the library
   - Automatically commits the changes with an LLM-generated message
   - Applies appropriate tags based on the content analysis

## Alternative Flows
- User wants to ingest a specific file:
  ```bash
  pmx ingest ./my-old-repo/copilot.json
  ```

- User wants to ingest from multiple sources:
  ```bash
  pmx ingest ./repo1 ./repo2 ./specific-file.md
  ```

- Tool detects duplicate or similar content:
  1. The tool shows a comparison between existing and new content
  2. User chooses to keep existing, use new, or merge the content
  3. The tool updates the library accordingly

## Postconditions
- New fragments are created in the library based on the ingested content
- Changes are automatically committed to the Git repository
- Fragments are properly tagged for later filtering and assembly

## Notes
- The ingest process attempts to intelligently split monolithic prompts into logical fragments
- Duplicate detection helps prevent redundant content in the library
- The tool can recognize various prompt file formats used by different AI assistants
- User confirmation is required before committing changes to prevent unwanted imports