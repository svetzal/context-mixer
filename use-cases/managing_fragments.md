# Creating and Managing Fragments

## Description
This use case covers the creation, editing, and management of prompt fragments within the Prompt Mixer library.

## Actors
- Developer or team member who wants to manage prompt fragments

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized

## Basic Flow
1. User creates a new fragment:
   ```bash
   pmx create rules-python-testing
   ```
   This opens the default editor with a new file.

2. User edits the fragment content in their editor, adding prompt instructions for Python testing.

3. Upon saving and closing the editor, the tool:
   - Validates the fragment format
   - Automatically commits the change with an LLM-generated commit message
   - Adds appropriate metadata tags based on the fragment name

4. User can view all fragments:
   ```bash
   pmx list
   ```

5. User can edit an existing fragment:
   ```bash
   pmx edit rules-python-testing
   ```

## Alternative Flows
- User renames a fragment:
  ```bash
  pmx rename rules-python-testing rules-python-pytest
  ```

- User deletes a fragment:
  ```bash
  pmx delete rules-python-testing
  ```

- User adds custom tags to a fragment:
  ```bash
  pmx tag rules-python-testing --add lang:python layer:testing
  ```

## Postconditions
- Fragment is created/edited/renamed/deleted in the library
- Changes are automatically committed to the Git repository
- Fragment is properly tagged for later filtering and assembly

## Notes
- Fragment filenames use kebab-case notation to encode their position in the logical hierarchy
- The first segment of the filename represents the facet (e.g., "rules")
- Additional segments represent sub-facets (e.g., "python", "testing")
- All operations automatically generate Git commits with meaningful messages