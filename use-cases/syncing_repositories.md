# Syncing with Remote Repositories

## Description
This use case covers the process of synchronizing the local prompt library with a remote Git repository to share fragments with team members.

## Actors
- Developer who wants to share prompt fragments with a team
- Team members who want to access shared prompt fragments

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized
- A remote Git repository has been linked to the library

## Basic Flow
1. User wants to synchronize their local library with the remote repository:
   ```bash
   pmx sync
   ```

2. The tool:
   - Performs a `git pull` to fetch and merge changes from the remote repository
   - Performs a `git push` to send local changes to the remote repository
   - Handles basic merge conflicts automatically when possible

3. User can continue working with the updated library

## Alternative Flows
- User wants to only pull changes from the remote:
  ```bash
  pmx sync --pull
  ```

- User wants to only push local changes to the remote:
  ```bash
  pmx sync --push
  ```

- User wants to use rebase strategy instead of merge:
  ```bash
  pmx sync --rebase
  ```

- Tool detects merge conflicts that cannot be resolved automatically:
  1. The tool notifies the user about the conflicts
  2. User resolves the conflicts manually using their preferred Git tools
  3. User runs `pmx sync` again to complete the synchronization

## Postconditions
- Local library is synchronized with the remote repository
- All changes are committed and pushed to the remote (if --push or default mode)
- Local library contains all changes from the remote (if --pull or default mode)

## Notes
- The default sync operation performs both pull and push to ensure full synchronization
- The tool uses merge strategy by default to minimize conflicts in collaborative environments
- Rebase strategy is available for users who prefer a cleaner history
- The tool relies on the system Git executable for all Git operations
- All changes are automatically committed with LLM-generated messages before syncing