# Fresh Installation and Setup

## Description
This use case covers the initial installation and setup of the Prompt Mixer tool for a new user.

## Actors
- Developer or team member who wants to start using Prompt Mixer

## Preconditions
- Python 3.12+ installed
- Git installed on the system
- Access to either Ollama (local) or OpenAI (remote) LLM services

## Basic Flow
1. User installs Prompt Mixer using pipx:
   ```bash
   pipx install prompt-mixer
   ```

2. User initializes a new prompt library:
   ```bash
   pmx init
   ```

3. During initialization:
   - The tool creates a library in the default location (`$HOME/.prompt-mixer`)
   - The user is prompted to select an LLM provider and model
   - The user can optionally link to a remote Git repository with `--remote <url>`
   - Default taxonomy folders/structure is created

4. The tool generates a configuration file (`config.yaml`) with the selected LLM profile

## Alternative Flows
- User specifies non-default location for the library:
  ```bash
  pmx init --path /custom/path/to/library
  ```

- User specifies provider and model directly via command line:
  ```bash
  pmx init --provider ollama --model phi3
  ```
  or
  ```bash
  pmx init --provider openai --model gpt-4o
  ```

## Postconditions
- A new prompt library is initialized as a Git repository
- Default taxonomy structure is created
- LLM provider and model configuration is saved
- The library is ready for adding prompt fragments

## Notes
- The library is automatically initialized as a Git repository for version control
- All subsequent operations will use the configured LLM profile unless overridden