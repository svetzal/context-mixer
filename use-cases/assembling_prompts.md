# Assembling Prompts for Specific Targets

## Description
This use case covers the process of assembling selected prompt fragments into a format suitable for a specific AI assistant tool (target).

## Actors
- Developer who wants to generate a ready-to-use prompt bundle for a specific AI assistant

## Preconditions
- Prompt Mixer is installed
- A prompt library has been initialized with fragments
- The target AI assistant is supported by Prompt Mixer

## Basic Flow
1. User decides to assemble a prompt for GitHub Copilot:
   ```bash
   pmx assemble --target copilot
   ```

2. The tool:
   - Collects all relevant fragments from the library
   - Orders them appropriately
   - Renders them into the format required by Copilot (a single Markdown/JSON hybrid file)
   - Outputs the file as `copilot-instructions.md` in the current directory

3. User can now use this file with GitHub Copilot

## Alternative Flows
- User wants to assemble prompts for a different target:
  ```bash
  pmx assemble --target claude
  ```
  This would generate a directory of Markdown files formatted for Claude.

- User wants to override the default LLM profile for this operation:
  ```bash
  pmx assemble --target copilot --profile openai://gpt-4o
  ```

- User wants to specify a custom output location:
  ```bash
  pmx assemble --target copilot --output /path/to/output
  ```

- User wants to include only specific fragments based on tags:
  ```bash
  pmx assemble --target copilot --filter lang:python,layer:testing
  ```

## Postconditions
- A prompt bundle is generated in the format required by the target AI assistant
- The bundle is saved to the specified location (or current directory by default)
- If the assembled content exceeds token limits, warnings are displayed with suggestions for condensation

## Notes
- Different targets require different output formats (single file vs. directory of files)
- The assembly process is handled by pluggable renderers specific to each target
- New renderers can be added via plugins to support additional AI assistants
- Token counting is performed to warn about potential size issues