# Token Counting

## Description
This use case covers the process of analyzing prompt fragments or assembled prompts to count tokens and ensure they fit within the limits of target AI assistants.

## Actors
- Developer who wants to check token usage of prompt fragments or assembled prompts

## Preconditions
- Prompt Mixer is installed
- Prompt fragments or assembled prompts exist to be analyzed

## Basic Flow
1. User wants to check the token count of a specific file:
   ```bash
   pmx tokens ./copilot-instructions.md
   ```

2. The tool:
   - Analyzes the content of the file
   - Counts tokens using the appropriate tokenizer for the default LLM profile
   - Displays the total token count and a breakdown by sections if applicable
   - Provides warnings if the token count exceeds common model limits

3. User can use this information to optimize their prompts if needed

## Alternative Flows
- User wants to check token counts for all fragments in a directory:
  ```bash
  pmx tokens ./fragments/
  ```
  The tool provides a summary of token counts for each file and a total.

- User wants to check token counts for a specific model:
  ```bash
  pmx tokens ./copilot-instructions.md --profile openai://gpt-4o
  ```
  The tool uses the tokenizer appropriate for the specified model.

- User wants a detailed breakdown of token usage:
  ```bash
  pmx tokens ./copilot-instructions.md --verbose
  ```
  The tool provides a more detailed analysis, including token distribution by section.

- User wants to check if a prompt will fit within a specific token limit:
  ```bash
  pmx tokens ./copilot-instructions.md --limit 8192
  ```
  The tool provides a clear pass/fail indication and suggestions for reduction if needed.

## Postconditions
- Token count information is displayed to the user
- No changes are made to the analyzed files

## Notes
- Token counting is a read-only operation that doesn't modify files
- Different models use different tokenization methods, so counts may vary by model
- The tool uses Mojentic utilities for token counting/estimation
- Token counts are estimates and may differ slightly from what the actual AI service calculates
- The tool can provide suggestions for reducing token count if limits are exceeded