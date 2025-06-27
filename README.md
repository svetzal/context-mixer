# Prompt Mixer

A command-line tool to create, organize, merge and deploy reusable prompt instructions across multiple GenAI coding assistants.

## Purpose

Prompt Mixer helps developers manage prompt fragments for different AI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie) in a structured, version-controlled way.

## Features

- **Mix & Slice** prompt fragments into variant-specific bundles
- **Ingest & Normalize** existing prompt artifacts from multiple projects
- **Maintain a Source-of-Truth Library** under Git for history and collaboration
- **Token Optimization** to fit assistant limits

## Installation

```bash
# Install with pipx (recommended)
pipx install prompt-mixer

# Or with pip
pip install prompt-mixer

# For development
git clone https://github.com/svetzal/prompt-mixer.git
cd prompt-mixer
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- Git
- Access to either Ollama (local) or OpenAI (remote) LLM services

## Quick Start

```bash
# Initialize a new prompt library
pmx init

# Assemble prompts for a specific target
pmx assemble --target copilot

# Slice fragments by tags
pmx slice lang:python layer:testing

# Ingest existing prompts
pmx ingest ./my-project

# Sync with remote repository
pmx sync
```

## License

MIT
