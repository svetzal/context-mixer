# Context Mixer

A command-line tool to create, organize, merge and deploy reusable context instructions across multiple GenAI coding assistants.

## Purpose

Context Mixer helps developers manage context fragments for different AI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie) in a structured, version-controlled way.

## Features

- **Mix & Slice** context fragments into variant-specific bundles
- **Ingest & Normalize** existing context artifacts from multiple projects
- **Maintain a Source-of-Truth Library** under Git for history and collaboration
- **Token Optimization** to fit assistant limits

## Installation

```bash
# Install with pipx (recommended)
pipx install context-mixer

# Or with pip
pip install context-mixer

# For development
git clone https://github.com/svetzal/context-mixer.git
cd context-mixer
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- Git
- Access to either Ollama (local) or OpenAI (remote) LLM services

## Quick Start

```bash
# Initialize a new context library
cmx init

# Assemble contexts for a specific target
cmx assemble --target copilot

# Slice fragments by tags
cmx slice lang:python layer:testing

# Ingest existing contexts
cmx ingest ./my-project

# Sync with remote repository
cmx sync
```

## License

MIT
