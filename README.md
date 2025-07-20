# Context Mixer

A command-line tool to create, organize, merge and deploy reusable context instructions across multiple GenAI coding assistants.

## Purpose

Context Mixer helps developers manage context fragments for different AI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie) in a structured, version-controlled way using the **CRAFT framework** for intelligent knowledge management.

## Features

- **🔄 Mix & Slice** context fragments into variant-specific bundles with CRAFT-aware filtering
- **📥 Ingest & Normalize** existing context artifacts from multiple projects with project isolation
- **📚 Source-of-Truth Library** under Git for history and collaboration
- **🛡️ Conflict Detection & Quarantine** with comprehensive quarantine management system
- **⚡ Token Optimization** with intelligent selection to fit assistant limits
- **🎯 Project Context Isolation** to prevent cross-project knowledge contamination
- **🔍 Semantic Search** with vector-based knowledge retrieval and hybrid search capabilities
- **📊 Event-Driven Architecture** with comprehensive progress tracking and monitoring
- **🏗️ Command Pattern Implementation** with consistent CLI interface and extensible commands

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

# Ingest existing contexts with project identification
cmx ingest ./my-react-project --project-id "react-frontend" --project-name "React Frontend App"
cmx ingest ./my-python-api --project-id "python-api" --project-name "Python REST API"

# Review and manage quarantined knowledge conflicts
cmx quarantine list                                    # List all quarantined chunks
cmx quarantine review <chunk-id>                       # Review a specific chunk
cmx quarantine resolve <chunk-id> accept "Approved after review"  # Resolve conflicts
cmx quarantine stats                                   # View quarantine statistics

# Assemble contexts for a specific target with project filtering
cmx assemble copilot --project-ids "react-frontend,python-api" --token-budget 8192

# Slice context into content categories with CRAFT-aware filtering
cmx slice --granularity detailed --domains technical,business --project-ids "react-frontend"

# Open and explore your context library
cmx open
```

## License

MIT
