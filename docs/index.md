# Context Mixer

**A command-line tool to create, organize, merge and deploy reusable context instructions across multiple GenAI coding assistants.**

## What is Context Mixer?

Context Mixer helps developers manage context fragments for different AI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie) in a structured, version-controlled way. Instead of manually copying and pasting context instructions between projects or losing track of your best practices, Context Mixer provides a systematic approach to knowledge management and context assembly.

## The Problem

As developers increasingly rely on AI coding assistants, we face several challenges:

- **Context Fragmentation**: Best practices and context instructions scattered across multiple projects
- **Knowledge Drift**: Context instructions become outdated or inconsistent over time
- **Manual Duplication**: Copying context between projects leads to version drift and maintenance overhead
- **Token Limits**: AI assistants have context limits, requiring careful selection of relevant information
- **Domain Mixing**: Including irrelevant context (e.g., Python patterns when working in Java)

## The Solution

Context Mixer implements the **CRAFT framework** for intelligent knowledge management:

- **C**hunk: Knowledge into small domain-coherent units
- **R**esist: Knowledge contamination and drift
- **A**dapt: Knowledge granularity for task context
- **F**it: Knowledge selection to task requirements
- **T**ranscend: Knowledge storage implementation

## Key Features

### 🔄 Mix & Slice
Transform context fragments into variant-specific bundles tailored for different AI assistants and project contexts.

### 📥 Ingest & Normalize
Import existing context artifacts from multiple projects and normalize them into a consistent structure.

### 📚 Source-of-Truth Library
Maintain all context fragments under Git version control for history, collaboration, and consistency.

### ⚡ Token Optimization
Intelligently select and optimize context to fit within AI assistant token limits while maximizing relevance.

## Quick Example

```bash
# Initialize a new context library
cmx init

# Ingest existing contexts from your projects
cmx ingest ./my-react-project
cmx ingest ./my-python-api

# Assemble context for a specific target
cmx assemble --target copilot --project my-new-app

# Slice fragments by tags for specific needs
cmx slice lang:python layer:testing
```

## Who Should Use Context Mixer?

- **Individual Developers** who want to maintain consistent AI assistant context across projects
- **Development Teams** who need to share and standardize context practices
- **Organizations** implementing AI-assisted development at scale
- **Consultants** working across multiple client codebases with different requirements

## Getting Started

Ready to streamline your AI assistant context management? Check out our [Installation Guide](installation.md) to get started, then explore the [CRAFT Framework](craft-overview.md) to understand the principles behind effective context management.