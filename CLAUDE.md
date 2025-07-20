# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Mixer is a command-line tool for managing reusable prompt instructions across GenAI coding assistants. The project follows the CRAFT framework (Chunk, Resist, Adapt, Fit, Transcend) for intelligent knowledge management, implementing a functional core with imperative shell architecture.

## Architecture

The codebase follows a layered architecture with clear separation of concerns:

- **CLI Layer** (`cli.py`): Typer-based command dispatch with rich terminal output
- **Commands Layer** (`commands/`): Command delegates implementing the Command pattern
  - `operations/`: Core domain operations (merge, commit, conflict detection)  
  - `interactions/`: Interactive command components for user workflows
- **Domain Layer** (`domain/`): Pure business logic and data structures
  - Knowledge management following CRAFT principles
  - Event-driven progress tracking system
  - Conflict detection and quarantine management
- **Gateways Layer** (`gateways/`): I/O isolation for external dependencies
  - Git operations, LLM integrations, ChromaDB vector storage
- **Utils Layer** (`utils/`): Cross-cutting concerns and helpers

## Key Components

### Knowledge Store System
The knowledge storage implements a multi-layered approach:
- `KnowledgeStore`: Abstract interface for storage operations
- `VectorKnowledgeStore`: Semantic search via ChromaDB embeddings  
- `FileKnowledgeStore`: Git-based file persistence
- `HybridKnowledgeStore`: Combines vector and file storage

### Event System
Event-driven architecture for progress tracking and system coordination:
- `ProgressEvents`: Domain events for tracking operations
- `EventDrivenProgress`: Observable progress system
- CLI observers provide real-time feedback

### Conflict Management
HDBSCAN-based clustering for intelligent conflict detection:
- Semantic similarity analysis for knowledge chunks
- Quarantine system for conflict resolution workflows
- Context-aware resolution strategies

## Development Commands

### Testing
```bash
# Run all tests (uses *_spec.py pattern)
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest src/context_mixer/domain/knowledge_store_spec.py
```

### Linting
```bash
# Critical errors only (used in pre-commit)
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics

# Full linting (warnings allowed)
flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --ignore=F401
```

### Installation & Setup
```bash
# Development installation
pip install -e ".[dev]"

# Install pre-commit hook
cp commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

### Documentation
```bash
# Serve docs locally
mkdocs serve

# Build documentation
mkdocs build
```

## Testing Patterns

- Tests use `*_spec.py` naming convention
- Test classes use `Describe*` pattern
- Test methods use `should_*` pattern  
- Co-located with implementation files
- Use `mocker.MagicMock(spec=ClassName)` for mocks
- Gateway testing focuses on integration, not mocked logic

## Code Conventions

- **Functional Core, Imperative Shell**: Pure business logic isolated from I/O
- **Gateway Pattern**: All external dependencies behind mockable interfaces
- **Event-Driven Design**: Domain events for cross-cutting concerns
- **CRAFT Principles**: Knowledge chunking, resistance to drift, adaptive granularity
- **Type Safety**: Full type hints with Pydantic for data validation
- **Command Pattern**: Consistent CLI interface with delegated command handlers

## Dependencies

Core runtime dependencies managed via `pyproject.toml`:
- `mojentic>=0.7.4`: LLM integration and validation
- `typer`: CLI framework  
- `rich`: Terminal formatting
- `chromadb`: Vector storage
- `pyyaml`: Configuration parsing

Development dependencies include `pytest`, `flake8`, `mkdocs-material`.

## CLI Usage Patterns

The tool provides a consistent command interface:
```bash
# Initialize context library
cmx init

# Ingest existing contexts  
cmx ingest ./project --project-id "web-app"

# Manage conflicts
cmx quarantine list
cmx quarantine resolve <chunk-id> accept "reason"

# Assemble contexts
cmx assemble copilot --project-ids "web-app" --token-budget 8192

# Slice by domains
cmx slice --granularity detailed --domains technical,business
```

## Mojentic Integration

When using Mojentic LLM library, configure logging before imports:
```python
import logging
logging.basicConfig(level=logging.WARN)
```

Use proper message role enumerations:
```python
from mojentic import LLMMessage, MessageRole
msg = LLMMessage(role=MessageRole.System, content="...")
```