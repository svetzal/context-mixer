# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Mixer is a command-line tool for managing reusable prompt instructions across GenAI coding assistants. The project follows the CRAFT framework (Chunk, Resist, Adapt, Fit, Transcend) for intelligent knowledge management, implementing a functional core with imperative shell architecture.

**Requirements**: Python 3.12+, Git, access to Ollama (local) or OpenAI (remote) LLM services.

## Identity and Branding Guidelines

As Context Mixer is a Mojility product, maintain a consistent identity and branding across all materials:

- **Logo**: Use the official Mojility logo in all branding materials, in its original colors and proportions.
- **Color Palette**:
    - Accent Green: #6bb660
    - Dark Grey: #666767

## Universal Engineering Principles

- **Code is communication** — optimise for the next human reader.
- **Simple Design Heuristics** — guiding principles, not iron laws; consult the user when you need to break them.
    1. **All tests pass** — correctness is non‑negotiable.
    2. **Reveals intent** — code should read like an explanation.
    3. **No *knowledge* duplication** — avoid multiple spots that must change together; identical code is only a smell when it hides duplicate *decisions*.
    4. **Minimal entities** — remove unnecessary indirection, classes, or parameters.
- **Small, safe increments** — single‑reason commits; avoid speculative work (**YAGNI**).
- **Tests are the executable spec** — red first, green always; test behaviour not implementation.
- **Compose over inherit**; favour pure functions where practical, avoid side-effects.
- **Functional core, imperative shell** — isolate pure business logic from I/O and side effects; push mutations to the system boundaries, build mockable gateways at those boundaries.
- **Psychological safety** — review code, not colleagues; critique ideas, not authors.
- **Version‑control etiquette** — descriptive commit messages, branch from `main`, PRs require green CI.

## Tech Stack

- Python 3.12+
- Key Dependencies:
    - mojentic: LLM integration and data validation
    - typer: CLI framework
    - rich: Rich text and beautiful formatting
    - pyyaml: YAML parsing
    - fastmcp: MCP (Model Context Protocol) support
    - chromadb: Vector database
    - chroma-hnswlib: HNSW indexing for ChromaDB
    - pytest: Testing
    - MkDocs: Documentation

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

### Chunking Engine
The `ChunkingEngine` (`domain/chunking_engine.py`) converts content into knowledge chunks:
- `chunk_by_structured_output()`: Main chunking method using LLM structured output
- The system prompt in this method controls chunking granularity and behavior
- Chunk IDs are generated from content hash + concept + **source** (source ensures uniqueness across files)
- When modifying chunking behavior, update the LLM system prompt - not the parsing logic

### Ingestion Pipeline
The ingestion flow (`commands/ingest.py`) follows this sequence:
1. **File Reading**: Parallel file reading for performance
2. **Chunking**: `ChunkingEngine.chunk_by_structured_output()` creates atomic chunks
3. **Validation**: Each chunk validated for completeness
4. **Internal Conflict Detection**: Check conflicts between new chunks (batch processing)
5. **External Conflict Detection**: Check conflicts with existing stored chunks
6. **Resolution**: Apply resolution strategies (automatic, LLM-based, or interactive)
7. **Storage**: Store resolved chunks in vector knowledge store
8. **Context.md Generation**: Write summary file for compatibility

### Workbench System
The workbench (`workbench/`) provides integration testing for conflict detection:

```bash
# Run all scenarios
python workbench/workbench_cli.py run

# Run specific scenario
python workbench/workbench_cli.py run --scenario indentation_conflict

# List available scenarios
python workbench/workbench_cli.py list-scenarios
```

**Scenario locations**: `workbench/scenarios/*.py`

Each scenario defines:
- `input_files`: Test content with potential conflicts
- `expected_conflicts`: What conflicts should/shouldn't be detected
- `validation_checks`: Assertions on final output content
- `expected_chunk_counts`: Optional validation of chunking granularity

When fixing conflict detection issues, create or update a workbench scenario first to capture the expected behavior, then implement the fix.

## Development Commands

**IMPORTANT**: Always activate the virtual environment before running Python commands:
```bash
source .venv/bin/activate
```

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
- Do not write Given/When/Then or Arrange/Act/Assert comments; separate phases with a blank line

## Code Conventions

- **Functional Core, Imperative Shell**: Pure business logic isolated from I/O
- **Gateway Pattern**: All external dependencies behind mockable interfaces
- **Event-Driven Design**: Domain events for cross-cutting concerns
- **CRAFT Principles**: Knowledge chunking, resistance to drift, adaptive granularity
- **Type Safety**: Full type hints with Pydantic (not `@dataclass`) for data validation
- **Command Pattern**: Consistent CLI interface with delegated command handlers
- Favor declarative code styles and comprehensions over imperative loops
- Max complexity: 10 per function

## Example Scripts

Example scripts in `src/_examples/`:
- **Do not use `test_*` prefix** for filenames or method names - reserved for pytest
- Use descriptive names: `chunking_experiment.py`, `llm_demo.py`
- Use method prefixes: `check_*`, `run_*`, `demo_*`, `experiment_*`

## Procedural Guidelines

- When identifying technical debt or approach issues, add analysis to `BACKLOG.md`
- When identifying future concerns or features, add to `PLAN.md`
- After changes complete with passing tests, run the workbench to verify conflict detection scenarios:
  ```bash
  python workbench/workbench_cli.py run
  ```
- When fixing conflict detection bugs, first create/update a workbench scenario that reproduces the issue

## Key Files for Conflict Detection

When working on conflict detection issues, these are the primary files:

| File | Purpose |
|------|---------|
| `domain/chunking_engine.py` | Creates atomic chunks from content; LLM prompt controls granularity |
| `commands/ingest.py` | Orchestrates the full ingestion pipeline including conflict detection |
| `commands/operations/merge.py` | `detect_conflicts()` and `detect_conflicts_batch()` functions |
| `domain/context_aware_prompts.py` | Builds context-aware prompts to reduce false positives |
| `domain/context_detection.py` | Detects context types (architectural, platform, environment, etc.) |
| `commands/interactions/conflict_resolution_strategies.py` | Resolution strategy implementations |
| `workbench/scenarios/*.py` | Integration test scenarios for conflict detection |

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

## LLM Tool Development

1. When writing a new LLM tool, model the implementation after `mojentic.llm.tools.date_resolver.ResolveDateTool`.
2. For LLM-based tools, accept the `LLMBroker` object as a parameter in the tool's initializer.
3. Don't ask the LLM to generate JSON directly; use `LLMBroker.generate_object()` instead.

## Documentation

- Built with MkDocs and Material theme.
- API documentation uses mkdocstrings:
  ```markdown
  ::: context_mixer.llm.MessageBuilder
      options:
          show_root_heading: true
          merge_init_into_class: false
          group_by_category: false
  ```
- Supports mermaid.js diagrams
- Build docs locally: `mkdocs serve`
- Build for production: `mkdocs build`
- Keep the navigation tree in `mkdocs.yml` up to date with changes in `docs/`.

## Release Process

This project follows Semantic Versioning (SemVer): MAJOR.MINOR.PATCH.

### Preparing a Release

1. Update **CHANGELOG.md**:
    - Document all notable changes under **[Unreleased]**.
    - When releasing, move **[Unreleased]** changes to a new version section (`## [x.y.z] - YYYY-MM-DD`).
    - Categorize under **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**, **Security**.
2. Update the version number in `pyproject.toml` following SemVer principles.
3. Update documentation (README and other docs) to reflect changes.
4. Final verification:
   ```bash
   flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
   pytest
   ```

### Release Types

- **Major Releases (x.0.0)**: Breaking API changes, significant architectural changes, removal of deprecated features. Provide migration guides.
- **Minor Releases (0.x.0)**: New features, non-breaking enhancements, deprecation notices. Update README and docs.
- **Patch Releases (0.0.x)**: Bug fixes, security updates, performance improvements, documentation corrections. Maintain strict backward compatibility.