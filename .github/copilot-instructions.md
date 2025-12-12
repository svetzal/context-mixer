# Context Mixer Development Guidelines

## Project Overview

Provide a single **command‑line tool** (with a roadmap toward optional desktop/UIs) that helps developers *create, organise, merge* and *deploy* reusable prompt instructions across multiple GenAI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie).

The project leverages the **CRAFT** (Chunk, Resist, Adapt, Fit, Transcend) philosophy (in docs/craft-overview.md and THEORY.md) for intelligent knowledge management and context assembly.

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
    3. **No *****knowledge***** duplication** — avoid multiple spots that must change together; identical code is only a
       smell when it hides duplicate *decisions*.
    4. **Minimal entities** — remove unnecessary indirection, classes, or parameters.
- **Small, safe increments** — single‑reason commits; avoid speculative work (**YAGNI**).
- **Tests are the executable spec** — red first, green always; test behaviour not implementation.
- **Compose over inherit**; favour pure functions where practical, avoid side-effects.
- **Functional core, imperative shell** — isolate pure business logic from I/O and side effects; push mutations to the
  system boundaries, build mockable gateways at those boundaries.
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

## Project Structure

- src/ # Python source code and tests
    - context_mixer/ # Main package
        - commands/ # Command delegates for the Typer CLI interface
            - operations/ # Elemental domain-specific operations (merge, sense conflicts)
            - interactions/ # Interactive command components
        - gateways/ # Gateways to isolate I/O (network, disk, etc.) from logic
        - domain/ # Core logic and data structures for prompt management
        - utils/ # Utility functions and helpers
        - examples/ # Example code and demonstrations
        - cli.py # Main program entry-point and command dispatcher
        - config.py # Configuration storage and data-object
        - spec_helpers.py # Testing utilities and helpers
    - _examples/ # Example scripts and demonstrations
- scenarios/ # End-user scenarios and use cases
- docs/ # MkDocs documentation files
- workbench/ # Development workspace and experiments
- refs/ # Reference materials and documentation
- README.md # Documentation for teammates and developers
- SPEC.md # Original design thoughts from the creator
- THEORY.md # Theoretical foundations and CRAFT philosophy
- pyproject.toml # Python project metadata and dependencies

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

## Development Setup

1. Install Python 3.12 or higher.
2. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

**CRITICAL: Python Environment**
- **ALWAYS** activate the virtual environment with `source .venv/bin/activate` before running any Python commands
- **NEVER** use `python3` or `pip3` directly - the environment is established and ready
- Use `python` and `pip` (without the 3) after activating the environment

4. Install pre-commit hooks (recommended):
   ```bash
   cat > .git/hooks/pre-commit << 'EOL'
   #!/bin/sh
   echo "Running flake8..."
   # stop the validation if there are Python syntax errors or undefined names
   flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
   if [ $? -ne 0 ]; then
       echo "flake8 found critical errors. Commit aborted."
       exit 1
   fi
   # exit-zero treats all errors as warnings
   flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --ignore=F401

   echo "Running pytest..."
   python -m pytest
   exit_code=$?
   exit $exit_code
   EOL
   chmod +x .git/hooks/pre-commit
   ```

## Coding Guidelines

- Use the Gateway pattern to isolate I/O from logic. Gateways should delegate to OS or libraries; do not test logic in
  gateways.
- Use pydantic2 (not `@dataclass`) for data objects requiring strong typing.
- Favor declarative code styles over imperative styles.
- Use type hints for all functions and methods.
- Favor list and dictionary comprehensions over for loops.
- Co-locate tests with implementation; write tests for all new functionality.
- Document using numpy-style docstrings.
- Keep code complexity low (max complexity: 10).

### Using Mojentic

When using Mojentic, unless you need the detailed logging, set it to warn. You must put this code in BEFORE ANY IMPORT
STATEMENTS for the mojentic library.

```
import logging
logging.basicConfig(level=logging.WARN)
```

When creating LLMMessage objects, use this pattern with the correct enumeration:

```
m = LLMMessage(
    role=MessageRole.System, # MessageRole.User is the default
    content="Some string",
)
```

## Testing Guidelines

- Tests are co-located with implementation files; test filenames use the `*_spec.py` suffix.
- Run tests:
  ```bash
  pytest
  ```
- Linting:
  ```bash
  flake8 src --max-line-length=127
  ```
- Follow numpy-style docstrings for code and tests.

### Testing Best Practices

- Use pytest with the `mocker` fixture for mocks.
- Create mocks with `mocker.MagicMock()`; use `spec=ClassName` when mocking specific classes.
- Use `@pytest.fixture` for fixtures. Keep fixtures small and focused.
- Do not write Given/When/Then or Arrange/Act/Assert comments; separate phases with a blank line.
- Do not use conditional assertions—each test must fail for only one clear reason.
- Use descriptive test names in `Describe*` classes; do not add docstrings to fixtures or Describe classes.
- For complex assertions, break into multiple clear assertions.
- Tests MUST ALWAYS PASS, if there are failing tests, your task isn't done.

#### Testing Example

Good example:

```python
import pytest
from typing import AnyStr
from context_mixer.some_module import SomeClass


@pytest.fixture
def mock_dependency(mocker):
    return mocker.MagicMock(spec=SomeDependency)


@pytest.fixture
def subject(mock_dependency):
    return SomeClass(mock_dependency)


class DescribeSomeClass:
    def should_do_something_when_condition(self, subject, mock_dependency):
        mock_dependency.get_message.return_value = AnyStr
        result = subject.do_something()
        assert result == expected_value
        assert mock_dependency.some_method.called_once()
```

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
  Adjust the module/class name after `:::` as needed.
- Supports mermaid.js diagrams:
  ```mermaid
  graph LR
      A[Doc] --> B[Feature]
  ```
- Build docs locally:
  ```bash
  mkdocs serve
  ```
- Build for production:
  ```bash
  mkdocs build
  ```
- Markdown conventions:
    - Use `#` for top-level headings.
    - Put blank lines above and below lists, headings, code blocks, and quotes.
- Keep the navigation tree in `mkdocs.yml` up to date with changes in `docs/`.

## LLM Tool Development

1. When writing a new LLM tool, model the implementation after `mojentic.llm.tools.date_resolver.ResolveDateTool`.
2. For LLM-based tools, accept the `LLMBroker` object as a parameter in the tool’s initializer.
3. Don’t ask the LLM to generate JSON directly; use `LLMBroker.generate_object()` instead.

## Running Examples

Example scripts are in `src/_examples/`:

- `simple_llm.py`: Basic LLM usage
- `chat_session.py`: Chat interactions
- `working_memory.py`: Context management
- Images for examples are in `src/_examples/images/`

### Example Script Naming Convention

**Important**: Do not use the `test_*` prefix for example/experiment scripts in `src/_examples/`. The `test_*` prefix is
reserved for actual pytest test files and will cause IDEs to incorrectly identify experiment scripts as test files to be
run with pytest. Use descriptive names like `chunking_experiment.py`, `llm_demo.py`, or `feature_exploration.py`
instead.

**Method Naming in Experiment Scripts**: Similarly, do not use the `test_*` prefix for method names within experiment
scripts. Use descriptive prefixes like `check_*`, `run_*`, `demo_*`, or `experiment_*` instead. For example, use
`check_chunking_approaches()` instead of `test_chunking_approaches()`.

Usage example:

```python
from context_mixer.llm import LLMBroker
from context_mixer.agents import BaseLLMAgent
```

## Procedural Guidelines

When realizing we have accumulated some kind of technical debt, an issue with our approach, or a problem we will run
into, incorporate the analysis for it into BACKLOG.md.

When realizing we should consider a specific concern or feature in the future, incorporate it into PLAN.md.

When changes are complete, and all unit tests are passing, run the workbench to verify conflict detection scenarios:
```bash
python workbench/workbench_cli.py run
```

When fixing conflict detection bugs, first create/update a workbench scenario that reproduces the issue.

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

- **Major Releases (x.0.0)**: Breaking API changes, significant architectural changes, removal of deprecated features.
  Provide migration guides.
- **Minor Releases (0.x.0)**: New features, non-breaking enhancements, deprecation notices. Update README and docs.
- **Patch Releases (0.0.x)**: Bug fixes, security updates, performance improvements, documentation corrections. Maintain
  strict backward compatibility.
