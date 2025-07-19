# Context Mixer Development Guidelines

## Project Overview

Provide a single command-line tool (with a roadmap toward optional desktop/UIs) that helps developers create, organise,
merge and deploy reusable prompt instructions across multiple GenAI coding assistants (e.g., GitHub Copilot,
Cursor/Windsor, Claude, Junie).

## Identity and Branding Guidelines

As Context Mixer is a Mojility product, maintain a consistent identity and branding across all materials:

- **Logo**: Use the official Mojility logo in all branding materials, in its original colors and proportions.
- **Color Palette**:
    - Accent Green: #6bb660
    - Dark Grey: #666767

## Tech Stack

- Python 3.12+
- Key Dependencies:
    - pydantic 2: Data validation
    - structlog: Logging
    - ollama/openai: LLM integration
    - pytest: Testing
    - MkDocs: Documentation

## Project Structure

- src/ # Python source code and tests
    - commands/ # Command delegates for the Typer CLI interface
        - operations/ # Elemental domain-specific operations (merge, sense conflicts)
    - gateways/ # Gateways to isolate I/O (network, disk, etc.) from logic
    - domain/ # Core logic and data structures for prompt management
    - cli.py # Main program entry-point and command dispatcher
    - config.py # Configuration storage and data-object
- use-cases/ # Numbered end-user scenarios, simple to complex
- docs/ # MkDocs documentation files
- README.md # Documentation for teammates and developers
- SPEC.md # Original design thoughts from the creator
- pyproject.toml # Python project metadata and dependencies

## Development Setup

1. Install Python 3.12 or higher.
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks (recommended):
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

When using Mojentic, unless you need the detailed logging, set it to warn. You must put this code in BEFORE ANY IMPORT STATEMENTS for the mojentic library.

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
- Follow google docstring style for code and tests.

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
