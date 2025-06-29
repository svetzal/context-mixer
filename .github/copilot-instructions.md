# Prompt Mixer Development Guidelines

## Project Overview

Provide a single **command‑line tool** (with a roadmap toward optional desktop/UIs) that helps developers *create, organise, merge* and *deploy* reusable prompt instructions across multiple GenAI coding assistants (e.g., GitHub Copilot, Cursor/Windsor, Claude, Junie).

## Identity and Branding Guidelines

As prompt-mixer is a Mojility product, it is important to maintain a consistent identity and branding across all materials. The following guidelines should be followed:

- **Logo**: Use the official Mojility logo in all branding materials. The logo should be used in its original colors and proportions
- **Color Palette**: Use the official Mojility color palette for all branding materials. The primary colors are:
  - Accent Green: #6bb660
  - Dark Grey: #666767

## Tech Stack

- Python 3.11+
- Key Dependencies:
  - pydantic 2: Data validation
  - structlog: Logging
  - ollama/openai: LLM integration
  - pytest: Testing
  - MkDocs: Documentation

## Project Structure

```
- src/             # Python source code and tests
  - commands/      # Command delegates for the Typer CLI interface
    - operations/  # Elemental domain specific operations (merge, sense conflicts)
  - gateways/      # Gateways to separate logic from I/O (don't test across I/O)
  - domain/        # Logic and data structurs within the realm of prompt mgmt
  - cli.py         # Main program entry-point and command dispatcher
  - config.py      # Configuration storage and data-object
- use-cases/       # Numbered end-user scenarios, simple to complex
- README.md        # Documentation for teammates and developers
- SPEC.md          # Original design thoughts from the creator
- pyproject.toml   # Python project metadata and dependencies
```

## Development Setup
1. Install Python 3.12 or higher
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks (recommended):
   ```bash
   # Create a pre-commit hook that runs pytest
   cat > .git/hooks/pre-commit << 'EOL'
   #!/bin/sh

   # Run pytest
   echo "Running pytest..."
   python -m pytest

   # Store the exit code
   exit_code=$?

   # Exit with pytest's exit code
   exit $exit_code
   EOL

   # Make the hook executable
   chmod +x .git/hooks/pre-commit
   ```

## Coding Guidelines

- Use the Gateway pattern to isolate I/O (network, disk, etc.) from logic
  - Gateways should contain minimal to no logic, simply delegate to the OS or
    libraries that perform I/O
  - Gateways should not be tested, to avoid mocking what we don't own

## Testing Guidelines
- Tests are co-located with implementation files (test file must be in the same folder as the implementation)
- We write tests as specifications, therefore you can find all the tests in the *_spec.py files
- Run tests: `pytest`
- Linting: `flake8 src`
- Code style:
  - Max line length: 127
  - Max complexity: 10
  - Follow numpy docstring style

### Testing Best Practices
- Use pytest for testing, with mocker if you require mocking
  - Always use the pytest `mocker` fixture for creating mocks: `def test_something(mocker):`
  - Create mocks with `mocker.MagicMock()` instead of importing from unittest.mock
  - When mocking a specific class, use `mocker.MagicMock(spec=ClassName)` to ensure type safety
- Use `@pytest.fixture` markers for pytest fixtures
  - Keep fixtures concise and focused on a single responsibility
  - Break up fixtures into smaller fixtures if they are too large
  - Pass the mocker fixture to other fixtures that need to create mocks: `def mock_something(mocker):`
- Test structure and readability:
  - Do not write docstring comments on your `should_` methods
  - Do not write docstring comments on your `Describe*` classes
  - Do not write comments in tests to delineate act, arrange, or assert phases
  - Separate test phases with a single blank line (not multiple blank lines)
  - Use descriptive variable and test method names instead of comments
- Test assertions:
  - Each test must fail for only one clear reason
  - Do not write conditional statements in tests
  - For complex assertions, break them into multiple clear assertions
  - When checking for text in output, prefer direct string checks over complex conditions
  - Use `typing.AnyStr` for string literals in tests when the actual content doesn't matter (e.g., in mock return values)

### Testing Examples

#### Good Example - Proper Fixture and Test Structure:
```python
import pytest
from typing import AnyStr
from prompt_mixer.some_module import SomeClass

@pytest.fixture
def mock_dependency(mocker):
    return mocker.MagicMock(spec=SomeDependency)

@pytest.fixture
def subject(mock_dependency):
    return SomeClass(mock_dependency)

class DescribeSomeClass:
    def should_do_something_when_condition(self, subject, mock_dependency):
        # Using AnyStr for mock return value where content doesn't matter
        mock_dependency.get_message.return_value = AnyStr

        result = subject.do_something()

        assert result == expected_value
        assert mock_dependency.some_method.called_once()
```

#### Bad Example - Avoid These Patterns:
```python
import pytest
from unittest.mock import MagicMock  # Don't import directly from unittest.mock

@pytest.fixture
def mock_dependency():
    """Create a mock dependency."""  # Don't add docstrings to fixtures
    return MagicMock()  # Use mocker.MagicMock() instead

class DescribeSomeClass:
    """Tests for SomeClass."""  # Don't add docstrings to Describe classes

    def should_do_something_when_condition(self, subject, mock_dependency):
        # Arrange
        mock_dependency.setup()  # Don't add comments for test phases

        # Act
        result = subject.do_something()

        # Assert
        assert result == expected_value  # Keep assertions simple and direct

        # Check if method was called  # Don't add explanatory comments
        if mock_dependency.some_method.called:  # Don't use conditionals in assertions
            assert True
        else:
            assert False
```

## Code Style Requirements
- Follow the existing project structure
- Write tests for new functionality
- Document using numpy-style docstrings
- Keep code complexity low
- Use type hints for all functions and methods
- Co-locate tests with implementation
- Favor declarative code styles over imperative code styles
- Use pydantic v2 (not @dataclass) for data objects with strong types
- Favor list and dictionary comprehensions over for loops

## Documentation

- Built with MkDocs and Material theme
- API documentation uses mkdocstrings
- Supports mermaid.js diagrams in markdown files:
  ```mermaid
  graph LR
      A[Doc] --> B[Feature]
  ```
- Build docs locally: `mkdocs serve`
- Build for production: `mkdocs build`
- Markdown files
    - Use `#` for top-level headings
    - Put blank lines above and below bulleted lists, numbered lists, headings, quotations, and code blocks
- Always keep the navigation tree in `mkdocs.yml` up to date with changes to the available documents in the `docs` folder

### API Documentation

API documentation uses mkdocstrings, which inserts module, class, and method documentation using certain markers in the markdown documents.

eg.

```
::: mojentic.llm.MessageBuilder
    options:
        show_root_heading: true
        merge_init_into_class: false
        group_by_category: false
```

Always use the same `show_root_heading`, `merge_init_into_class`, and `group_by_category` options. Adjust the module and class name after the `:::` as needed.

## Release Process

1. Update CHANGELOG.md:
   - The CHANGELOG.md format follows the well known "Keep a Changelog" format as described at https://keepachangelog.com/
   - All notable changes should be documented under the [Unreleased] section
   - Group changes into categories:
     - Added: New features
     - Changed: Changes in existing functionality
     - Deprecated: Soon-to-be removed features
     - Removed: Removed features
     - Fixed: Bug fixes
     - Security: Security vulnerability fixes
   - Each entry should be clear and understandable to end-users
   - Reference relevant issue/PR numbers where applicable

2. Creating a Release:
   - Ensure `pyproject.toml` has the next release version
   - Ensure all changes are documented in CHANGELOG.md
     - Move [Unreleased] changes to the new version section (e.g., [1.0.0])
   - Follow semantic versioning:
     - MAJOR version for incompatible API changes
     - MINOR version for backward-compatible new functionality
     - PATCH version for backward-compatible bug fixes

3. Best Practices:
   - Keep entries concise but descriptive
   - Write from the user's perspective
   - Include migration instructions for breaking changes
   - Document API changes thoroughly
   - Update documentation to reflect the changes

## Release Process

This project follows [Semantic Versioning](https://semver.org/) (SemVer) for version numbering. The version format is MAJOR.MINOR.PATCH, where:

1. MAJOR version increases for incompatible API changes
2. MINOR version increases for backward-compatible functionality additions
3. PATCH version increases for backward-compatible bug fixes

### Preparing a Release

When preparing a release, follow these steps:

1. **Update CHANGELOG.md**:
   - Move items from the "[Next]" section to a new version section
   - Add the new version number and release date: `## [x.y.z] - YYYY-MM-DD`
   - Ensure all changes are properly categorized under "Added", "Changed", "Deprecated", "Removed", "Fixed", or "Security"
   - Keep the empty "[Next]" section at the top for future changes

2. **Update Version Number**:
   - Update the version number in `pyproject.toml`
   - Ensure the version number follows semantic versioning principles based on the nature of changes:
     - **Major Release**: Breaking changes that require users to modify their code
     - **Minor Release**: New features that don't break backward compatibility
     - **Patch Release**: Bug fixes that don't add features or break compatibility

3. **Update Documentation**:
   - Review and update `README.md` to reflect any new features, changed behavior, or updated requirements
   - Update any other documentation files that reference features or behaviors that have changed
   - Ensure installation instructions and examples are up to date

4. **Final Verification**:
   - Run flake8 to ensure we have clean code (`flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics`)
   - Run all tests to ensure they pass (`pytest`)
   - Check that all documentation accurately reflects the current state of the project

### Release Types

#### Major Releases (x.0.0)

Major releases may include:

- Breaking API changes (eg tool plugin interfacing)
- Significant architectural changes
- Removal of deprecated features
- Changes that require users to modify their code or workflow

For major releases, consider:
- Providing migration guides
- Updating all documentation thoroughly
- Highlighting breaking changes prominently in the CHANGELOG

#### Minor Releases (0.x.0)

Minor releases may include:

- New features
- Non-breaking enhancements
- Deprecation notices (but not removal of deprecated features)
- Performance improvements

For minor releases:
- Document all new features
- Update README to highlight new capabilities
- Ensure backward compatibility

#### Patch Releases (0.0.x)

Patch releases should be limited to:

- Bug fixes
- Security updates
- Performance improvements that don't change behavior
- Documentation corrections

For patch releases:

- Clearly describe the issues fixed
- Avoid introducing new features
- Maintain strict backward compatibility
