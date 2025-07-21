# Installation

Context Mixer is a Python command-line tool that can be installed using several methods. We recommend using `pipx` for the best experience.

## Prerequisites

- **Python 3.12 or higher**
- **Git** (for version control of your context library)
- **Access to LLM services**: Either Ollama (local) or OpenAI (remote)

## Recommended Installation: pipx

[pipx](https://pypa.github.io/pipx/) is the recommended way to install Context Mixer as it creates an isolated environment for the tool while making the `cmx` command globally available.

### Install pipx (if not already installed)

```bash
# On macOS with Homebrew
brew install pipx

# On Ubuntu/Debian
sudo apt update
sudo apt install pipx

# On other systems with pip
python -m pip install --user pipx
python -m pipx ensurepath
```

### Install Context Mixer

```bash
pipx install context-mixer
```

### Verify Installation

```bash
cmx --version
```

## Alternative Installation Methods

### Using pip

If you prefer to use pip directly:

```bash
pip install context-mixer
```

**Note**: This installs Context Mixer in your current Python environment. Consider using a virtual environment to avoid conflicts.

### Using pip with virtual environment

```bash
python -m venv cmx-env
source cmx-env/bin/activate  # On Windows: cmx-env\Scripts\activate
pip install context-mixer
```

## Development Installation

If you want to contribute to Context Mixer or run the latest development version:

```bash
# Clone the repository
git clone https://github.com/svetzal/context-mixer.git
cd context-mixer

# Install in development mode
pip install -e ".[dev]"
```

This installs Context Mixer with all development dependencies including testing and documentation tools.

## LLM Service Setup

Context Mixer requires access to Large Language Model services for intelligent context processing. You can use either:

### Option 1: Ollama (Local, Free)

Install Ollama for local LLM processing:

```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Then pull a model
ollama pull llama2
```

### Option 2: OpenAI (Remote, Paid)

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

## Verify Your Installation

Test that everything is working correctly:

```bash
# Check version
cmx --version

# Get help
cmx --help

# Initialize a test context library
mkdir test-context-library
cd test-context-library
cmx init
```

If the initialization completes successfully, you're ready to start using Context Mixer!

## Troubleshooting

### Command not found: cmx

If you get a "command not found" error after installation:

1. **With pipx**: Run `pipx ensurepath` and restart your terminal
2. **With pip**: Make sure your Python scripts directory is in your PATH
3. **Check installation**: Run `pip list | grep context-mixer` to verify installation

### Python version issues

Context Mixer requires Python 3.12+. Check your Python version:

```bash
python --version
```

If you have an older version, consider using [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions.

### LLM service connection issues

- **Ollama**: Ensure the Ollama service is running (`ollama serve`)
- **OpenAI**: Verify your API key is set correctly and has sufficient credits

## Next Steps

Now that Context Mixer is installed:

1. **Learn the Concepts**: Understand the [CRAFT Framework](craft-overview.md) that powers intelligent context management
2. **Start Ingesting**: Begin [ingesting prompts](ingesting-prompts.md) from your existing projects
3. **Optimize Performance**: For large knowledge bases, explore [Performance Optimization](performance-optimization.md) with HDBSCAN clustering to achieve **70%+ speed improvements**