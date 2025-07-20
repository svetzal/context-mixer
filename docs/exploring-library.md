# Exploring Your Context Library

The `cmx open` command provides a convenient way to explore and edit your context library directly in your preferred editor. This is particularly useful for manual review, editing, and understanding the structure of your knowledge chunks.

## Basic Usage

```bash
# Open the context library in your default editor
cmx open
```

This command will:
- Open your context library directory in the configured editor
- Display a confirmation message showing which editor is being used
- Allow you to browse and edit your knowledge chunks directly

## Editor Configuration

### Default Behavior

By default, Context Mixer uses Visual Studio Code (`code`) as the editor. If VS Code is installed and available in your PATH, the command will work immediately.

### Configuring Your Preferred Editor

You can configure your preferred editor using the `EDITOR` environment variable:

```bash
# Use VS Code (default)
export EDITOR="code -w"
cmx open

# Use Vim
export EDITOR="vim"
cmx open

# Use Emacs
export EDITOR="emacs"
cmx open

# Use any other editor
export EDITOR="your-favorite-editor"
cmx open
```

### Persistent Configuration

To make your editor choice persistent, add the export command to your shell configuration file:

```bash
# For Bash users (~/.bashrc or ~/.bash_profile)
echo 'export EDITOR="code"' >> ~/.bashrc

# For Zsh users (~/.zshrc)
echo 'export EDITOR="code"' >> ~/.zshrc

# For Fish users (~/.config/fish/config.fish)
echo 'set -gx EDITOR code' >> ~/.config/fish/config.fish
```

## What You'll See

When you open your context library, you'll find a structured directory containing:

### Directory Structure
```
your-context-library/
├── .git/                    # Git repository for version control
├── knowledge/               # Your knowledge chunks organized by project
│   ├── project-1/
│   │   ├── chunk-001.md
│   │   ├── chunk-002.md
│   │   └── metadata.json
│   └── project-2/
│       ├── chunk-003.md
│       └── metadata.json
├── quarantine/              # Quarantined chunks awaiting resolution
│   ├── conflict-001.md
│   └── conflict-002.md
├── config.json              # Library configuration
└── index.json               # Knowledge index and metadata
```

### Knowledge Chunks

Each knowledge chunk is stored as a Markdown file with:
- **Content**: The actual knowledge or context information
- **Metadata**: YAML frontmatter with chunk metadata
- **Provenance**: Information about the source and project

Example chunk structure:
```markdown
---
id: "chunk-001"
domains: ["technical", "security"]
authority: "official"
granularity: "detailed"
project_id: "react-frontend"
project_name: "React Frontend App"
created_at: "2024-01-15T10:30:00Z"
---

# Authentication Best Practices

When implementing authentication in React applications...
```

## Common Use Cases

### 1. Manual Review and Editing

```bash
# Open library to review and edit chunks
cmx open

# Then in your editor:
# - Review quarantined chunks
# - Edit chunk content
# - Update metadata
# - Organize knowledge structure
```

### 2. Understanding Library Structure

Use the open command to:
- Explore how your knowledge is organized
- Understand the CRAFT framework implementation
- Review project boundaries and isolation
- Examine metadata and provenance information

### 3. Bulk Operations

```bash
# Open for bulk editing
cmx open

# Then use your editor's features for:
# - Find and replace across multiple chunks
# - Bulk metadata updates
# - Reorganizing chunk structure
# - Adding tags or domains
```

### 4. Integration with Git Workflows

```bash
# Open library for Git operations
cmx open

# Then use Git integration in your editor:
# - Review changes before committing
# - Create branches for experimental knowledge
# - Merge knowledge from different sources
# - Resolve merge conflicts
```

## Editor-Specific Tips

### Visual Studio Code

VS Code provides excellent support for Context Mixer libraries:

```bash
# Install useful extensions
code --install-extension ms-vscode.vscode-json
code --install-extension yzhang.markdown-all-in-one
code --install-extension davidanson.vscode-markdownlint
```

Useful VS Code features:
- **Markdown preview**: View formatted chunks
- **JSON schema validation**: Validate metadata files
- **Git integration**: Built-in Git support
- **Search and replace**: Powerful find/replace across files
- **File explorer**: Easy navigation of library structure

### Vim/Neovim

For Vim users, consider these plugins:
- `vim-markdown` for Markdown syntax highlighting
- `fzf.vim` for fuzzy file finding
- `vim-fugitive` for Git integration
- `coc.nvim` for JSON schema validation

### Sublime Text

Useful Sublime Text packages:
- `MarkdownEditing` for enhanced Markdown support
- `GitSavvy` for Git integration
- `SublimeLinter` for file validation
- `Goto Anything` for quick navigation

## Best Practices

### 1. Use Version Control

Always commit your changes when working directly with the library:

```bash
# After making changes in your editor
cd your-context-library
git add .
git commit -m "Manual updates to knowledge chunks"
```

### 2. Validate Changes

After manual editing, validate your changes:

```bash
# Check for any issues after manual editing
cmx quarantine list
cmx quarantine stats
```

### 3. Backup Before Major Changes

```bash
# Create a backup branch before major manual changes
cd your-context-library
git checkout -b backup-before-manual-edit
git checkout main
# Make your changes
```

### 4. Use Editor Features

Take advantage of your editor's features:
- **Syntax highlighting** for Markdown and JSON
- **Linting** to catch formatting issues
- **Search and replace** for bulk updates
- **Git integration** for change tracking

## Troubleshooting

### Editor Not Found

If you get an error that the editor is not found:

```bash
# Check if your editor is in PATH
which code  # or vim, emacs, etc.

# If not found, install or add to PATH
# For VS Code on macOS:
# Install VS Code, then run: Shell Command: Install 'code' command in PATH

# For other editors, ensure they're properly installed and in PATH
```

### Permission Issues

If you encounter permission issues:

```bash
# Check library permissions
ls -la ~/.context-mixer

# Fix permissions if needed
chmod -R u+rw ~/.context-mixer
```

### Library Not Found

If the library path is not found:

```bash
# Initialize library if it doesn't exist
cmx init

# Or specify custom library path
cmx open --library-path /path/to/your/library
```

## Integration with Other Commands

The open command works well with other Context Mixer commands:

```bash
# Typical workflow
cmx ingest ./project --project-id "my-project"  # Ingest knowledge
cmx quarantine list                              # Check for conflicts
cmx open                                         # Review and edit manually
cmx quarantine resolve <id> accept "Reviewed"   # Resolve conflicts
cmx assemble copilot --project-ids "my-project" # Generate context
```

The open command provides direct access to your knowledge library, enabling manual curation, review, and editing that complements the automated features of Context Mixer.