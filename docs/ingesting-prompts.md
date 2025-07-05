# Ingesting Prompts from Your Projects

One of Context Mixer's most powerful features is its ability to discover, analyze, and ingest existing context artifacts from your projects. This allows you to build a centralized knowledge library from the context instructions you've already created across different projects and AI assistants.

## What Can Context Mixer Ingest?

Context Mixer can automatically discover and process various types of context artifacts:

### AI Assistant Main Configuration Files
- **GitHub Copilot**: `.github/copilot-instructions.md`
- **Claude**: `CLAUDE.md`
- **Junie**: `.junie/guidelines.md`

### Documentation Files
- `README.md` files with development guidelines
- `CONTRIBUTING.md` files with team practices
- `docs/` folders with project-specific documentation
- Architecture decision records (ADRs)

### Configuration and Standards
- Code style configurations (`.eslintrc`, `prettier.config.js`)
- Testing patterns and examples
- Deployment and CI/CD documentation
- API documentation and examples

## Basic Ingestion Workflow

### 1. Initialize Your Context Library

First, create a dedicated directory for your context library:

```bash
mkdir my-context-library
cd my-context-library
cmx init
```

This creates the basic structure for your CRAFT-organized knowledge library.

### 2. Ingest from a Single Project

Point Context Mixer at any project directory:

```bash
cmx ingest /path/to/your/project
```

Context Mixer will:
- Scan the project for known context artifacts
- Analyze the content using CRAFT principles
- Extract domain-specific knowledge chunks
- Organize them in your context library
- Resolve conflicts with existing knowledge

### 3. Ingest from Multiple Projects

Build a comprehensive knowledge base by ingesting from multiple projects:

```bash
# Ingest from different types of projects
cmx ingest ~/projects/react-frontend
cmx ingest ~/projects/python-api
cmx ingest ~/projects/mobile-app
cmx ingest ~/work/enterprise-system
```

## Advanced Ingestion Options

### Selective Ingestion

Use filters to ingest only specific types of content:

```bash
# Only ingest AI assistant configurations
cmx ingest ./project --filter ai-configs

# Only ingest documentation
cmx ingest ./project --filter docs

# Only ingest from specific file patterns
cmx ingest ./project --include "*.md" --exclude "node_modules/"
```

### Domain-Specific Ingestion

Organize ingested content by domain:

```bash
# Tag content by technology stack
cmx ingest ./react-project --domain frontend --tags react,typescript

# Tag content by project type
cmx ingest ./api-project --domain backend --tags python,fastapi,enterprise

# Tag content by team or organization
cmx ingest ./project --domain mobile --tags ios,team-alpha
```

### Authority Level Assignment

Specify the authority level of ingested content:

```bash
# Mark as official team standards
cmx ingest ./project --authority official

# Mark as experimental patterns
cmx ingest ./prototype --authority experimental

# Mark as deprecated (for migration tracking)
cmx ingest ./legacy-project --authority deprecated
```

## Understanding the Ingestion Process

### 1. Discovery Phase

Context Mixer scans your project using intelligent heuristics:

```bash
cmx ingest ./project --verbose
```

Output example:
```
🔍 Discovering context artifacts...
  ✓ Found .github/copilot-instructions.md (GitHub Copilot)
  ✓ Found .cursorrules (Cursor)
  ✓ Found README.md with development guidelines
  ✓ Found docs/api-patterns.md
  ✓ Found tests/testing-guidelines.md
  
📊 Analysis complete: 5 artifacts found
```

### 2. Analysis Phase

Each artifact is analyzed for domain classification and chunking:

```bash
🧠 Analyzing content with CRAFT principles...
  ✓ Extracting frontend patterns from copilot-instructions.md
  ✓ Identifying testing practices from testing-guidelines.md
  ✓ Parsing API documentation patterns
  ✓ Detecting code style preferences
  
🏷️  Applying metadata tags...
```

### 3. Integration Phase

New knowledge is integrated with your existing library:

```bash
🔄 Integrating with knowledge library...
  ✓ Added 12 new knowledge chunks
  ⚠️  Resolved 2 conflicts with existing patterns
  ✓ Updated 3 existing chunks with new information
  
📚 Knowledge library updated: 45 total chunks
```

## Handling Conflicts and Duplicates

### Automatic Conflict Detection

Context Mixer automatically detects when ingested content conflicts with existing knowledge:

```bash
⚠️  Conflict detected:
  Existing: technical/frontend/react-patterns.md
    "Use functional components with hooks"
  
  New: project-a/copilot-instructions.md
    "Prefer class components for complex state"
  
  Resolution: Keep both with authority tags
  - Existing marked as 'official'
  - New marked as 'project-specific'
```

### Manual Conflict Resolution

Review and resolve conflicts interactively:

```bash
cmx ingest ./project --interactive
```

This allows you to:
- Choose which version to keep
- Merge conflicting information
- Tag variants for different contexts
- Mark outdated information as deprecated

### Deduplication

Context Mixer identifies and handles duplicate content:

```bash
🔍 Duplicate content detected:
  Source 1: project-a/README.md
  Source 2: project-b/docs/setup.md
  
  Action: Merged into single chunk with multiple source references
```

## Best Practices for Ingestion

### 1. Start with Your Best Projects

Begin ingestion with projects that have well-documented, proven practices:

```bash
# Start with your most mature project
cmx ingest ~/projects/flagship-app --authority official

# Then add experimental projects
cmx ingest ~/experiments/new-framework --authority experimental
```

### 2. Use Consistent Tagging

Develop a consistent tagging strategy across projects:

```bash
# Technology tags
cmx ingest ./project --tags react,typescript,frontend

# Team/organization tags  
cmx ingest ./project --tags team-alpha,enterprise

# Project type tags
cmx ingest ./project --tags api,microservice,production
```

### 3. Regular Re-ingestion

Keep your knowledge library up-to-date by re-ingesting projects periodically:

```bash
# Update knowledge from evolving projects
cmx ingest ./active-project --update

# Batch update multiple projects
cmx batch-ingest ~/projects/* --update
```

### 4. Validate After Ingestion

Review what was ingested to ensure quality:

```bash
# List recently ingested chunks
cmx list --recent --limit 10

# Search for specific patterns
cmx search "react patterns" --domain frontend

# Validate chunk quality
cmx validate --domain frontend --authority official
```

## Troubleshooting Common Issues

### No Artifacts Found

If Context Mixer doesn't find any artifacts:

```bash
# Check what files are being scanned
cmx ingest ./project --dry-run --verbose

# Manually specify file patterns
cmx ingest ./project --include "**/*.md" --include "**/.*rules"

# Check for non-standard locations
cmx ingest ./project --scan-all
```

### Poor Quality Extraction

If extracted content is low quality:

```bash
# Use higher quality LLM for analysis
cmx ingest ./project --model gpt-4

# Provide additional context
cmx ingest ./project --context "This is a React TypeScript project using Material-UI"

# Manual review mode
cmx ingest ./project --review-chunks
```

### Large Projects

For very large projects, optimize the ingestion process:

```bash
# Limit file size
cmx ingest ./project --max-file-size 100KB

# Skip certain directories
cmx ingest ./project --exclude "node_modules/" --exclude "dist/"

# Process in batches
cmx ingest ./project --batch-size 10
```

## Next Steps

Once you've ingested content from your projects, you can:

1. **Explore your knowledge library**: Use `cmx list` and `cmx search` to browse what you've collected
2. **Assemble context for new projects**: Learn how to [assemble copilot instructions](assembling-copilot-instructions.md)
3. **Refine and organize**: Use `cmx tag`, `cmx merge`, and `cmx split` to improve organization
4. **Share with your team**: Set up Git-based collaboration for your knowledge library

The goal is to transform your scattered context artifacts into a well-organized, searchable, and reusable knowledge base that grows more valuable over time.