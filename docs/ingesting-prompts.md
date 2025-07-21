# Ingesting Prompts from Your Projects

One of Context Mixer's most powerful features is its ability to discover, analyze, and ingest existing context artifacts from your projects. This allows you to build a centralized knowledge library from the context instructions you've already created across different projects and AI assistants.

**NEW: Project Context Isolation** - Context Mixer now supports project-aware ingestion to prevent cross-project knowledge contamination. You can organize knowledge by project and choose which project contexts to include when assembling instructions for AI assistants.

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
- Detect and resolve conflicts with existing knowledge
- Quarantine chunks that cannot be automatically resolved

### 3. Ingest from Multiple Projects with Project Context

Build a comprehensive knowledge base by ingesting from multiple projects while maintaining project boundaries:

```bash
# Ingest from different types of projects with project identification
cmx ingest ~/projects/react-frontend --project-id "react-frontend" --project-name "React Frontend App"
cmx ingest ~/projects/python-api --project-id "python-api" --project-name "Python REST API"
cmx ingest ~/projects/mobile-app --project-id "mobile-app" --project-name "Mobile Application"
cmx ingest ~/work/enterprise-system --project-id "enterprise-system" --project-name "Enterprise System"
```

### 4. High-Performance Ingestion with HDBSCAN Clustering

**NEW: Intelligent Conflict Detection** - Context Mixer now uses HDBSCAN clustering to optimize conflict detection during ingestion, reducing processing time by up to 70% while maintaining accuracy.

```bash
# Default behavior - clustering optimization enabled automatically
cmx ingest ~/large-project --project-id "large-system" --project-name "Large Enterprise System"

# Fine-tune clustering behavior for optimal performance
cmx ingest ~/complex-project \
    --project-id "complex-app" \
    --project-name "Complex Application" \
    --min-cluster-size 5 \
    --batch-size 10

# Disable clustering for smaller projects (fallback to traditional O(n²) detection)
cmx ingest ~/small-project \
    --project-id "small-app" \
    --project-name "Small Application" \
    --no-clustering
```

**Why use project identification?**
- **Prevents cross-project contamination**: Knowledge from different projects won't be mixed inappropriately
- **Enables selective context assembly**: Choose which project contexts to include when generating AI assistant instructions
- **Maintains knowledge provenance**: Track which project each piece of knowledge came from
- **Supports project-scoped operations**: Filter and search knowledge by project boundaries

## Project Context Isolation

### Preventing Cross-Project Contamination

When working with multiple projects, it's crucial to maintain clear boundaries between different project contexts. Context Mixer's project-aware ingestion prevents knowledge from different projects from being mixed inappropriately.

```bash
# Ingest with project context to maintain boundaries
cmx ingest ./react-project --project-id "frontend-app" --project-name "React Frontend Application"
cmx ingest ./api-project --project-id "backend-api" --project-name "Python REST API"
```

### Project-Aware Context Assembly

When assembling context for AI assistants, you can now choose which project contexts to include:

```bash
# Include only specific projects
cmx assemble copilot --project-ids "frontend-app,backend-api"

# Exclude specific projects (useful for legacy or deprecated projects)
cmx assemble copilot --exclude-projects "legacy-system,deprecated-app"

# Combine both for precise control
cmx assemble copilot --project-ids "frontend-app,backend-api" --exclude-projects "experimental-features"
```

### Benefits of Project Context Isolation

- **Clean Context**: AI assistants receive only relevant project knowledge
- **Reduced Confusion**: Prevents conflicting guidance from different projects
- **Better Organization**: Knowledge is logically grouped by project boundaries
- **Selective Assembly**: Choose exactly which project contexts to include
- **Provenance Tracking**: Always know which project knowledge came from

## Conflict Detection and Quarantine

### Automatic Conflict Detection

During ingestion, Context Mixer automatically detects various types of conflicts:

- **Semantic Conflicts**: Knowledge that contradicts existing information
- **Authority Conflicts**: Lower authority knowledge conflicting with official guidance
- **Temporal Conflicts**: Current knowledge conflicting with deprecated information
- **Dependency Violations**: Knowledge chunks missing required dependencies
- **Validation Failures**: Knowledge that fails completeness validation

### Quarantine Process

When conflicts cannot be automatically resolved, Context Mixer quarantines the problematic knowledge chunks:

```bash
# Example ingestion with quarantine output
cmx ingest ./my-project --project-id "my-app"

# Output:
# ✓ Ingested 15 knowledge chunks successfully
# ⚠ Quarantined 2 chunks due to conflicts:
#   - semantic_conflict: React patterns conflict with existing guidance
#   - authority_conflict: Experimental API usage conflicts with official standards
# 
# Run 'cmx quarantine list' to review and resolve quarantined items
```

### Managing Quarantined Knowledge

After ingestion, review and resolve quarantined chunks:

```bash
# List all quarantined chunks
cmx quarantine list

# Review a specific chunk
cmx quarantine review <chunk-id>

# Resolve conflicts
cmx quarantine resolve <chunk-id> accept "Approved after review"
```

For detailed information about managing quarantined knowledge, see the [Managing Knowledge Quarantine](managing-quarantine.md) guide.

## Advanced Ingestion Options

### Current Implementation

The current version of Context Mixer supports the following ingestion parameters:

```bash
# Basic ingestion with clustering optimization (default)
cmx ingest /path/to/project

# Project-aware ingestion (recommended for multi-project setups)
cmx ingest /path/to/project --project-id "my-project" --project-name "My Project Name"

# Fine-tune clustering optimization
cmx ingest /path/to/project \
    --project-id "my-project" \
    --project-name "My Project Name" \
    --clustering \
    --min-cluster-size 3 \
    --clustering-fallback \
    --batch-size 5
```

### HDBSCAN Clustering Optimization

Context Mixer uses advanced HDBSCAN clustering to intelligently optimize conflict detection during ingestion:

#### Clustering Parameters

- `--clustering/--no-clustering`: Enable or disable clustering optimization (default: enabled)
- `--min-cluster-size INTEGER`: Minimum number of chunks required to form a cluster (default: 3)
- `--clustering-fallback/--no-clustering-fallback`: Fall back to traditional detection if clustering fails (default: enabled)
- `--batch-size INTEGER`: Number of conflict detections to process concurrently (default: 5)

#### Performance Benefits

**For Large Knowledge Bases (1000+ chunks):**
- ⚡ **70-80% reduction** in conflict detection time
- 🧠 **Intelligent grouping** of semantically related knowledge
- 📈 **Scalable performance** that improves with dataset size
- 🛡️ **Graceful fallback** to traditional detection when needed

**For Small Knowledge Bases (<100 chunks):**
- 🔄 **Automatic optimization** with minimal overhead  
- 🎯 **Consistent accuracy** maintained across all dataset sizes
- ⚙️ **Zero configuration** required for most use cases

#### When to Use Clustering Options

```bash
# Large enterprise projects (recommended for 500+ knowledge chunks)
cmx ingest ~/enterprise-system \
    --project-id "enterprise-system" \
    --min-cluster-size 5 \
    --batch-size 10

# Small to medium projects (default settings work well)
cmx ingest ~/small-project --project-id "small-app"

# Performance-critical ingestion (maximum optimization)
cmx ingest ~/massive-codebase \
    --project-id "massive-system" \
    --min-cluster-size 7 \
    --batch-size 15

# Development/testing environments (disable for predictable timing)
cmx ingest ~/test-project \
    --project-id "test-env" \
    --no-clustering
```

#### Clustering Process Overview

1. **Semantic Analysis**: HDBSCAN groups knowledge chunks by semantic similarity
2. **Intelligent Candidate Selection**: Only checks conflicts within and between related clusters
3. **Performance Optimization**: Reduces expensive LLM API calls by focusing on likely conflicts
4. **Quality Preservation**: Maintains the same conflict detection accuracy as traditional methods
5. **Automatic Fallback**: Switches to traditional O(n²) detection if clustering fails

### Future Capabilities

The following advanced options are planned for future releases:

- `--domain` - Organize content by domain (technical, business, design)
- `--tags` - Add searchable tags to ingested content
- `--authority` - Specify authority level (official, experimental, deprecated)
- `--filter` - Selective ingestion by content type
- `--verbose` - Detailed ingestion progress reporting
- `--interactive` - Manual conflict resolution

## Understanding the Ingestion Process

### 1. Discovery Phase

Context Mixer scans your project using intelligent heuristics:

```bash
cmx ingest ./project
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

Context Mixer automatically handles conflicts using CRAFT principles and authority levels. Manual conflict resolution features are planned for future releases.

The current system:
- Automatically detects semantic conflicts between knowledge chunks
- Preserves both conflicting pieces with appropriate metadata
- Uses authority levels and temporal information for resolution
- Maintains complete provenance tracking for all decisions

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
# Start with your most mature project using project identification
cmx ingest ~/projects/flagship-app --project-id "flagship-app" --project-name "Flagship Application"

# Then add experimental projects with clear identification
cmx ingest ~/experiments/new-framework --project-id "experimental-framework" --project-name "New Framework Experiment"
```

### 2. Use Consistent Project Naming

Develop a consistent project identification strategy:

```bash
# Use descriptive project IDs that reflect the project's purpose
cmx ingest ./frontend-app --project-id "frontend-react" --project-name "React Frontend Application"

# Include team or organization context in project names
cmx ingest ./api-service --project-id "backend-api" --project-name "Team Alpha - REST API Service"

# Use clear naming for different project types
cmx ingest ./mobile-app --project-id "mobile-ios" --project-name "iOS Mobile Application"
```

### 3. Regular Re-ingestion

Keep your knowledge library up-to-date by re-ingesting projects periodically:

```bash
# Update knowledge from evolving projects (maintain same project ID)
cmx ingest ./active-project --project-id "active-project" --project-name "Active Development Project"

# Re-ingest multiple projects with consistent identification
cmx ingest ~/projects/frontend --project-id "frontend-app" --project-name "Frontend Application"
cmx ingest ~/projects/backend --project-id "backend-api" --project-name "Backend API"
```

### 4. Validate After Ingestion

Review what was ingested to ensure quality using the assemble command:

```bash
# Test context assembly for specific projects
cmx assemble copilot --project-ids "frontend-app"

# Test cross-project context assembly
cmx assemble copilot --project-ids "frontend-app,backend-api"

# Verify project isolation by excluding specific projects
cmx assemble copilot --exclude-projects "experimental-framework"
```

## Troubleshooting Common Issues

### No Artifacts Found

If Context Mixer doesn't find any artifacts, check the following:

1. **Verify file types**: Context Mixer currently looks for `.md` and `.txt` files
2. **Check directory structure**: Ensure you're pointing to the correct project directory
3. **Review file locations**: Common locations include:
   - `.github/copilot-instructions.md`
   - `README.md`
   - `docs/` folder
   - `.cursorrules`
   - `.junie/guidelines.md`

```bash
# Basic troubleshooting - try ingesting with project context
cmx ingest ./project --project-id "my-project" --project-name "My Project"
```

### Poor Quality Extraction

If extracted content quality is not meeting expectations:

1. **Check source content**: Ensure your source files contain clear, well-structured information
2. **Use project identification**: This helps with better context organization
3. **Review chunk validation**: The system shows validation results during ingestion

```bash
# Ingest with clear project identification for better organization
cmx ingest ./project --project-id "clear-project-id" --project-name "Descriptive Project Name"
```

### Large Projects

For very large projects:

1. **Focus on key files**: Start with the most important documentation files
2. **Use project identification**: This helps organize large amounts of content
3. **Ingest incrementally**: Process different parts of the project separately

```bash
# Process different parts of a large project with clear identification
cmx ingest ./project/docs --project-id "large-project-docs" --project-name "Large Project - Documentation"
cmx ingest ./project/config --project-id "large-project-config" --project-name "Large Project - Configuration"
```

## Next Steps

Once you've ingested content from your projects, you can:

1. **Assemble project-aware context**: Learn how to [assemble copilot instructions](assembling-copilot-instructions.md) using the new project filtering capabilities
2. **Test project isolation**: Use `cmx assemble` with `--project-ids` and `--exclude-projects` to verify your project boundaries
3. **Organize by project**: Continue ingesting projects with consistent project identification
4. **Share with your team**: Set up Git-based collaboration for your knowledge library

The goal is to transform your scattered context artifacts into a well-organized, project-aware knowledge base that prevents cross-project contamination while enabling selective context assembly for AI assistants.
