# Assembling Copilot Instructions

After building your knowledge library by [ingesting prompts from your projects](ingesting-prompts.md), Context Mixer's most powerful feature is its ability to intelligently assemble context-specific instructions for AI coding assistants. This guide shows you how to create tailored copilot-instructions files for your projects.

## Understanding Context Assembly

Context assembly is where the CRAFT framework really shines. Instead of manually copying and pasting context fragments, Context Mixer:

1. **Analyzes your project** to understand its domain, technology stack, and requirements
2. **Selects relevant knowledge** from your library based on the project context
3. **Resolves conflicts** between different approaches or patterns
4. **Optimizes for token limits** while maximizing relevance
5. **Assembles coherent instructions** tailored to your specific project

## Semantic Deduplication

**NEW**: Context Mixer now automatically removes redundant content during assembly using semantic deduplication. This powerful feature:

- **Detects semantically similar chunks** across different projects and sources
- **Eliminates duplicate guidance** that would otherwise clutter your assembled instructions
- **Selects the highest authority version** when multiple similar chunks exist
- **Maintains content quality** by preserving the most comprehensive and authoritative guidance

### How Semantic Deduplication Works

When you use domain-based filtering (e.g., `--filter-tags "domain:python"`), Context Mixer automatically:

1. **Analyzes semantic similarity** between all selected chunks using vector embeddings
2. **Groups similar content** together (e.g., multiple chunks about Python type hints)
3. **Selects the best chunk** from each group based on:
   - **Authority level**: FOUNDATIONAL > OFFICIAL > CONVENTIONAL > EXPERIMENTAL > DEPRECATED
   - **Content comprehensiveness**: Longer, more detailed guidance is preferred
   - **Recency**: Newer content is preferred when authority and length are equal

### Benefits for Multi-Project Workflows

This is especially valuable when working with multiple projects that share common practices:

```bash
# Before: Would include duplicate Python practices from multiple projects
cmx assemble copilot --filter "domain:python,domain:testing"

# Now: Automatically deduplicates similar content, keeping only the best version
# Result: Clean, non-redundant instructions with the highest quality guidance
```

**Example**: If you have Python type hint guidance in three different projects with different authority levels (CONVENTIONAL, OFFICIAL, EXPERIMENTAL), semantic deduplication will automatically select and include only the OFFICIAL version in your assembled instructions.

## Basic Assembly Workflow

### 1. Assemble Instructions for GitHub Copilot

Generate a copilot-instructions file for your project:

```bash
cmx assemble copilot
```

This creates `.github/copilot-instructions.md` with context tailored to your project, automatically selecting relevant knowledge from your library based on the project context.

### 2. Assemble for Other AI Assistants

Context Mixer supports multiple AI assistant formats:

```bash
# For Cursor
cmx assemble cursor

# For Claude
cmx assemble claude

# For Junie
cmx assemble junie

# For multiple targets at once (not currently supported - run separately)
cmx assemble copilot
cmx assemble cursor
cmx assemble claude
```

## Advanced Assembly Options

### Project-Aware Assembly

Control which project contexts are included to prevent cross-project contamination:

```bash
# Include only specific projects
cmx assemble copilot --project-ids "react-frontend,python-api"

# Exclude specific projects (useful for legacy or deprecated projects)
cmx assemble copilot --exclude-projects "legacy-system,deprecated-app"

# Combine both for precise control
cmx assemble copilot --project-ids "frontend-app,backend-api" --exclude-projects "experimental-features"
```

### Selective Knowledge Assembly

Choose which knowledge domains to include using tag-based filtering:

```bash
# Only include frontend patterns
cmx assemble copilot --filter "domain:frontend"

# Include multiple domains
cmx assemble copilot --filter "domain:frontend,domain:backend,domain:testing"

# Filter by technology and layer
cmx assemble copilot --filter "tech:react,layer:testing"
```

### Token Optimization

Optimize for different context limits and quality thresholds:

```bash
# Optimize for GitHub Copilot's default limits
cmx assemble copilot --token-budget 8192

# Create a condensed version for smaller limits
cmx assemble copilot --token-budget 4096 --quality-threshold 0.9

# Create an extended version for larger contexts
cmx assemble copilot --token-budget 16384 --quality-threshold 0.7
```

## Understanding Assembly Output

### Standard Assembly Structure

A typical assembled copilot-instructions file includes:

```markdown
# Project Context for AI Assistant

## Project Overview
[Generated project description based on analysis]

## Technology Stack
[Relevant technologies and frameworks]

## Development Patterns
[Coding patterns and conventions from your knowledge library]

## Architecture Guidelines
[Architectural decisions and patterns]

## Testing Approach
[Testing patterns and practices]

## Code Style and Standards
[Formatting, naming, and style guidelines]

## Common Patterns and Examples
[Specific code examples and patterns]

## Project-Specific Considerations
[Unique requirements or constraints]
```

### Metadata and Traceability

Each assembled file includes metadata for traceability:

```markdown
<!-- Generated by Context Mixer CRAFT system -->
<!-- Source chunks: 12 -->
<!-- Domains: frontend, testing, architecture -->
<!-- Authority: official, experimental -->
<!-- Generated: 2024-01-15T10:30:00Z -->
```

## Assembly Strategies

### For New Projects

When starting a new project, assemble comprehensive instructions with appropriate filtering:

```bash
# Create comprehensive instructions for a new React project
cmx assemble copilot \
  --filter "tech:react,tech:typescript,domain:frontend,domain:testing" \
  --token-budget 8192 \
  --quality-threshold 0.8
```

### For Existing Projects

When adding Context Mixer to existing projects, focus on relevant project contexts:

```bash
# Assemble instructions for a specific project context
cmx assemble copilot \
  --project-ids "my-existing-project" \
  --filter "domain:backend,domain:testing" \
  --token-budget 6144
```

### For Team Standardization

Create consistent instructions across team projects by including multiple related projects:

```bash
# Use knowledge from multiple team projects
cmx assemble copilot \
  --project-ids "frontend-app,backend-api,shared-components" \
  --filter "domain:frontend,domain:backend" \
  --token-budget 10240
```

### For Specialized Contexts

Tailor instructions for specific development contexts using domain filtering:

```bash
# For testing-focused development
cmx assemble copilot --filter "domain:testing,layer:unit,layer:integration"

# For performance optimization work
cmx assemble copilot --filter "domain:performance,domain:optimization"

# For security-focused development
cmx assemble copilot --filter "domain:security,domain:compliance"
```

## Best Practices

### Regular Updates

Keep your instructions current as your knowledge library evolves:

```bash
# Re-assemble instructions after ingesting new knowledge
cmx assemble copilot --project-ids "my-project" --token-budget 8192

# Use quarantine system to review conflicts from new ingestions
cmx quarantine list
cmx quarantine resolve <chunk-id> accept "Reviewed and approved"
```

### Combining Multiple Projects

When working on projects that share common patterns:

```bash
# Include knowledge from related projects
cmx assemble copilot \
  --project-ids "frontend-app,backend-api" \
  --exclude-projects "legacy-system" \
  --filter "domain:shared,domain:common"
```

## Multi-Project Workflows

### Consistent Instructions Across Projects

Maintain consistency across multiple related projects using project filtering:

```bash
# Include knowledge from multiple related projects
cmx assemble copilot \
  --project-ids "frontend-app,backend-api,shared-components" \
  --filter "domain:shared,domain:common" \
  --token-budget 8192

# Exclude legacy projects while including current ones
cmx assemble copilot \
  --project-ids "current-frontend,current-backend" \
  --exclude-projects "legacy-system,deprecated-app"
```

### Project-Specific Customizations

While maintaining consistency, focus on specific project contexts:

```bash
# Focus on specific project with targeted filtering
cmx assemble copilot \
  --project-ids "my-graphql-project" \
  --filter "tech:graphql,domain:api" \
  --token-budget 6144
```

## Quality Assurance

### Validation and Testing

Ensure your assembled instructions are high quality by using the quarantine system:

```bash
# Review any quarantined knowledge conflicts
cmx quarantine list --priority 1

# Resolve conflicts to improve knowledge quality
cmx quarantine resolve <chunk-id> accept "Reviewed and approved"

# Check quarantine statistics
cmx quarantine stats
```

### Iterative Improvement

Improve your knowledge library based on real usage:

```bash
# Re-assemble after resolving quarantine issues
cmx assemble copilot --project-ids "my-project" --token-budget 8192

# Ingest new knowledge and review conflicts
cmx ingest ./new-project --project-id "new-project"
cmx quarantine list
```

## Troubleshooting

### Common Assembly Issues

**Problem**: Token limit exceeded
```bash
# Solution: Reduce token budget or increase quality threshold
cmx assemble copilot --token-budget 6144 --quality-threshold 0.9
```

**Problem**: Irrelevant content included
```bash
# Solution: Use more specific filtering
cmx assemble copilot --filter "domain:frontend,tech:react" --project-ids "my-react-app"
```

**Problem**: Missing important knowledge
```bash
# Solution: Check if knowledge is quarantined
cmx quarantine list
# Or broaden the filtering criteria
cmx assemble copilot --project-ids "project1,project2" --quality-threshold 0.7
```

## Next Steps

With your copilot instructions assembled, you can:

1. **Test the instructions** with your AI assistant to ensure they work well
2. **Iterate and improve** based on real usage feedback
3. **Share with your team** to standardize practices
4. **Expand your knowledge library** by ingesting more projects
5. **Automate the process** with CI/CD integration for consistent updates

The goal is to create a living system where your context instructions continuously improve and adapt to your evolving development practices.
