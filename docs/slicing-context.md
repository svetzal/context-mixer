# Slicing Context with CRAFT-Aware Filtering

The `cmx slice` command allows you to organize and filter your knowledge chunks based on various criteria, making it easy to create focused context bundles for specific use cases, projects, or domains.

## What is Context Slicing?

Context slicing is the process of taking your comprehensive knowledge library and creating focused subsets based on specific criteria. This is particularly useful when you want to:

- Create domain-specific context bundles (e.g., only security-related knowledge)
- Filter by project boundaries to avoid cross-project contamination
- Select knowledge at specific granularity levels for different tasks
- Generate context bundles with specific authority levels

## Basic Usage

### Simple Slicing

```bash
# Slice all knowledge with basic granularity
cmx slice

# Slice with detailed granularity level
cmx slice --granularity detailed

# Slice with comprehensive granularity level
cmx slice --granularity comprehensive
```

### Domain-Specific Slicing

Filter your knowledge by specific domains:

```bash
# Focus on technical knowledge only
cmx slice --domains technical

# Include multiple domains
cmx slice --domains technical,business,security

# Combine domains with granularity
cmx slice --domains technical,operational --granularity detailed
```

### Project-Aware Slicing

Slice knowledge from specific projects to maintain context boundaries:

```bash
# Include knowledge from specific projects
cmx slice --project-ids "react-frontend,python-api"

# Exclude specific projects (useful for legacy or deprecated projects)
cmx slice --exclude-projects "legacy-system,experimental-features"

# Combine project filtering with other criteria
cmx slice --project-ids "frontend-app" --domains technical --granularity detailed
```

### Authority-Based Slicing

Filter knowledge based on authority levels to ensure quality:

```bash
# Include only official and verified knowledge
cmx slice --authority-level official

# Include verified and community knowledge
cmx slice --authority-level verified

# Include all knowledge including experimental
cmx slice --authority-level experimental
```

## Advanced Slicing Scenarios

### Creating Focused Context Bundles

```bash
# Create a security-focused bundle for production systems
cmx slice --domains security,operational \
          --authority-level official \
          --granularity detailed \
          --project-ids "production-api,frontend-app"

# Create a development-focused bundle with comprehensive details
cmx slice --domains technical,business \
          --granularity comprehensive \
          --exclude-projects "legacy-system"

# Create a quick reference bundle with basic information
cmx slice --granularity basic \
          --authority-level official \
          --domains technical
```

### Output Management

```bash
# Specify custom output location
cmx slice --output ./contexts/security-context.md \
          --domains security \
          --authority-level official

# Use different library path
cmx slice --library-path ./team-contexts \
          --domains business,operational \
          --granularity detailed
```

## Understanding Granularity Levels

### Basic Granularity
- High-level overviews and key principles
- Essential patterns and practices
- Quick reference information
- Minimal token usage

### Detailed Granularity
- Comprehensive explanations with examples
- Implementation details and best practices
- Common pitfalls and solutions
- Moderate token usage

### Comprehensive Granularity
- Complete documentation with full context
- Multiple examples and edge cases
- Historical context and rationale
- Maximum token usage

## Understanding Domains

Context Mixer organizes knowledge into several key domains:

- **Technical**: Code patterns, architecture, implementation details
- **Business**: Requirements, processes, domain logic
- **Security**: Security practices, compliance, threat models
- **Operational**: Deployment, monitoring, maintenance procedures
- **Design**: UI/UX patterns, design systems, user experience
- **Testing**: Testing strategies, patterns, quality assurance

## Understanding Authority Levels

Knowledge is classified by authority level to help you filter by quality and reliability:

- **Official**: Authoritative documentation, official standards
- **Verified**: Team-approved practices, validated patterns
- **Community**: Widely-accepted community practices
- **Experimental**: Emerging patterns, experimental approaches

## Best Practices

### 1. Start with Broad Slices
Begin with broader criteria and narrow down as needed:

```bash
# Start broad
cmx slice --domains technical

# Then narrow down
cmx slice --domains technical --project-ids "current-project" --granularity detailed
```

### 2. Use Authority Levels Appropriately
- Use `official` for production-critical contexts
- Use `verified` for team-standard practices
- Use `community` for general development guidance
- Use `experimental` when exploring new approaches

### 3. Combine with Assembly
Slicing works well in combination with the assemble command:

```bash
# First slice to create focused context
cmx slice --domains security --authority-level official --output security-context.md

# Then assemble for specific AI assistant
cmx assemble copilot --project-ids "production-api" --token-budget 4096
```

### 4. Project Boundary Management
Always consider project boundaries when slicing:

```bash
# Good: Include related projects
cmx slice --project-ids "frontend,backend,shared-lib"

# Good: Exclude unrelated projects
cmx slice --exclude-projects "legacy-system,deprecated-app"

# Avoid: Mixing unrelated project contexts without explicit intention
```

## Common Use Cases

### 1. Onboarding New Team Members
```bash
# Create comprehensive onboarding context
cmx slice --domains technical,business,operational \
          --granularity comprehensive \
          --authority-level verified \
          --project-ids "main-application"
```

### 2. Security Reviews
```bash
# Create security-focused context for reviews
cmx slice --domains security,operational \
          --authority-level official \
          --granularity detailed
```

### 3. Quick Reference Guides
```bash
# Create quick reference for common patterns
cmx slice --granularity basic \
          --authority-level official \
          --domains technical
```

### 4. Project-Specific Context
```bash
# Create context specific to current project
cmx slice --project-ids "current-project" \
          --domains technical,business \
          --granularity detailed
```

## Troubleshooting

### No Results Returned
If slicing returns no results, try:
- Broadening your criteria (remove some filters)
- Checking if your project IDs are correct
- Verifying that knowledge exists in the specified domains
- Lowering the authority level requirement

### Too Much Content
If slicing returns too much content, try:
- Narrowing your domain selection
- Increasing the authority level requirement
- Using more specific project filtering
- Reducing the granularity level

### Performance Issues
For large knowledge bases:
- Use more specific filtering criteria
- Consider slicing in smaller batches
- Use project filtering to reduce scope
- Monitor token usage with different granularity levels

## Integration with Other Commands

Slicing works seamlessly with other Context Mixer commands:

```bash
# Workflow: Ingest → Slice → Assemble
cmx ingest ./project --project-id "my-project"
cmx slice --project-ids "my-project" --domains technical --output technical-context.md
cmx assemble copilot --project-ids "my-project" --token-budget 8192
```

The slice command is a powerful tool for creating focused, relevant context bundles that help AI assistants provide better, more targeted assistance while respecting project boundaries and knowledge quality levels.