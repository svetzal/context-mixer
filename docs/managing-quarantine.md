# Managing Knowledge Quarantine

The **Knowledge Quarantine System** is Context Mixer's implementation of the CRAFT framework's "Resist" principle. It provides a robust mechanism for isolating and managing knowledge chunks that cannot be automatically resolved due to conflicts, validation failures, or other issues.

## What is Knowledge Quarantine?

When Context Mixer ingests knowledge from multiple sources, it may encounter conflicts that cannot be automatically resolved:

- **Semantic Conflicts**: Two pieces of knowledge contradict each other
- **Authority Conflicts**: Lower authority knowledge conflicts with official guidance
- **Temporal Conflicts**: Current knowledge conflicts with deprecated information
- **Dependency Violations**: Knowledge chunks missing required dependencies
- **Validation Failures**: Knowledge chunks that fail completeness validation

Rather than rejecting this knowledge outright or allowing contamination, the quarantine system isolates these chunks for human review and resolution.

## How Quarantine Works

### Automatic Quarantine During Ingestion

When Context Mixer detects conflicts during ingestion, it automatically quarantines problematic chunks:

```bash
# During ingestion, conflicts are automatically detected and quarantined
cmx ingest ./my-project --project-id "my-app" --project-name "My Application"

# Output might show:
# ✓ Ingested 15 knowledge chunks
# ⚠ Quarantined 2 chunks due to conflicts
# Run 'cmx quarantine list' to review quarantined items
```

### Quarantine Reasons

The system categorizes quarantine reasons to help with resolution:

- **`semantic_conflict`**: Conflicts with existing knowledge semantically
- **`authority_conflict`**: Conflicts with higher authority knowledge
- **`temporal_conflict`**: Conflicts with current/deprecated status
- **`dependency_violation`**: Missing required dependencies
- **`validation_failure`**: Failed chunk completeness validation
- **`manual_quarantine`**: Manually quarantined by user

## Managing Quarantined Knowledge

### Listing Quarantined Chunks

View all quarantined knowledge chunks:

```bash
# List all quarantined chunks
cmx quarantine list

# Filter by specific criteria
cmx quarantine list --reason semantic_conflict
cmx quarantine list --priority 1
cmx quarantine list --resolved false
cmx quarantine list --project my-app
```

The list shows key information about each quarantined chunk:
- Unique ID (truncated for display)
- Quarantine reason
- Priority level (1=high, 5=low)
- Project context
- Age in days
- Resolution status
- Description

### Reviewing Individual Chunks

Get detailed information about a specific quarantined chunk:

```bash
cmx quarantine review <chunk-id>
```

This displays:
- Complete chunk content
- Detailed quarantine reason
- Conflicting chunks (if any)
- Project information
- Quarantine metadata (when, by whom, priority)
- Resolution status and details (if resolved)

### Resolving Quarantined Chunks

Resolve quarantined chunks using various resolution actions:

```bash
# Accept the chunk, overriding conflicts
cmx quarantine resolve <chunk-id> accept "Reviewed and approved as valid"

# Reject the chunk permanently
cmx quarantine resolve <chunk-id> reject "Outdated information, no longer relevant"

# Mark for merging with existing knowledge
cmx quarantine resolve <chunk-id> merge "Should be combined with existing patterns"

# Mark for modification before acceptance
cmx quarantine resolve <chunk-id> modify "Needs updates to resolve conflicts"

# Defer resolution to later time
cmx quarantine resolve <chunk-id> defer "Requires team discussion"

# Escalate to higher authority
cmx quarantine resolve <chunk-id> escalate "Needs architectural review"
```

#### Resolution Actions Explained

- **`accept`**: Accept the quarantined chunk, overriding any conflicts
- **`reject`**: Permanently reject the chunk and remove it from consideration
- **`merge`**: Indicate that the chunk should be merged with existing conflicting knowledge
- **`modify`**: Mark that the chunk needs modification to resolve conflicts
- **`defer`**: Defer resolution to a later time (useful for complex decisions)
- **`escalate`**: Escalate the decision to higher authority or team discussion

### Adding Resolution Context

Provide additional context when resolving quarantine:

```bash
cmx quarantine resolve <chunk-id> accept "Approved after team review" \
  --resolved-by "john.doe@company.com" \
  --notes "Conflicts were minor and acceptable for this project context"
```

## Quarantine Statistics and Monitoring

### System Statistics

Get an overview of the quarantine system status:

```bash
cmx quarantine stats
```

This shows:
- Total quarantined chunks
- Resolved vs. unresolved counts
- Breakdown by quarantine reason
- Priority distribution for unresolved items
- Age statistics (average and oldest)
- High-priority items requiring attention

### Maintenance Operations

Keep the quarantine system clean by removing resolved items:

```bash
# Clear all resolved quarantined chunks
cmx quarantine clear
```

This removes resolved chunks from the quarantine system while preserving their resolution records in the knowledge store.

## Best Practices

### Regular Review Workflow

1. **Daily Check**: Run `cmx quarantine stats` to monitor system health
2. **Weekly Review**: Address high-priority unresolved items
3. **Monthly Cleanup**: Clear resolved chunks to maintain system performance

### Resolution Guidelines

- **High Priority (1-2)**: Resolve within 24-48 hours
- **Medium Priority (3)**: Resolve within 1 week
- **Low Priority (4-5)**: Resolve within 1 month

### Team Collaboration

- Use `--resolved-by` to track who made resolution decisions
- Add detailed `--notes` to explain resolution reasoning
- Use `escalate` action for decisions requiring team input
- Use `defer` action when more information is needed

## Integration with CRAFT Principles

The quarantine system directly implements the **Resist** principle of CRAFT:

- **Prevents Knowledge Contamination**: Isolates conflicting information
- **Maintains Authority Hierarchies**: Respects official vs. experimental knowledge
- **Preserves Temporal Consistency**: Handles current vs. deprecated conflicts
- **Enables Human Oversight**: Provides mechanisms for expert review
- **Tracks Resolution History**: Maintains audit trail of decisions

## Troubleshooting

### Common Issues

**No quarantined chunks appear**: 
- Verify that conflict detection is enabled during ingestion
- Check that you're using the correct library path

**Cannot resolve quarantine**:
- Ensure the chunk ID is correct (use `cmx quarantine list` to verify)
- Check that the chunk hasn't already been resolved

**High number of quarantined items**:
- Review ingestion sources for quality issues
- Consider adjusting authority levels in source projects
- Use batch resolution for similar conflicts

### Getting Help

If you encounter issues with the quarantine system:

1. Check the quarantine statistics: `cmx quarantine stats`
2. Review specific problematic chunks: `cmx quarantine review <chunk-id>`
3. Consult the [CRAFT Framework documentation](craft-overview.md) for theoretical background
4. Review the [ingestion documentation](ingesting-prompts.md) for prevention strategies

## Next Steps

- Learn about [assembling context for AI assistants](assembling-copilot-instructions.md)
- Understand the [CRAFT framework principles](craft-overview.md)
- Explore [advanced ingestion techniques](ingesting-prompts.md)