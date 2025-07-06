# CRAFT Framework Overview

The **CRAFT framework** is the theoretical foundation that powers Context Mixer's intelligent knowledge management and context assembly capabilities. CRAFT provides a systematic approach to organizing, maintaining, and deploying context information for AI coding assistants.

## What is CRAFT?

CRAFT stands for:

- **C**hunk: Knowledge into small domain-coherent units
- **R**esist: Knowledge contamination and drift
- **A**dapt: Knowledge granularity for task context
- **F**it: Knowledge selection to task requirements
- **T**ranscend: Knowledge storage implementation

## The Knowledge Assembly Problem

Traditional approaches to AI assistant context suffer from several critical failures:

- **Domain Bleed**: Including Python debugging knowledge when working on Java code
- **Temporal Mismatch**: Using outdated architectural decisions that have been superseded
- **Scope Creep**: Loading enterprise patterns when building a simple prototype
- **Authority Confusion**: Mixing authoritative documentation with experimental notes
- **Granularity Mismatch**: Providing high-level strategy when detailed implementation guidance is needed

CRAFT addresses these issues by treating knowledge as a multi-dimensional, versionable, and contextually-bounded resource.

## The Five CRAFT Principles

### 🧩 Chunk: Domain-Coherent Knowledge Units

Knowledge should be **atomically scoped** to prevent one domain's assumptions from polluting another. Each knowledge chunk represents a single, coherent concept within a specific domain and authority level.

**Example**: Instead of mixing React patterns, accessibility guidelines, and backend API patterns in one document, separate them into distinct chunks:

- `technical/frontend/react-patterns.md`
- `design/accessibility/wcag-guidelines.md`
- `technical/backend/java-api-patterns.md`

### 🛡️ Resist: Knowledge Contamination and Drift

Implement safeguards against knowledge degradation through:

- **Version control** for all knowledge artifacts
- **Authority levels** to distinguish official vs. experimental knowledge
- **Validation rules** to prevent conflicting information
- **Knowledge quarantine** to isolate conflicts that cannot be automatically resolved
- **Regular audits** to identify outdated content

Context Mixer's **Knowledge Quarantine System** is the primary implementation of the Resist principle. When conflicts are detected during ingestion that cannot be automatically resolved, the system quarantines problematic knowledge chunks for human review. This prevents knowledge contamination while preserving potentially valuable information for expert evaluation.

Learn more about managing quarantined knowledge in the [Managing Knowledge Quarantine](managing-quarantine.md) guide.

### 🔄 Adapt: Context-Aware Granularity

Knowledge granularity should match the task context:

- **High-level strategy** for architectural decisions
- **Detailed patterns** for implementation guidance
- **Specific examples** for debugging scenarios
- **Quick references** for syntax lookups

### 🎯 Fit: Intelligent Knowledge Selection

Select the right knowledge for the specific task and constraints:

- **Token optimization** to fit within AI assistant limits
- **Relevance scoring** based on project context
- **Dependency resolution** to include related knowledge
- **Conflict detection** to avoid contradictory guidance

### 🚀 Transcend: Implementation Independence

Abstract knowledge management from specific storage or retrieval mechanisms:

- **Multiple storage backends** (file system, databases, cloud)
- **Flexible retrieval strategies** (semantic search, tag-based, hierarchical)
- **Pluggable processing** for different AI assistants
- **Format independence** (Markdown, JSON, YAML)

## CRAFT in Practice

### Knowledge Organization

CRAFT organizes knowledge across multiple dimensions:

```
knowledge-library/
├── domains/
│   ├── business/
│   │   ├── user-research/
│   │   └── compliance/
│   ├── technical/
│   │   ├── frontend/
│   │   └── backend/
│   └── design/
│       ├── visual/
│       └── interaction/
├── authority/
│   ├── official/
│   ├── experimental/
│   └── deprecated/
└── scope/
    ├── enterprise/
    ├── prototype/
    └── mobile-only/
```

### Context Assembly Pipeline

1. **Query Analysis**: Understand what knowledge is needed
2. **Chunk Retrieval**: Find relevant knowledge chunks
3. **Conflict Resolution**: Handle contradictory information
4. **Granularity Adaptation**: Adjust detail level for context
5. **Token Optimization**: Fit within assistant constraints
6. **Assembly**: Combine chunks into coherent context

### Metadata-Driven Intelligence

Each knowledge chunk includes rich metadata:

```yaml
metadata:
  domain: "technical/frontend"
  authority: "official"
  granularity: "detailed"
  scope: ["web", "mobile"]
  dependencies: ["technical/backend/api-patterns"]
  conflicts: ["technical/frontend/legacy-patterns"]
  last_validated: "2024-01-15"
```

## Benefits of CRAFT

### For Individual Developers

- **Consistency**: Maintain the same high standards across all projects
- **Efficiency**: Reuse proven patterns without manual copying
- **Learning**: Build a personal knowledge base that grows over time

### For Teams

- **Standardization**: Ensure all team members follow the same practices
- **Knowledge Sharing**: Capture and distribute expertise across the team
- **Onboarding**: Help new team members quickly understand established patterns

### For Organizations

- **Scalability**: Manage context practices across multiple teams and projects
- **Compliance**: Ensure consistent adherence to security and regulatory requirements
- **Innovation**: Experiment with new approaches while maintaining stability

## Getting Started with CRAFT

Context Mixer implements CRAFT principles automatically, but understanding the framework helps you:

1. **Organize your knowledge** more effectively
2. **Tag and categorize** context fragments appropriately
3. **Resolve conflicts** when ingesting from multiple sources
4. **Optimize context assembly** for different scenarios

Ready to see CRAFT in action? Learn how to [ingest prompts from your projects](ingesting-prompts.md) or [assemble copilot instructions](assembling-copilot-instructions.md) using Context Mixer.

## Further Reading

For a deep dive into CRAFT theory and implementation details, see the complete [CRAFT Theory document](https://github.com/svetzal/context-mixer/blob/main/THEORY.md) in the Context Mixer repository.
