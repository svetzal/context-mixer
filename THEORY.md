# CRAFT Theory: Knowledge Management for Intelligent Context Assembly

- **C**hunk: Knowledge into small domain-coherent units
- **R**esist: Knowledge contamination and drift
- **A**dapt: Knowledge granularity for task context
- **F**it: Knowledge selection to task requirements
- **T**ranscend: Knowledge storage implementation

## Foundational Premise: Context as Knowledge Architecture

> **Context is knowledge in motion.** Every piece of information an agent receives—from business rules to technical constraints to design principles—represents crystallized knowledge from different domains of expertise. The challenge isn't just storing this knowledge, but **assembling the right knowledge at the right time** to enable aligned decision-making.

When we ask an agent to implement a feature, we're implicitly asking it to synthesize knowledge across multiple interconnected domains:

**Strategic Layer**:
- **Market Context**: Industry trends, competitive landscape, regulatory environment
- **Organizational Knowledge**: Company values, strategic goals, resource constraints
- **Stakeholder Context**: User personas, business stakeholders, team dynamics

**Operational Layer**:
- **Business Problem Domain**: What user need are we solving? What constraints exist?
- **Business Solution Domain**: How do we solve it within our product strategy?
- **Technical Solution Domain**: What implementation approach fits our stack?
- **Architecture Knowledge**: How does this fit our system design patterns?
- **Visual Design Knowledge**: What UI patterns and brand guidelines apply?
- **Interaction Design Knowledge**: How should users experience this feature?

**Execution Layer**:
- **Communications Knowledge**: How do we document, explain, and coordinate this work?
- **Process Knowledge**: Development workflows, testing protocols, deployment procedures
- **Tool Knowledge**: IDE configurations, build systems, monitoring setup

These domains form a **knowledge network**, not a strict hierarchy—architecture decisions influence design choices, which affect technical implementation, which constrains business solutions. The agent's success depends on accessing and combining relevant knowledge from interconnected domains while **avoiding contamination** from irrelevant domains (e.g., Python expertise when working in Java, or retail business logic when building a healthcare app).

## The Knowledge Assembly Problem

Traditional approaches to agent context suffer from **knowledge mixing failures**:

1. **Domain Bleed**: Including Python debugging knowledge when working on Java code
2. **Temporal Mismatch**: Using outdated architectural decisions that have been superseded
3. **Scope Creep**: Loading enterprise patterns when building a simple prototype
4. **Authority Confusion**: Mixing authoritative documentation with experimental notes
5. **Granularity Mismatch**: Providing high-level strategy when detailed implementation guidance is needed

CRAFT provides a systematic approach to **knowledge curation and assembly** that prevents these failures by treating knowledge as a multi-dimensional, versionable, and contextually-bounded resource.

---

## C — **Chunk** Knowledge into Domain-Coherent Units

### Theory: Semantic Boundaries Prevent Knowledge Interference

Knowledge should be **atomically scoped** to prevent one domain's assumptions from polluting another. Each knowledge chunk should represent a single, coherent concept within a specific domain and authority level.

### Knowledge Chunking Strategies

| Strategy | Application | Example |
|----------|-------------|---------|
| **Domain Isolation** | Separate knowledge by problem domain | `business/user-research/`, `technical/java/`, `design/interaction/` |
| **Authority Layering** | Distinguish authoritative vs. experimental knowledge | `official/coding-standards.md` vs. `experiments/performance-ideas.md` |
| **Temporal Versioning** | Version knowledge by decision epochs | `architecture/v2-microservices/` vs. `architecture/v1-monolith/` |
| **Scope Tagging** | Tag knowledge by applicable scope | `#enterprise`, `#prototype`, `#mobile-only` |
| **Network Clustering** | Group interconnected knowledge regardless of domain | Authentication cluster: security policies + OAuth patterns + UI flows + compliance rules |

### Practical Example: Feature Implementation Knowledge

**Bad Chunking** (domain mixing):
```
feature-development.md:
- Use React hooks for state management
- Follow WCAG 2.1 AA accessibility standards
- Implement user analytics tracking
- Use Java Spring for backend APIs
- Follow brand color palette #FF6B6B
- Consider GDPR compliance for EU users
```

**Good Chunking** (domain separation):
```
technical/frontend/react-patterns.md:
- Use React hooks for state management
- Implement error boundaries for robustness

design/accessibility/wcag-guidelines.md:
- Follow WCAG 2.1 AA standards
- Ensure keyboard navigation support

business/analytics/tracking-requirements.md:
- Track user engagement on key features
- Measure conversion funnel performance

technical/backend/java-api-patterns.md:
- Use Spring Boot for REST endpoints
- Implement proper exception handling

design/visual/brand-standards.md:
- Primary brand color: #FF6B6B
- Typography hierarchy standards

business/compliance/privacy-requirements.md:
- GDPR compliance for EU user data
- Cookie consent implementation
```

### Engineering Concerns: Implementing Effective Knowledge Chunking

From an engineering perspective, effective chunking requires balancing retrieval precision with system complexity. Consider these implementation factors:

**Chunk Size Optimization**:
- **Token density**: Target 200-500 tokens per chunk for optimal embedding quality
- **Semantic completeness**: Ensure each chunk contains complete thoughts, not fragment sentences
- **Boundary detection**: Use semantic parsing to identify natural conceptual boundaries rather than arbitrary overlaps
- **Hierarchical nesting**: Structure chunks to support both fine-grained and coarse-grained retrieval

**Semantic Boundary Detection**:
Instead of relying on token overlaps and arbitrary character, word, or sentence boundaries, implement intelligent boundary detection:

```typescript
interface SemanticBoundaryDetector {
  identifyConceptualBreaks(document: string): BoundaryPoint[]
  validateChunkCompleteness(chunk: string): CompletenessScore
  detectIncompleteReferences(chunk: string): IncompleteReference[]
}

class StructuralBoundaryDetector implements SemanticBoundaryDetector {
  identifyConceptualBreaks(document: string): BoundaryPoint[] {
    // 1. Parse document structure (headings, sections, lists)
    // 2. Identify topic transitions using NLP coherence scoring
    // 3. Detect complete code blocks, examples, and procedures
    // 4. Ensure each chunk represents a complete conceptual unit

    return this.analyzeSemanticCoherence(document)
      .filter(boundary => boundary.coherenceScore > COHERENCE_THRESHOLD)
      .map(boundary => this.validateCompleteness(boundary))
  }

  validateChunkCompleteness(chunk: string): CompletenessScore {
    // Check for:
    // - Dangling references ("as mentioned above", "the following")
    // - Incomplete examples or code snippets
    // - Missing context required to understand the concept
    // - Circular dependencies with other chunks
  }
}
```

**Problems with Overlap-Based Chunking**:
- **Information duplication**: Same content appears in multiple chunks, confusing retrieval
- **Inconsistent updates**: When source changes, overlapped content becomes inconsistent
- **Semantic fragmentation**: Concepts get split across arbitrary boundaries
- **Retrieval noise**: Multiple chunks with similar content dilute relevance scoring
- **Storage bloat**: Redundant content wastes embedding space and compute
- **Version skew**: Different overlapped versions of the same content create conflicts

**Storage and Indexing Architecture**:
```typescript
interface KnowledgeChunk {
  id: string
  content: string
  metadata: {
    domain: string[]        // ["technical", "frontend"]
    authority: AuthorityLevel
    scope: string[]         // ["enterprise", "production"]
    granularity: GranularityLevel
    dependencies: string[]  // chunk IDs this depends on
    conflicts: string[]     // chunk IDs this conflicts with
    timestamp: number
    version: string
  }
  embedding: number[]       // vector representation
  tags: string[]           // searchable labels
}
```

**Knowledge Graph Enhancement**:
While vector embeddings excel at semantic similarity, knowledge graphs provide superior relationship modeling for complex knowledge networks:

```typescript
interface KnowledgeGraph {
  nodes: KnowledgeNode[]
  relationships: KnowledgeRelationship[]

  // Graph operations for rich relationship traversal
  findRelatedConcepts(nodeId: string, relationTypes: string[], depth: number): KnowledgeNode[]
  detectImplicitConnections(domain1: string, domain2: string): Connection[]
  validateDependencyChains(chunks: KnowledgeChunk[]): ValidationResult[]
}

interface KnowledgeNode extends KnowledgeChunk {
  // Graph-specific properties
  centrality: number        // How connected this node is
  clusterId: string        // Which knowledge cluster it belongs to
  bridgeScore: number      // How much it connects different domains
}

interface KnowledgeRelationship {
  source: string           // Source chunk ID
  target: string           // Target chunk ID
  type: RelationshipType   // DEPENDS_ON, CONFLICTS_WITH, IMPLEMENTS, EXTENDS
  strength: number         // Relationship strength (0-1)
  context: string[]        // Contexts where this relationship applies
  temporal: TemporalScope  // When this relationship is valid
}

enum RelationshipType {
  DEPENDS_ON = "depends_on",           // Technical dependencies
  CONFLICTS_WITH = "conflicts_with",   // Contradictory guidance
  IMPLEMENTS = "implements",           // Concrete implementation of principle
  EXTENDS = "extends",                 // Builds upon concept
  RELATES_TO = "relates_to",          // General semantic relationship
  SUPERSEDES = "supersedes",          // Newer version replaces older
  COMPLEMENTS = "complements",        // Works together synergistically
  REQUIRES_CONTEXT = "requires_context" // Needs additional context to apply
}
```

**Hybrid Vector-Graph Architecture**:
```typescript
class HybridKnowledgeStore {
  constructor(
    private vectorStore: VectorStore,
    private graphStore: GraphStore
  ) {}

  async intelligentRetrieval(query: string, context: TaskContext): Promise<KnowledgeChunk[]> {
    // 1. Vector similarity for initial candidates
    const semanticMatches = await this.vectorStore.search(query, 50)

    // 2. Graph traversal for relationship discovery
    const relatedNodes = await this.graphStore.expandNetworkFromNodes(
      semanticMatches.map(m => m.id),
      context.allowedRelationTypes,
      context.maxTraversalDepth
    )

    // 3. Relationship-aware ranking
    const rankedResults = this.rankByRelationshipRelevance(
      [...semanticMatches, ...relatedNodes],
      query,
      context
    )

    return rankedResults
  }

  async detectKnowledgeGaps(requiredChunks: KnowledgeChunk[]): Promise<Gap[]> {
    // Use graph analysis to find missing dependencies or broken chains
    const dependencyGraph = await this.graphStore.buildDependencySubgraph(
      requiredChunks.map(c => c.id)
    )

    return this.graphStore.findMissingNodes(dependencyGraph)
  }
}
```

**Graph-Enhanced Quality Assurance**:
- **Dependency validation**: Ensure all prerequisite knowledge is included when a chunk is selected
- **Conflict detection**: Use graph traversal to identify contradictory guidance paths
- **Knowledge completeness**: Detect missing concepts by analyzing relationship gaps
- **Domain bridging**: Identify chunks that effectively connect different knowledge domains
- **Temporal consistency**: Validate that relationship chains respect temporal ordering (e.g., don't reference future concepts in past contexts)

**Retrieval Performance Considerations**:
- **Index organization**: Separate indexes by domain for faster filtering
- **Metadata queries**: Pre-filter by domain/scope before vector similarity
- **Caching strategies**: Cache frequently accessed chunk combinations
- **Lazy loading**: Load chunk content only when selected for context

**Quality Assurance Mechanisms**:
- **Automated chunk validation**: Check for semantic completeness and proper metadata
- **Cross-reference detection**: Identify missing dependencies between chunks
- **Staleness monitoring**: Track when chunks become outdated relative to source material
- **Retrieval analytics**: Monitor which chunks are co-retrieved to identify clustering opportunities

---

## R — **Resist** Knowledge Contamination and Drift

### Theory: Knowledge Provenance and Immutability

New knowledge should **extend** the agent's capabilities without **invalidating** existing reliable knowledge. This requires treating knowledge as an append-only system with clear provenance and conflict resolution strategies.

### Knowledge Resistance Strategies

| Strategy | Purpose | Implementation |
|----------|---------|----------------|
| **Provenance Tracking** | Know the source and authority of each knowledge piece | `source: "CTO architecture review 2024-Q4"` |
| **Knowledge Versioning** | Maintain historical knowledge while adding new | `v1-patterns/` alongside `v2-patterns/` |
| **Conflict Detection** | Identify when new knowledge contradicts existing | Automated checks for contradictory guidance |
| **Authority Hierarchies** | Establish which knowledge takes precedence | Official docs > team conventions > individual experiments |

### Practical Example: Evolving Architecture Patterns

**Scenario**: Team migrates from REST to GraphQL but needs to maintain existing services.

**Resistant Knowledge Structure**:
```
architecture/
├── api-patterns/
│   ├── v1-rest/
│   │   ├── authority: "deprecated-but-supported"
│   │   ├── scope: "existing-services"
│   │   └── rest-best-practices.md
│   └── v2-graphql/
│       ├── authority: "current-standard"
│       ├── scope: "new-development"
│       └── graphql-patterns.md
└── migration/
    ├── rest-to-graphql-guide.md
    └── compatibility-requirements.md
```

**Example Contamination Scenario**:
Team decides to adopt React Server Components (RSC) for new features while keeping existing client-side React patterns.

**Bad Approach** (knowledge contamination):
```
react-patterns.md: // Single file - knowledge conflict!
- Use useState and useEffect for client state (OLD)
- Server Components don't use hooks (NEW)
- Fetch data in useEffect (OLD)
- Fetch data during server render (NEW)
- All components should be functions (CONFLICTING!)
```
*Result*: Agent gets contradictory guidance, mixes paradigms incorrectly.

**Good Approach** (resistant knowledge structure):
```
frontend/
├── client-patterns/
│   ├── authority: "legacy-maintained"
│   ├── scope: "existing-features"
│   ├── react-hooks-patterns.md
│   └── client-data-fetching.md
├── server-patterns/
│   ├── authority: "current-standard"
│   ├── scope: "new-development"
│   ├── server-components.md
│   └── server-data-fetching.md
└── transition/
    ├── hybrid-app-architecture.md
    ├── migration-strategy.md
    └── compatibility-rules.md
```

**Context Assembly Logic**:
- **New feature development**: Load server-patterns + transition/compatibility-rules
- **Existing feature maintenance**: Load client-patterns + transition/migration-strategy
- **Full-stack features**: Load both patterns + transition/hybrid-app-architecture

*Result*: Agent gets clean, non-contradictory guidance appropriate for the specific work context.

### Engineering Concerns: Building Resilient Knowledge Systems

Resistance mechanisms must be engineered into the knowledge infrastructure from the ground up. Key implementation challenges include:

**Conflict Detection Algorithms**:
```typescript
interface ConflictDetector {
  detectSemanticConflicts(newChunk: KnowledgeChunk, existingChunks: KnowledgeChunk[]): Conflict[]
  detectTemporalConflicts(newChunk: KnowledgeChunk): TemporalConflict[]
  detectAuthorityConflicts(chunks: KnowledgeChunk[]): AuthorityConflict[]
}

// Example: Detecting contradictory technical recommendations
class TechnicalConflictDetector implements ConflictDetector {
  detectSemanticConflicts(newChunk, existingChunks) {
    // Use embedding similarity + keyword analysis
    // Flag chunks with high similarity but contradictory directives
    // E.g., "use React hooks" vs "use class components"
  }
}
```

**Versioning Strategy Trade-offs**:
- **Git-style branching**: Enables parallel evolution but increases complexity
- **Sequential versioning**: Simpler but can create knowledge silos
- **Semantic versioning**: Clear deprecation paths but requires human judgment
- **Time-based versioning**: Automatic but may not align with logical transitions

**Authority Resolution Mechanisms**:
```typescript
interface AuthorityResolver {
  resolveConflict(conflictingChunks: KnowledgeChunk[]): ResolvedKnowledge
  getAuthorityHierarchy(domain: string): AuthorityLevel[]
  validateAuthorityChain(chunk: KnowledgeChunk): ValidationResult
}

// Authority levels with clear precedence rules
enum AuthorityLevel {
  FOUNDATIONAL = 100,    // Company mission, legal requirements
  OFFICIAL = 80,         // Approved standards, architectural decisions
  CONVENTIONAL = 60,     // Team conventions, best practices
  EXPERIMENTAL = 40,     // Trials, proof-of-concepts
  DEPRECATED = 20,       // Outdated but maintained
  INVALID = 0           // Contradicted or superseded
}
```

**Data Integrity Safeguards**:
- **Immutable chunk storage**: Prevent accidental modification of historical knowledge
- **Audit logging**: Track all knowledge changes with full provenance
- **Rollback capabilities**: Quick recovery from contaminated knowledge ingestion
- **Validation pipelines**: Automated testing of knowledge consistency before deployment

**Performance vs. Consistency Trade-offs**:
- **Eventual consistency**: Faster ingestion but temporary conflicts possible
- **Strong consistency**: Immediate conflict resolution but slower updates
- **Hybrid approaches**: Critical domains use strong consistency, others eventual
- **Conflict quarantine**: Isolate conflicting knowledge until human review

---

## A — **Adapt** Knowledge Granularity for Task Context

### Theory: Knowledge Zoom Levels

Different tasks require different levels of detail from the same knowledge domain. The system should provide **knowledge substitutability**—the ability to swap high-level principles with detailed implementation guidance seamlessly.

### Knowledge Adaptation Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Hierarchical Summaries** | Provide overview first, details on demand | Strategic goals → tactical approaches → implementation steps |
| **Progressive Disclosure** | Start simple, add complexity as needed | Basic pattern → advanced variations → edge cases |
| **Context-Aware Filtering** | Show relevant detail level for current task | Architecture overview for planning, detailed APIs for implementation |
| **Multi-Resolution Storage** | Store same knowledge at multiple granularities | Principle + guideline + checklist + examples |

### Practical Example: API Design Knowledge

**Multi-Granular Knowledge Structure**:
```
api-design/
├── principles/
│   └── rest-principles.md           # High-level: "Resources, not actions"
├── guidelines/
│   └── rest-guidelines.md           # Medium: "Use nouns for endpoints"
├── patterns/
│   └── rest-patterns.md            # Detailed: "GET /users/{id}/orders"
└── examples/
    └── user-management-api.md       # Concrete: Full implementation
```

**Example Adaptation Scenario**:
Team needs database design guidance for different contexts - from architecture planning to implementation.

**Multi-Granular Knowledge Structure**:
```
database/
├── philosophy/
│   └── data-modeling-principles.md    # "Design for queries, not storage"
├── strategy/
│   └── schema-design-guidelines.md    # "Normalize to 3NF, denormalize for performance"
├── tactics/
│   └── postgres-patterns.md          # "Use JSONB for flexible fields"
├── implementation/
│   └── migration-procedures.md       # "CREATE TABLE users (id SERIAL PRIMARY KEY...)"
└── examples/
    └── user-system-schema.sql        # Complete working schema
```

**Adaptive Context Assembly Examples**:

**Planning Meeting Context**:
```
Agent receives: philosophy + strategy
- "Design for your most common queries"
- "Consider read vs write patterns early"
- "Plan for data growth and access patterns"
```

**Technical Design Context**:
```
Agent receives: strategy + tactics
- "Normalize to 3NF for consistency"
- "Use JSONB for user preferences in Postgres"
- "Index foreign keys and query columns"
```

**Implementation Context**:
```
Agent receives: tactics + implementation + examples
- "Use JSONB for flexible fields"
- "CREATE TABLE users (id SERIAL PRIMARY KEY, preferences JSONB)"
- [Complete working schema with proper indexes]
```

**Code Review Context**:
```
Agent receives: ALL levels for comprehensive evaluation
- Can check alignment with principles
- Verify adherence to guidelines
- Validate implementation details
- Compare against proven examples
```

*Result*: Same knowledge domain provides appropriate detail level for each work context, avoiding information overload or insufficient detail.

### Engineering Concerns: Implementing Adaptive Knowledge Systems

Adaptive granularity requires sophisticated orchestration between storage, retrieval, and context assembly systems:

**Multi-Resolution Storage Architecture**:
```typescript
interface MultiResolutionChunk {
  baseId: string
  resolutions: {
    summary: KnowledgeChunk     // 50-100 tokens
    overview: KnowledgeChunk    // 200-300 tokens
    detailed: KnowledgeChunk    // 500-800 tokens
    comprehensive: KnowledgeChunk // 1000+ tokens
  }
  crossReferences: {
    upward: string[]    // links to higher-level concepts
    downward: string[]  // links to implementation details
    lateral: string[]   // links to related concepts
  }
}
```

**Dynamic Context Window Management**:
- **Token budget allocation**: Reserve portions of context for different granularity levels
- **Progressive loading**: Start with summaries, expand high-confidence matches
- **Compression algorithms**: Automatic summarization when approaching token limits
- **Swapping strategies**: Replace low-relevance detailed chunks with high-relevance summaries

**Granularity Selection Algorithms**:
```typescript
interface GranularitySelector {
  selectOptimalGranularity(
    query: string,
    taskContext: TaskContext,
    availableTokens: number,
    userExpertise: ExpertiseLevel
  ): GranularitySelection[]
}

class ContextAwareSelector implements GranularitySelector {
  selectOptimalGranularity(query, context, tokens, expertise) {
    // Algorithm factors:
    // 1. Task phase (planning → overview, implementation → detailed)
    // 2. Query specificity (vague → summary, specific → detailed)
    // 3. User expertise (novice → comprehensive, expert → minimal)
    // 4. Available context budget (limited → summaries, ample → full)
    // 5. Domain complexity (simple → basic, complex → progressive)
  }
}
```

**Substitutability Engineering**:
- **Interface consistency**: All granularity levels expose same conceptual structure
- **Semantic preservation**: Summaries maintain logical equivalence to detailed versions
- **Dependency tracking**: Ensure substitutions don't break knowledge networks
- **Quality metrics**: Automated validation that abstractions preserve key information

**Real-time Adaptation Mechanisms**:
```typescript
interface AdaptationEngine {
  monitorContextUtilization(): UtilizationMetrics
  detectGranularityMismatch(): MismatchSignal[]
  requestGranularityAdjustment(chunks: string[], targetLevel: GranularityLevel): void
  optimizeContextWindow(): OptimizationResult
}

// Example: Detecting when user needs more detail
class UsageBasedAdapter implements AdaptationEngine {
  detectGranularityMismatch() {
    // Signals: repeated clarification questions, low task completion rate,
    // frequent requests for "more details" or "can you be more specific"
    // Auto-upgrade to higher granularity for future similar queries
  }
}
```

**Caching and Performance Optimization**:
- **Granularity-aware caching**: Cache common granularity combinations
- **Precomputed abstractions**: Generate summaries during ingestion, not retrieval
- **Hot-swapping**: Zero-downtime granularity level changes
- **Load balancing**: Distribute granularity computation across processing nodes

---

## F — **Fit** Knowledge Selection to Task Requirements

### Theory: Relevance-Driven Knowledge Assembly

The agent should receive **only the knowledge necessary** for the current task context. This requires sophisticated filtering based on task type, domain requirements, and user intent.

### Knowledge Fitting Strategies

| Strategy | Mechanism | Example |
|----------|-----------|---------|
| **Domain Filtering** | Load only relevant knowledge domains | Frontend task → exclude backend knowledge |
| **Scope Matching** | Match knowledge scope to task scope | Prototype task → exclude enterprise patterns |
| **Task-Type Routing** | Different knowledge for different task types | Debug task → load troubleshooting knowledge |
| **Competency Gating** | Adjust knowledge complexity to user level | Junior dev → include more explanatory context |

### Practical Example: Feature Development Task

**Task**: "Add a user profile editing form to our React app"

**Knowledge Assembly Decision Tree**:
```
1. Domain Analysis:
   - Frontend: YES (React development needed)
   - Backend: NO (no API changes mentioned)
   - Design: YES (form UI needed)
   - Business: MINIMAL (standard user management)

2. Scope Analysis:
   - Component: Form creation and validation
   - Integration: Existing user system
   - Complexity: Standard CRUD operation

3. Knowledge Selection:
   ✓ react/form-patterns.md
   ✓ react/validation-approaches.md
   ✓ design/form-ui-standards.md
   ✓ accessibility/form-accessibility.md
   ✗ react/advanced-performance.md (not needed for simple form)
   ✗ backend/user-apis.md (using existing APIs)
   ✗ business/user-acquisition.md (not relevant to editing)
```

**Example Fitting Scenario**:
Developer asks: "Help me add real-time notifications to our e-commerce app"

**Available Knowledge Domains**:
```
Available knowledge base contains:
✓ frontend/react-websockets.md
✓ frontend/notification-ui-patterns.md
✓ backend/websocket-servers.md
✓ backend/nodejs-realtime.md
✓ backend/python-django-channels.md
✓ backend/java-spring-websockets.md
✓ infrastructure/websocket-scaling.md
✓ infrastructure/kubernetes-deployments.md
✓ business/notification-preferences.md
✓ business/ecommerce-marketing.md
✓ business/healthcare-compliance.md
✓ design/mobile-notification-patterns.md
✓ design/accessibility-alerts.md
✗ backend/machine-learning-pipelines.md
✗ business/financial-reporting.md
✗ design/print-layout-systems.md
```

**Knowledge Fitting Decision Process**:
```
1. Domain Analysis:
   ✓ Frontend: YES (UI notifications needed)
   ✓ Backend: YES (real-time server needed)
   ✓ Infrastructure: MAYBE (depends on scale)
   ✓ Business: MINIMAL (notification preferences only)
   ✓ Design: YES (notification UX needed)

2. Technology Context Detection:
   - Codebase scan reveals: React frontend, Node.js backend
   ✓ Include: react-websockets, nodejs-realtime, notification-ui-patterns
   ✗ Exclude: python-django-channels, java-spring-websockets

3. Business Context Filtering:
   - App type: e-commerce
   ✓ Include: ecommerce notification patterns, user preferences
   ✗ Exclude: healthcare-compliance (wrong domain)

4. Scale Assessment:
   - Current users: <1000 (detected from analytics)
   ✗ Exclude: websocket-scaling, kubernetes-deployments (premature)
```

**Resulting Fitted Context** (lean and targeted):
```
Selected Knowledge:
- frontend/react-websockets.md
- frontend/notification-ui-patterns.md
- backend/nodejs-realtime.md
- business/notification-preferences.md
- design/accessibility-alerts.md

Excluded Knowledge (avoiding noise):
- All non-Node.js backend patterns
- Healthcare compliance rules
- Enterprise scaling infrastructure
- Marketing automation (not real-time notifications)
- Machine learning (not requested)
```

*Result*: Agent receives precisely the knowledge needed for React + Node.js real-time notifications in an e-commerce context, without being distracted by irrelevant technologies, business domains, or premature optimization concerns.

### Engineering Concerns: Precision Knowledge Selection Systems

Fitting requires sophisticated query understanding and relevance ranking systems that operate in real-time:

**Query Analysis and Intent Recognition**:
```typescript
interface QueryAnalyzer {
  extractDomains(query: string): Domain[]
  inferTaskType(query: string, context: ProjectContext): TaskType
  assessComplexity(query: string): ComplexityLevel
  detectScope(query: string, codebase: CodebaseContext): ScopeLevel
}

class SemanticQueryAnalyzer implements QueryAnalyzer {
  extractDomains(query: string): Domain[] {
    // NLP pipeline: named entity recognition, keyword extraction,
    // technology stack detection, domain classification
    // E.g., "React form validation" → [frontend, react, forms, validation]
  }

  inferTaskType(query, context): TaskType {
    // Pattern matching + ML classification:
    // "debug", "fix", "error" → DEBUG
    // "implement", "add", "create" → DEVELOP
    // "optimize", "improve", "performance" → OPTIMIZE
    // "document", "explain", "how does" → EXPLAIN
  }
}
```

**Relevance Scoring and Ranking**:
```typescript
interface RelevanceEngine {
  calculateDomainRelevance(chunk: KnowledgeChunk, domains: Domain[]): number
  calculateScopeRelevance(chunk: KnowledgeChunk, scope: ScopeLevel): number
  calculateTemporalRelevance(chunk: KnowledgeChunk, context: TaskContext): number
  calculateAuthorityWeight(chunk: KnowledgeChunk, taskCriticality: number): number
}

class MultiFactorRelevanceEngine implements RelevanceEngine {
  calculateOverallRelevance(chunk: KnowledgeChunk, query: AnalyzedQuery): number {
    const weights = {
      domain: 0.4,      // How well does chunk match required domains?
      scope: 0.3,       // Does chunk scope match task scope?
      temporal: 0.15,   // Is chunk current for the task context?
      authority: 0.15   // Does chunk authority match task criticality?
    }

    return (
      weights.domain * this.calculateDomainRelevance(chunk, query.domains) +
      weights.scope * this.calculateScopeRelevance(chunk, query.scope) +
      weights.temporal * this.calculateTemporalRelevance(chunk, query.context) +
      weights.authority * this.calculateAuthorityWeight(chunk, query.criticality)
    )
  }
}
```

**Dynamic Filtering Pipelines**:
```typescript
interface FilteringPipeline {
  applyDomainFilters(chunks: KnowledgeChunk[], domains: Domain[]): KnowledgeChunk[]
  applyScopeFilters(chunks: KnowledgeChunk[], scope: ScopeLevel): KnowledgeChunk[]
  applyCompetencyFilters(chunks: KnowledgeChunk[], userLevel: ExpertiseLevel): KnowledgeChunk[]
  applyContextFilters(chunks: KnowledgeChunk[], context: TaskContext): KnowledgeChunk[]
}

// Configurable pipeline with pluggable filters
class ConfigurableFilterPipeline implements FilteringPipeline {
  private filters: Filter[] = [
    new StaleKnowledgeFilter(),      // Remove outdated chunks
    new ConflictingKnowledgeFilter(), // Remove contradictory chunks
    new OverspecializationFilter(),   // Remove overly narrow chunks
    new RedundancyFilter()           // Remove duplicate information
  ]

  process(chunks: KnowledgeChunk[], query: AnalyzedQuery): KnowledgeChunk[] {
    return this.filters.reduce(
      (filtered, filter) => filter.apply(filtered, query),
      chunks
    )
  }
}
```

**Context Budget Optimization**:
```typescript
interface ContextBudgetManager {
  calculateOptimalAllocation(
    availableTokens: number,
    relevantChunks: ScoredChunk[],
    diversityRequirement: number
  ): ChunkSelection[]

  handleTokenOverflow(
    selectedChunks: KnowledgeChunk[],
    tokenLimit: number
  ): ChunkSelection[]
}

class GreedyBudgetManager implements ContextBudgetManager {
  calculateOptimalAllocation(tokens, chunks, diversity) {
    // Multi-objective optimization:
    // 1. Maximize relevance score sum
    // 2. Maintain domain diversity
    // 3. Respect token budget constraints
    // 4. Prefer authoritative sources

    // Uses modified knapsack algorithm with diversity constraints
  }
}
```

**Real-time Performance Requirements**:
- **Sub-100ms filtering**: Query analysis and initial filtering must be near-instantaneous
- **Scalable indexing**: Support millions of chunks with constant-time domain filtering
- **Caching strategies**: Pre-compute common query patterns and filter combinations
- **Fallback mechanisms**: Graceful degradation when filtering fails or times out

---

## T — **Transcend** Knowledge Storage Implementation

### Theory: Knowledge Interface Abstraction

Higher-level reasoning about knowledge should be **independent of storage mechanism**. Whether knowledge lives in vector databases, graph structures, or file systems, the agent should interact through consistent knowledge interfaces.

### Knowledge Transcendence Patterns

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| **Knowledge Interfaces** | Abstract away storage details | `KnowledgeDomain.get(query, context)` |
| **Cross-Domain Synthesis** | Combine knowledge from multiple sources | Business rules + technical constraints → implementation guidance |
| **Knowledge Transformation** | Convert between knowledge representations | Code → documentation → examples → tests |
| **Adaptive Resolution** | Switch knowledge sources based on availability | Primary docs unavailable → fallback to examples |

### Practical Example: Cross-Domain Knowledge Synthesis

**Scenario**: Implementing user authentication with specific business and technical requirements.

**Knowledge Sources**:
- **Business Domain**: `security-requirements/`, `user-experience/`
- **Technical Domain**: `authentication/jwt/`, `authentication/oauth/`
- **Architecture Domain**: `microservices/`, `api-gateway/`
- **Compliance Domain**: `gdpr/`, `security-standards/`

**Synthesis Process**:
```
1. Query: "Implement user login with social OAuth"

2. Cross-Domain Assembly:
   Business: "Support Google and GitHub OAuth"
   + Technical: "Use JWT for session management"
   + Architecture: "Route through API gateway"
   + Compliance: "Collect minimal required data"

3. Synthesized Guidance:
   "Implement OAuth2 flow with Google/GitHub providers,
    issue JWT tokens through API gateway,
    store only email and display name per GDPR"

4. Supporting Knowledge:
   - OAuth implementation patterns
   - JWT token structure standards
   - API gateway configuration
   - Data minimization compliance checks
```

**Example Transcendence Scenario**:
Developer needs guidance for implementing user authentication, but knowledge is distributed across multiple systems.

**Knowledge Storage Reality**:
```
Knowledge exists in different formats and locations:
- Official docs: Confluence wiki pages
- Code examples: GitHub repositories
- Team knowledge: Slack conversations + Notion pages
- Compliance rules: Legal team's shared drive PDFs
- Architecture decisions: Miro boards + ADR documents
- Security standards: Company security portal
```

**Without Transcendence** (storage-dependent):
```
Agent instructions:
"Check the Confluence page 'Auth Patterns' and the GitHub repo
'auth-examples' and look at Slack #security-team for recent
discussions and download the GDPR PDF from legal drive..."
```
*Result*: Brittle, manual, breaks when storage systems change.

**With Transcendence** (storage-agnostic):
```typescript
// Knowledge Interface Abstraction
interface AuthenticationKnowledge {
  getSecurityRequirements(context: BusinessContext): SecurityRules
  getImplementationPatterns(stack: TechStack): CodePatterns
  getComplianceGuidance(regions: string[]): ComplianceRules
  getArchitectureDecisions(scope: string): ArchitectureGuidance
}

// Cross-Domain Synthesis
const authGuidance = await knowledgeOrchestrator.synthesize({
  query: "Implement user authentication for EU customers",
  domains: [
    SecurityDomain,     // -> OAuth2 + JWT patterns
    ComplianceDomain,   // -> GDPR requirements
    ArchitectureDomain, // -> Microservice integration
    TechnicalDomain     // -> Node.js implementation
  ],
  context: {
    region: "EU",
    stack: "node-react",
    scale: "startup"
  }
})
```

**Synthesized Result**:
```
Assembled Guidance:
"Implement OAuth2 with Google/GitHub providers (Security)
+ Issue JWT tokens with 24hr expiry (Architecture)
+ Store only email and display name (GDPR Compliance)
+ Use passport.js with express-session (Technical Implementation)
+ Include cookie consent banner (Legal Requirement)
+ Log authentication events for audit (Security Monitoring)"

Supporting Context:
- OAuth2 flow diagrams (from architecture docs)
- Passport.js code examples (from GitHub repos)
- GDPR compliance checklist (from legal PDFs)
- JWT token structure (from security wiki)
- Cookie consent implementation (from design system)
```

**Storage Evolution Resilience**:
```
✓ Team migrates from Confluence to Notion → Interface unchanged
✓ Code examples move from GitHub to GitLab → Interface unchanged
✓ Compliance rules update in legal system → Interface unchanged
✓ New security standards added → Interface extends seamlessly
```

*Result*: Agent receives coherent, synthesized guidance regardless of where knowledge is stored, and the system remains robust as storage systems evolve or knowledge sources change.

### Engineering Concerns: Building Storage-Agnostic Knowledge Systems

Transcendence requires careful architectural design to decouple knowledge operations from storage implementation details:

**Knowledge Interface Design**:
```typescript
// Abstract knowledge interface - storage agnostic
interface KnowledgeDomain {
  query(request: KnowledgeQuery): Promise<KnowledgeResult[]>
  synthesize(domains: string[], context: SynthesisContext): Promise<SynthesizedKnowledge>
  transform(source: KnowledgeChunk, targetFormat: KnowledgeFormat): Promise<KnowledgeChunk>
  validate(knowledge: KnowledgeChunk): ValidationResult
}

// Concrete implementations for different storage backends
class VectorKnowledgeDomain implements KnowledgeDomain {
  constructor(private vectorStore: VectorStore) {}

  async query(request: KnowledgeQuery): Promise<KnowledgeResult[]> {
    const embedding = await this.embedQuery(request.text)
    return this.vectorStore.similaritySearch(embedding, request.filters)
  }
}

class GraphKnowledgeDomain implements KnowledgeDomain {
  constructor(private graphStore: GraphStore) {}

  async query(request: KnowledgeQuery): Promise<KnowledgeResult[]> {
    const entities = this.extractEntities(request.text)
    return this.graphStore.traverseRelations(entities, request.depth)
  }
}

// Knowledge orchestrator - routes to appropriate implementation
class KnowledgeOrchestrator {
  private domains: Map<string, KnowledgeDomain> = new Map()

  registerDomain(name: string, domain: KnowledgeDomain) {
    this.domains.set(name, domain)
  }

  async synthesize(request: CrossDomainQuery): Promise<SynthesizedKnowledge> {
    const results = await Promise.all(
      request.domains.map(domainName =>
        this.domains.get(domainName)?.query(request.query)
      )
    )

    return this.synthesizeResults(results, request.context)
  }
}
```

**Cross-Domain Synthesis Architecture**:
```typescript
interface KnowledgeSynthesizer {
  combineKnowledge(sources: KnowledgeSource[]): SynthesizedKnowledge
  resolveConflicts(conflictingKnowledge: ConflictSet[]): ResolutionStrategy
  validateSynthesis(synthesized: SynthesizedKnowledge): QualityMetrics
}

class SemanticSynthesizer implements KnowledgeSynthesizer {
  combineKnowledge(sources: KnowledgeSource[]): SynthesizedKnowledge {
    // 1. Identify overlapping concepts across domains
    // 2. Merge complementary information
    // 3. Resolve contradictions using authority hierarchy
    // 4. Generate coherent unified guidance
    // 5. Maintain provenance links to original sources
  }

  resolveConflicts(conflicts: ConflictSet[]): ResolutionStrategy {
    // Conflict resolution strategies:
    // - Authority-based: Higher authority sources win
    // - Temporal-based: More recent knowledge preferred
    // - Domain-specific: Each domain's expertise respected
    // - Context-aware: Best fit for current task context
  }
}
```

**Storage Backend Abstraction**:
```typescript
interface StorageBackend {
  store(chunk: KnowledgeChunk): Promise<string>
  retrieve(id: string): Promise<KnowledgeChunk | null>
  search(query: SearchQuery): Promise<SearchResult[]>
  update(id: string, chunk: KnowledgeChunk): Promise<void>
  delete(id: string): Promise<void>
}

// Adapter pattern for different storage systems
class VectorStoreAdapter implements StorageBackend {
  constructor(private pineconeClient: PineconeClient) {}
  // Vector-specific implementation
}

class GraphStoreAdapter implements StorageBackend {
  constructor(private neo4jDriver: Neo4jDriver) {}
  // Graph-specific implementation
}

class HybridStoreAdapter implements StorageBackend {
  constructor(
    private vectorStore: VectorStoreAdapter,
    private graphStore: GraphStoreAdapter
  ) {}

  async search(query: SearchQuery): Promise<SearchResult[]> {
    // Route to appropriate backend based on query type
    if (query.requiresRelationalTraversal) {
      return this.graphStore.search(query)
    } else if (query.requiresSemanticSimilarity) {
      return this.vectorStore.search(query)
    } else {
      // Combine results from both backends
      const [vectorResults, graphResults] = await Promise.all([
        this.vectorStore.search(query),
        this.graphStore.search(query)
      ])
      return this.mergeResults(vectorResults, graphResults)
    }
  }
}
```

**Migration and Evolution Support**:
```typescript
interface KnowledgeMigrator {
  migrateToNewBackend(
    source: StorageBackend,
    target: StorageBackend,
    migrationStrategy: MigrationStrategy
  ): Promise<MigrationResult>

  validateMigration(result: MigrationResult): ValidationResult
  rollbackMigration(migrationId: string): Promise<void>
}

// Support for zero-downtime storage migrations
class GradualMigrator implements KnowledgeMigrator {
  async migrateToNewBackend(source, target, strategy) {
    // 1. Set up dual-write to both backends
    // 2. Gradually migrate existing data in batches
    // 3. Validate consistency between backends
    // 4. Switch read traffic to new backend
    // 5. Clean up old backend
  }
}
```

**Service Discovery and Load Balancing**:
- **Backend registration**: Dynamic discovery of available storage services
- **Health monitoring**: Automatic failover when backends become unavailable
- **Load distribution**: Route queries based on backend capacity and specialization
- **Circuit breakers**: Prevent cascade failures when backends are overloaded

### Practical Example: Cross-Domain Knowledge Synthesis

**Scenario**: Implementing user authentication with specific business and technical requirements.

**Knowledge Sources**:
- **Business Domain**: `security-requirements/`, `user-experience/`
- **Technical Domain**: `authentication/jwt/`, `authentication/oauth/`
- **Architecture Domain**: `microservices/`, `api-gateway/`
- **Compliance Domain**: `gdpr/`, `security-standards/`

**Synthesis Process**:
```
1. Query: "Implement user login with social OAuth"

2. Cross-Domain Assembly:
   Business: "Support Google and GitHub OAuth"
   + Technical: "Use JWT for session management"
   + Architecture: "Route through API gateway"
   + Compliance: "Collect minimal required data"

3. Synthesized Guidance:
   "Implement OAuth2 flow with Google/GitHub providers,
    issue JWT tokens through API gateway,
    store only email and display name per GDPR"

4. Supporting Knowledge:
   - OAuth implementation patterns
   - JWT token structure standards
   - API gateway configuration
   - Data minimization compliance checks
```

**Example Transcendence Scenario**:
Developer needs guidance for implementing user authentication, but knowledge is distributed across multiple systems.

**Knowledge Storage Reality**:
```
Knowledge exists in different formats and locations:
- Official docs: Confluence wiki pages
- Code examples: GitHub repositories
- Team knowledge: Slack conversations + Notion pages
- Compliance rules: Legal team's shared drive PDFs
- Architecture decisions: Miro boards + ADR documents
- Security standards: Company security portal
```

**Without Transcendence** (storage-dependent):
```
Agent instructions:
"Check the Confluence page 'Auth Patterns' and the GitHub repo
'auth-examples' and look at Slack #security-team for recent
discussions and download the GDPR PDF from legal drive..."
```
*Result*: Brittle, manual, breaks when storage systems change.

**With Transcendence** (storage-agnostic):
```typescript
// Knowledge Interface Abstraction
interface AuthenticationKnowledge {
  getSecurityRequirements(context: BusinessContext): SecurityRules
  getImplementationPatterns(stack: TechStack): CodePatterns
  getComplianceGuidance(regions: string[]): ComplianceRules
  getArchitectureDecisions(scope: string): ArchitectureGuidance
}

// Cross-Domain Synthesis
const authGuidance = await knowledgeOrchestrator.synthesize({
  query: "Implement user authentication for EU customers",
  domains: [
    SecurityDomain,     // -> OAuth2 + JWT patterns
    ComplianceDomain,   // -> GDPR requirements
    ArchitectureDomain, // -> Microservice integration
    TechnicalDomain     // -> Node.js implementation
  ],
  context: {
    region: "EU",
    stack: "node-react",
    scale: "startup"
  }
})
```

**Synthesized Result**:
```
Assembled Guidance:
"Implement OAuth2 with Google/GitHub providers (Security)
+ Issue JWT tokens with 24hr expiry (Architecture)
+ Store only email and display name (GDPR Compliance)
+ Use passport.js with express-session (Technical Implementation)
+ Include cookie consent banner (Legal Requirement)
+ Log authentication events for audit (Security Monitoring)"

Supporting Context:
- OAuth2 flow diagrams (from architecture docs)
- Passport.js code examples (from GitHub repos)
- GDPR compliance checklist (from legal PDFs)
- JWT token structure (from security wiki)
- Cookie consent implementation (from design system)
```

**Storage Evolution Resilience**:
```
✓ Team migrates from Confluence to Notion → Interface unchanged
✓ Code examples move from GitHub to GitLab → Interface unchanged
✓ Compliance rules update in legal system → Interface unchanged
✓ New security standards added → Interface extends seamlessly
```

*Result*: Agent receives coherent, synthesized guidance regardless of where knowledge is stored, and the system remains robust as storage systems evolve or knowledge sources change.

---

## CRAFT in Practice: Knowledge Assembly Pipeline

### 1. Knowledge Ingestion (Chunk + Resist)
```
Input: New team coding standard document
Process:
  1. Parse into domain-specific chunks
  2. Validate against existing standards
  3. Detect conflicts with current knowledge
  4. Version and store with provenance
Output: Versioned, conflict-free knowledge chunks
```

### 2. Task Analysis (Fit)
```
Input: User request "Build a REST API for user management"
Process:
  1. Identify required domains (backend, API design, user management)
  2. Determine task scope (new development vs. maintenance)
  3. Assess user competency level
  4. Filter knowledge to relevant pieces
Output: Focused knowledge requirements
```

### 3. Knowledge Assembly (Adapt + Transcend)
```
Input: Filtered knowledge requirements
Process:
  1. Select appropriate granularity for each domain
  2. Resolve cross-domain dependencies
  3. Synthesize coherent guidance
  4. Format for agent consumption
Output: Coherent, multi-domain context
```

### 4. Context Delivery
```
Input: Synthesized context + user query
Process:
  1. Inject context into agent prompt
  2. Execute task with assembled knowledge
  3. Capture any knowledge gaps or conflicts
  4. Update knowledge base with learnings
Output: Task completion + knowledge refinement
```

## Measuring Knowledge Assembly Effectiveness

- **Domain Coherence**: Are knowledge chunks cleanly separated by domain?
- **Temporal Consistency**: Does the agent use current, not outdated knowledge?
- **Scope Alignment**: Is knowledge complexity appropriate for task scope?
- **Synthesis Quality**: Does cross-domain knowledge combine coherently?
- **Knowledge Utilization**: Is provided knowledge actually used in the output?

CRAFT transforms knowledge management from an ad-hoc "dump everything into context" approach into a systematic discipline for curating, versioning, and assembling domain-specific knowledge that enables truly intelligent agent behavior.

---

## Knowledge Refactoring: From Monoliths to Atomic Ideas

Most existing knowledge exists as **monolithic documents**—comprehensive guides that mix multiple domains, authority levels, and granularities. The first step in applying CRAFT is **knowledge refactoring**: systematically decomposing monoliths into atomic, well-tagged ideas.

#### Chunking Taxonomies

Different taxonomies reveal different organizational principles within monolithic knowledge:

| Taxonomy | Purpose | Example Breakdown |
|----------|---------|-------------------|
| **Why/Who/What/How** | Separate purpose, audience, content, and method | Mission statement / User personas / Feature specs / Implementation guides |
| **Think/Feel/Do** | Cognitive, emotional, and behavioral aspects | Strategic reasoning / User experience principles / Actionable procedures |
| **Principles/Patterns/Practices** | Theoretical foundations, proven solutions, concrete implementations | REST principles / API design patterns / Specific endpoint examples |
| **Context/Content/Constraints** | Environmental factors, core information, limiting conditions | Market conditions / Product requirements / Technical limitations |
| **Past/Present/Future** | Historical context, current state, planned evolution | Legacy decisions / Current architecture / Migration roadmap |

#### Strategies for Discovering Inherent Taxonomies

**1. Content Analysis Approach**:
- Scan for repeated structural patterns (numbered lists, consistent headings)
- Identify different "voices" (strategic vs. tactical, authoritative vs. experimental)
- Look for natural breakpoints where topics shift domains

**2. Usage Pattern Analysis**:
- Track which sections are referenced together in practice
- Identify knowledge that's needed at different project phases
- Map knowledge to different user roles and competency levels

**3. Dependency Mapping**:
- Find knowledge that must be understood sequentially
- Identify knowledge that can stand alone
- Discover cross-references and knowledge clusters

#### Atomic Idea Profiling

Each chunked atomic idea should carry a **context profile** indicating when it applies:

```yaml
knowledge_profile:
  domains: [technical, frontend, react]
  authority: official-standard
  scope: [enterprise, production]
  audience: [senior-developer, tech-lead]
  granularity: implementation-detail
  temporal: current
  dependencies: [react-basics, component-patterns]
  conflicts: [legacy-class-components]
  confidence: high
```

**Example Refactoring**: Monolithic "Frontend Development Guide"

**Before** (monolithic):
```
frontend-guide.md:
- Our company builds user-centric products (Why)
- Frontend developers and designers use this guide (Who)
- We use React with TypeScript for all new projects (What)
- Install dependencies with npm, use hooks for state (How)
- Accessibility is important for our users (Why)
- WCAG 2.1 AA compliance required (What)
- Use aria-labels and semantic HTML (How)
- Our brand colors are #FF6B6B primary, #4ECDC4 secondary (What)
- Apply colors consistently across components (How)
```

**After** (atomic ideas with profiles):
```
why/user-centric-mission.md:
  profile: {domains: [business], authority: foundational, scope: [all]}

who/frontend-audience.md:
  profile: {domains: [process], authority: official, scope: [team]}

what/react-typescript-stack.md:
  profile: {domains: [technical], authority: standard, scope: [new-projects]}

how/react-setup-procedures.md:
  profile: {domains: [technical], granularity: step-by-step, dependencies: [react-stack]}

why/accessibility-importance.md:
  profile: {domains: [business, design], authority: foundational, scope: [all]}

what/accessibility-standards.md:
  profile: {domains: [design, compliance], authority: requirement, scope: [all]}

how/accessibility-implementation.md:
  profile: {domains: [technical, design], granularity: implementation, dependencies: [accessibility-standards]}

what/brand-color-palette.md:
  profile: {domains: [design], authority: official, scope: [all], temporal: current}

how/color-application-patterns.md:
  profile: {domains: [design, technical], granularity: implementation, dependencies: [brand-colors]}
```

This refactoring enables **precise knowledge assembly**: instead of loading the entire guide, the agent can load only relevant atomic ideas based on the current task's domain, scope, and granularity requirements.
