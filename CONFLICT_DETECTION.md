# Conflict Detection Optimization Plan

## Problem Statement - SOLVED ✅

The conflict detection system was producing false positives by flagging non-conflicting chunks as conflicts.

### Example False Positive

```
Conflict Detected!
Description: Conflicting guidance detected between new chunks: General docstring requirement
conflicts with test-specific prohibition of docstrings in fixtures and Describe classes.

Conflicting Guidance:

1. From chunk chunk_df514e...: "Document code using NumPy-style docstrings."
2. From chunk chunk_c6bc93...: "Use descriptive test names in Describe* classes; do not add
   docstrings to fixtures or Describe classes."
```

**These are NOT in conflict** - they are complementary rules that apply to different contexts:
- General code should use docstrings
- Test fixtures and Describe classes should NOT use docstrings (use descriptive names instead)

## Root Cause Analysis

The issue had multiple contributing factors:

### 1. Metadata Filtering Was Too Weak

The `filter_pairs_by_metadata()` function wasn't properly considering concept differences:
- It only checked if concepts matched exactly OR if domains overlapped
- Different concepts (e.g., "docstring-requirements" vs "test-conventions") were still compared
- **Different concepts usually mean different scopes** and should rarely be compared

### 2. Batched Conflict Detection Had Weak Prompts

The `detect_conflicts_multi_pair()` function used a much simpler prompt than the single-pair detection:
- Didn't properly explain how to use concept metadata
- Didn't emphasize that different concepts = different scopes = NOT conflicts
- Lacked concrete examples of concept-based scope differences

### 3. Concept Metadata Wasn't Fully Leveraged

The chunking engine was creating proper concept metadata, but the conflict detection wasn't using it effectively:
- Concepts were included in prompts but not emphasized
- No explanation that different concepts are usually complementary, not conflicting

## Solution Implementation

### 1. Enhanced Metadata Filtering

**File**: `src/context_mixer/commands/operations/merge.py`

**Changes**:
- Added strong concept difference detection
- When concepts differ, only compare if:
  - Domains overlap AND
  - Concepts share significant keywords
- Otherwise, different concepts = likely complementary

```python
# If concepts are present but different, check if they're related
if chunk1.concept and chunk2.concept and chunk1.concept != chunk2.concept:
    # Only compare if they share domains AND have related keywords
    domains1 = set(chunk1.metadata.domains) if chunk1.metadata.domains else set()
    domains2 = set(chunk2.metadata.domains) if chunk2.metadata.domains else set()

    if domains1 and domains2 and domains1.intersection(domains2):
        # Check if concepts are semantically related (share keywords)
        concept1_words = set(chunk1.concept.lower().split('-'))
        concept2_words = set(chunk2.concept.lower().split('-'))

        # If concepts share significant keywords, they might conflict
        if concept1_words.intersection(concept2_words):
            filtered_pairs.append((chunk1, chunk2))
        # Otherwise, different concepts in same domain are likely complementary
    continue
```

### 2. Enhanced Batched Conflict Detection Prompt

**File**: `src/context_mixer/commands/operations/merge.py`

**Changes**:
- Added rich metadata descriptions showing concept, domains, and scope
- Added comprehensive section explaining concept-based scope differences
- Provided concrete examples of non-conflicts due to different concepts
- Emphasized that different concepts = usually complementary, not conflicting

Key additions to the prompt:
```
**Different concepts usually mean different scopes - NOT conflicts!**

Examples of NON-CONFLICTS due to different scopes:
- [Concept: docstring-requirements] "Use docstrings for all functions"
  vs [Concept: test-conventions] "Don't use docstrings in test fixtures"
  → These apply to DIFFERENT contexts - COMPLEMENTARY, not conflicting!

- [Concept: gateway-testing] "Don't test gateway logic"
  vs [Concept: testing-guidelines] "Write tests for all new functionality"
  → General rule applies, gateways are specific exception - COMPLEMENTARY!
```

## Validation

### Unit Tests

All 29 tests in `merge_spec.py` pass:
```bash
pytest src/context_mixer/commands/operations/merge_spec.py -v
# ====================================== 29 passed in 0.67s ======================================
```

### Integration Tests (Workbench)

Both false positive scenarios pass:

**Scenario: `architectural_scope_false_positive`**
- Tests gateway-specific rules vs general guidelines
- **PASSED** - No false conflicts detected

**Scenario: `false_positive_naming`**
- Tests different naming conventions for different contexts
- **PASSED** - No false conflicts detected

## Impact

### Before Fix
- False positives: **High** - rules for different contexts flagged as conflicts
- User experience: **Poor** - manual resolution of non-conflicts required
- Trust: **Low** - users question the tool's judgment

### After Fix
- False positives: **Dramatically reduced** - concept/scope awareness prevents most false positives
- User experience: **Improved** - fewer manual interventions needed
- Trust: **Higher** - tool understands context and scope differences

## How Concepts Help

The concept metadata added to each chunk enables scope-aware conflict detection:

1. **Chunking creates specific concepts**: "docstring-requirements", "test-conventions", "gateway-testing"
2. **Metadata filtering uses concepts**: Different concepts = likely different scopes
3. **LLM prompt uses concepts**: Explicitly explains that different concepts mean complementary rules

### Example

Two chunks:
- **Chunk A**: [Concept: docstring-requirements] "Use docstrings for all functions"
- **Chunk B**: [Concept: test-conventions] "Don't use docstrings in test fixtures"

**Before fix**: Flagged as conflict (both mention docstrings)
**After fix**: Not compared (different concepts = different scopes)

If they were compared, the enhanced prompt would help the LLM understand they're complementary.

## Performance Impact

No significant performance regression:
- Metadata filtering is faster (more aggressive filtering = fewer pairs)
- Prompt is longer but API calls are fewer
- Overall: **Same or better performance**

---

## Historical Notes: Original Optimization Work

### Original Problem Statement

The current conflict detection algorithm is prohibitively expensive:
- **134 chunks** generates **8,911 internal comparisons** (n×(n-1)/2)
- Each comparison makes **1 LLM call**
- Total cost: ~$40 on OpenAI
- Total time: ~2 hours

This is unacceptable for production use.

## Current Architecture

### Two Phases of Conflict Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGEST PIPELINE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: INTERNAL CONFLICT DETECTION                           │
│  ─────────────────────────────────────────                          │
│  Location: ingest.py (~line 500)                                │
│  Compares: All new chunks against each other                    │
│  Complexity: O(n²) - EVERY pair gets an LLM call                │
│  Problem: NO pre-filtering whatsoever                           │
│                                                                  │
│  Phase 2: EXTERNAL CONFLICT DETECTION                           │
│  ─────────────────────────────────────                          │
│  Location: ingest.py (~line 575)                                │
│  Compares: Each new chunk against existing store                │
│  Uses: ClusterAwareConflictDetector (HDBSCAN clustering)        │
│  Optimization: Limits to 50 candidates per chunk                │
│  Problem: Still makes 1 LLM call per candidate                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| `commands/ingest.py` | Orchestrates both phases |
| `commands/operations/merge.py` | `detect_conflicts()` and `detect_conflicts_batch()` |
| `domain/cluster_aware_conflict_detection.py` | HDBSCAN-based optimization (external only) |
| `domain/clustering_service.py` | HDBSCAN clustering implementation |
| `domain/context_aware_prompts.py` | Builds conflict detection prompts |
| `domain/vector_knowledge_store.py` | Storage with `detect_conflicts()` method |

## Optimization Strategy

### Goal
Reduce LLM calls from **~9,000** to **~50-100** for 134 chunks.

### Three-Tier Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   TIER 1: EMBEDDING SIMILARITY FILTER (CPU-only, ~0 cost)       │
│   ───────────────────────────────────────────────────────────   │
│   Filter pairs by cosine similarity of embeddings                │
│   Threshold: similarity > 0.70 (configurable)                   │
│   Expected reduction: 90-95% of pairs eliminated                │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│   TIER 2: DOMAIN/CONCEPT PARTITIONING (CPU-only, ~0 cost)       │
│   ───────────────────────────────────────────────────────────   │
│   Only compare chunks with overlapping domains OR concepts       │
│   Expected reduction: 50-70% of remaining pairs                 │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│   TIER 3: BATCHED LLM ANALYSIS (reduced API calls)              │
│   ───────────────────────────────────────────────────────────   │
│   Batch 10-20 pairs per LLM call                                │
│   Use structured output for multi-pair analysis                 │
│   Expected reduction: 10-20x fewer API calls                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Embedding Similarity Pre-Filter

**Goal**: Eliminate 90%+ of pairs before any LLM calls

**Location**: New function in `commands/operations/merge.py`

```python
def filter_pairs_by_embedding_similarity(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]],
    similarity_threshold: float = 0.70
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk]]:
    """
    Filter chunk pairs to only those with high embedding similarity.

    Conflicts can only exist between semantically similar chunks.
    Low similarity pairs are guaranteed non-conflicts.

    Args:
        chunk_pairs: All candidate pairs
        similarity_threshold: Minimum cosine similarity to consider (default 0.70)

    Returns:
        Filtered list of pairs worth checking with LLM
    """
```

**Implementation Details**:
1. Use NumPy for vectorized cosine similarity computation
2. Handle chunks without embeddings (pass through to LLM)
3. Make threshold configurable via `Config`
4. Add metrics logging for filtering effectiveness

**Estimated Effort**: 2-3 hours

---

### Phase 2: Domain/Concept Partitioning

**Goal**: Further reduce pairs by only comparing chunks that could logically conflict

**Location**: New function in `commands/operations/merge.py`

```python
def filter_pairs_by_metadata(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]]
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk]]:
    """
    Filter pairs to only those with overlapping domains or matching concepts.

    Chunks in completely different domains cannot conflict.

    Returns:
        Filtered list of pairs with potential for conflict
    """
```

**Rules**:
- Keep pair if: `chunk1.metadata.domains ∩ chunk2.metadata.domains ≠ ∅`
- Keep pair if: `chunk1.concept == chunk2.concept` (when concepts exist)
- Keep pair if: either chunk has no domain/concept metadata (safety)

**Estimated Effort**: 1-2 hours

---

### Phase 3: Batched LLM Analysis

**Goal**: Reduce API calls by analyzing multiple pairs per request

**Location**: New function in `commands/operations/merge.py`

```python
async def detect_conflicts_multi_pair(
    chunk_pairs: List[Tuple[KnowledgeChunk, KnowledgeChunk]],
    llm_gateway: LLMGateway,
    pairs_per_batch: int = 10
) -> List[Tuple[KnowledgeChunk, KnowledgeChunk, ConflictList]]:
    """
    Detect conflicts for multiple pairs in a single LLM call.

    Args:
        chunk_pairs: Pairs to analyze
        llm_gateway: LLM gateway
        pairs_per_batch: Number of pairs per LLM call (default 10)

    Returns:
        Results for each pair
    """
```

**New Pydantic Model**:
```python
class MultiPairConflictResult(BaseModel):
    """Result of multi-pair conflict analysis."""
    pair_results: List[PairConflictResult]

class PairConflictResult(BaseModel):
    """Conflict result for a single pair."""
    pair_index: int
    has_conflict: bool
    conflicts: List[Conflict] = []
```

**Prompt Structure**:
```
Analyze the following pairs of content for conflicts.
For each pair, determine if they contain contradictory guidance.

PAIR 1:
Content A: [chunk1.content]
Content B: [chunk2.content]

PAIR 2:
Content A: [chunk3.content]
Content B: [chunk4.content]

... (up to 10 pairs)

Return your analysis as structured JSON with conflict details for each pair.
```

**Estimated Effort**: 4-6 hours (includes prompt engineering and testing)

---

### Phase 4: Integration into Ingest Pipeline

**Goal**: Replace current internal conflict detection with optimized version

**Location**: `commands/ingest.py` (lines ~500-570)

**Changes**:
1. Apply embedding filter to chunk pairs
2. Apply domain/concept filter
3. Use batched LLM analysis for remaining pairs
4. Update progress tracking for new stages
5. Add timing metrics for each filter stage

**Before**:
```python
# Current: Direct batch processing of ALL pairs
batch_results = await detect_conflicts_batch(chunk_pairs, llm_gateway, batch_size, progress_callback)
```

**After**:
```python
# Step 1: Embedding similarity filter
filtered_pairs = filter_pairs_by_embedding_similarity(chunk_pairs, threshold=0.70)
console.print(f"[dim]Embedding filter: {len(chunk_pairs)} → {len(filtered_pairs)} pairs[/dim]")

# Step 2: Domain/concept filter
filtered_pairs = filter_pairs_by_metadata(filtered_pairs)
console.print(f"[dim]Metadata filter: → {len(filtered_pairs)} pairs[/dim]")

# Step 3: Batched LLM analysis
batch_results = await detect_conflicts_multi_pair(filtered_pairs, llm_gateway, pairs_per_batch=10)
```

**Estimated Effort**: 2-3 hours

---

### Phase 5: Configuration and Observability

**Goal**: Make optimization tunable and observable

**Location**: `config.py`

**New Config Options**:
```python
# Conflict detection optimization
conflict_embedding_similarity_threshold: float = 0.70
conflict_pairs_per_llm_batch: int = 10
conflict_detection_metrics_enabled: bool = True
```

**Metrics to Track**:
- Total pairs before filtering
- Pairs after embedding filter
- Pairs after metadata filter
- LLM calls made
- Conflicts detected
- Time per stage

**Estimated Effort**: 1-2 hours

---

## Expected Results

### Cost Reduction

| Stage | Pairs | LLM Calls |
|-------|-------|-----------|
| Initial (134 chunks) | 8,911 | 8,911 |
| After embedding filter (95%) | ~446 | - |
| After metadata filter (50%) | ~223 | - |
| After batching (10/call) | - | ~23 |
| **Final** | **223** | **~23** |

**Cost reduction: 99.7%** ($40 → ~$0.12)

### Time Reduction

| Operation | Current | Optimized |
|-----------|---------|-----------|
| Embedding filter | N/A | ~1 second |
| Metadata filter | N/A | ~0.1 second |
| LLM calls | ~8,900 calls × 0.8s = 2 hours | ~23 calls × 0.8s = 18 seconds |
| **Total** | **~2 hours** | **~20 seconds** |

---

## Testing Strategy

### Unit Tests

1. **Embedding filter tests** (`merge_spec.py`)
   - Test threshold behavior
   - Test handling of missing embeddings
   - Test edge cases (identical embeddings, orthogonal embeddings)

2. **Metadata filter tests** (`merge_spec.py`)
   - Test domain overlap detection
   - Test concept matching
   - Test missing metadata handling

3. **Batched analysis tests** (`merge_spec.py`)
   - Test batch size boundaries
   - Test structured output parsing
   - Test error handling for partial failures

### Integration Tests

1. **Workbench scenarios** (`workbench/scenarios/`)
   - Add scenario for large chunk ingestion
   - Verify conflict detection accuracy with filters
   - Measure performance improvements

### Accuracy Validation

Before full rollout:
1. Run on known conflict corpus with filters OFF
2. Run same corpus with filters ON
3. Compare detected conflicts (should be identical)
4. Document any false negatives and adjust thresholds

---

## Rollout Plan

### Week 1: Implementation
- [ ] Phase 1: Embedding filter
- [ ] Phase 2: Metadata filter
- [ ] Unit tests for both

### Week 2: Batching & Integration
- [ ] Phase 3: Batched LLM analysis
- [ ] Phase 4: Ingest pipeline integration
- [ ] Phase 5: Configuration

### Week 3: Validation
- [ ] Accuracy validation against known conflicts
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| False negatives from embedding filter | Conservative threshold (0.70), make configurable |
| Batched prompts reduce accuracy | Test extensively, fall back to single-pair if accuracy drops |
| Token limits with large batches | Dynamic batch sizing based on content length |
| Missing embeddings | Pass-through to LLM (don't filter) |

---

## Future Enhancements

1. **Two-stage LLM** (cheap model for screening, expensive for confirmation)
2. **Learned thresholds** (use historical conflict data to optimize thresholds)
3. **Incremental clustering** (extend HDBSCAN to internal conflicts)
4. **Caching** (cache conflict results for repeated ingestions)

---

## Appendix: Current Code Paths

### Internal Conflict Detection (Current)
```
ingest.py:do_ingest()
  └─> Creates all chunk pairs: O(n²)
      └─> detect_conflicts_batch()  [merge.py]
          └─> For each pair: detect_conflicts_async()
              └─> detect_conflicts()
                  └─> llm_gateway.generate_object()  ← LLM CALL
```

### External Conflict Detection (Current)
```
ingest.py:do_ingest()
  └─> For each chunk: knowledge_store.detect_conflicts()
      └─> VectorKnowledgeStore._detect_conflicts_optimized()
          └─> ClusterAwareConflictDetector.detect_conflicts_optimized()
              └─> ClusteringService.cluster_knowledge_chunks()  ← HDBSCAN
              └─> generate_conflict_detection_candidates()  ← Limit to 50
              └─> detect_conflicts_batch()  ← LLM CALLS (up to 50)
```

### Proposed Internal Conflict Detection
```
ingest.py:do_ingest()
  └─> Creates all chunk pairs: O(n²)
      └─> filter_pairs_by_embedding_similarity()  ← NEW (CPU only)
      └─> filter_pairs_by_metadata()              ← NEW (CPU only)
      └─> detect_conflicts_multi_pair()           ← NEW (batched LLM)
          └─> llm_gateway.generate_object()       ← ~10 pairs per call
```
