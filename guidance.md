# Context-Mixer Development Guidance

Based on the comprehensive analysis of PLAN.md and the current codebase, here's your clear guidance on what to do next:

## 🎯 **CRITICAL PRIORITY: Project Context Isolation**

**Why this is now the top priority:**
- **CRITICAL GAP IDENTIFIED:** Current system mixes knowledge from different projects without context tracking
- **Cross-project contamination risk:** Guidance from different projects gets mixed inappropriately
- **User requirement:** Users need to choose which project contexts to include when assembling
- **Foundation requirement:** Must be implemented before assemble command to prevent knowledge interference

**Immediate Action:** Implement project-aware capabilities in domain models and CLI commands

## 🎯 **SECONDARY PRIORITY: Complete the Assemble Command**

**Why this follows project isolation:**
- ChunkingEngine is now fully implemented ✅
- Knowledge store infrastructure is complete ✅  
- Must include project-aware filtering to prevent cross-project contamination
- This will make the CLI fully functional and demonstrate the CRAFT system working end-to-end

**Immediate Action:** Implement `src/context_mixer/commands/assemble.py` with project-aware filtering

## Current Status Summary

**✅ MAJOR ACHIEVEMENTS COMPLETED:**
- Complete CRAFT domain models (KnowledgeChunk, ChunkMetadata, AuthorityLevel, etc.)
- Vector storage infrastructure with ChromaDB integration
- Knowledge store architecture with abstract interfaces
- Conflict detection and semantic search capabilities
- Basic CLI commands (init, ingest, slice, open)
- Comprehensive testing infrastructure

**🎯 PHASE 1 CRITICAL GAPS TO COMPLETE:**

### 0. **Project Context Isolation** (CRITICAL PRIORITY)
**Status:** ❌ **MISSING** - Identified as critical gap preventing multi-project usage
**Impact:** Cross-project knowledge contamination, inability to select project contexts
**Action:** Implement project-aware capabilities across the system:
```python
# Enhanced domain models
class ProvenanceInfo(BaseModel):
    project_id: Optional[str] = Field(None, description="Project identifier")
    project_name: Optional[str] = Field(None, description="Human-readable project name")
    project_path: Optional[str] = Field(None, description="Root path of the source project")

# Enhanced search capabilities
class SearchQuery(BaseModel):
    project_ids: Optional[List[str]] = Field(None, description="Filter by specific projects")
    exclude_projects: Optional[List[str]] = Field(None, description="Exclude specific projects")
```
**CLI Commands:**
```bash
cmx ingest --project-id "react-frontend" --project-name "React Frontend App"
cmx assemble --project-ids "react-frontend" --exclude-projects "legacy-system"
cmx projects --list --show-stats
cmx context-select --projects "react-frontend,python-api" --output copilot-context.md
```

### 1. **ChunkingEngine with Semantic Boundary Detection** ✅ **COMPLETED**
**Status:** ✅ **IMPLEMENTED** - Full implementation with 988 lines of code
**Impact:** Intelligent concept-based chunking now available with semantic boundary detection
**Completed:** `src/context_mixer/domain/chunking_engine.py` with comprehensive functionality:
```python
class ChunkingEngine:
    def detect_semantic_boundaries(self, content: str) -> List[ChunkBoundary]  ✅
    def chunk_by_concepts(self, content: str) -> List[KnowledgeChunk]  ✅
    def validate_chunk_completeness(self, chunk: KnowledgeChunk) -> bool  ✅
    # Plus many additional methods for hierarchical detection, unit analysis, etc.
```
**Evidence:** Working examples in `src/_examples/` demonstrate the system in action

### 2. **KnowledgeQuarantine System** (HIGH PRIORITY)
**Status:** Missing - marked as `[ ]` in plan  
**Impact:** No isolation mechanism for conflicting knowledge
**Action:** Implement `src/context_mixer/domain/knowledge_quarantine.py`
```python
class KnowledgeQuarantine:
    def quarantine_chunk(self, chunk: KnowledgeChunk, reason: str)
    def review_quarantined_chunks(self) -> List[QuarantinedChunk]
    def resolve_quarantine(self, chunk_id: str, resolution: Resolution)
```

### 3. **Migration Tool for Existing Fragments** (MEDIUM PRIORITY)
**Status:** Missing - marked as `[ ]` in plan
**Impact:** Cannot migrate existing flat-file fragments to new CRAFT structure
**Action:** Implement `src/context_mixer/commands/migrate.py`

### 4. **Enhanced CLI Commands** (MEDIUM PRIORITY)
**Status:** Basic commands exist, but missing CRAFT-aware features
**Current Gap:** The `assemble` command shows "not yet implemented"
**Action:** Enhance existing commands with CRAFT parameters:
```bash
cmx ingest --detect-boundaries --authority-level official
cmx slice --granularity detailed --domains technical,business  
cmx assemble --target copilot --token-budget 8192 --quality-threshold 0.8
```

## **IMMEDIATE NEXT STEPS (Priority Order):**

### **STEP 0: Implement Project Context Isolation (CRITICAL PRIORITY)**
This foundation prevents cross-project knowledge contamination:
1. Enhance `ProvenanceInfo` domain model with project context fields
2. Update `SearchQuery` to support project filtering parameters
3. Modify `ChunkingEngine` to accept project identification during ingestion
4. Update CLI commands to support project-aware parameters
5. Implement `ProjectDetector` for automatic project identification
6. Create `ProjectContextManager` for project-scoped operations
7. Add project-aware filtering to knowledge store operations

**Key Implementation Points:**
- Update `src/context_mixer/domain/knowledge.py` with enhanced ProvenanceInfo
- Modify `src/context_mixer/commands/ingest.py` to accept project parameters
- Enhance `src/context_mixer/domain/vector_knowledge_store.py` for project filtering
- Update CLI interface in `src/context_mixer/cli.py` with project options

### **STEP 1: Implement ChunkingEngine (Week 1)** ✅ **COMPLETED**
This foundation for intelligent knowledge processing is now complete:
1. ✅ Create `src/context_mixer/domain/chunking_engine.py` (988 lines)
2. ✅ Implement semantic boundary detection using NLP
3. ✅ Add concept-based chunking logic
4. ✅ Create validation for chunk completeness
5. ✅ Write comprehensive tests in `chunking_engine_spec.py` (18,903 bytes)

### **STEP 2: Complete Assemble Command with Project Awareness (SECONDARY PRIORITY)**
Make the CLI fully functional and demonstrate end-to-end CRAFT workflow:
1. Create `src/context_mixer/commands/assemble.py`
2. Replace the "not yet implemented" placeholder in `cli.py`
3. Integrate with existing ChunkingEngine and KnowledgeStore
4. Add CRAFT-aware context assembly logic
5. Support multiple target formats (copilot, claude, etc.)
6. Add token budget optimization using existing infrastructure

**Key Integration Points:**
- Use `VectorKnowledgeStore` for semantic retrieval
- Leverage `ChunkingEngine` for intelligent content processing
- Apply authority-level filtering and granularity selection
- Output formatted context suitable for AI assistants

### **STEP 3: Build KnowledgeQuarantine (Week 3)**
Essential for conflict management (can be done in parallel):
1. Create `src/context_mixer/domain/knowledge_quarantine.py`
2. Implement quarantine storage and retrieval
3. Add conflict resolution workflows
4. Integrate with existing conflict detection system
5. Add CLI commands for quarantine management

### **STEP 4: Enhanced CLI Parameters (Week 4)**
Add CRAFT-aware parameters to existing commands:
1. Update `ingest` command with boundary detection
2. Add granularity and domain filtering to `slice`
3. Implement authority-level filtering across commands
4. Add comprehensive help and examples

## **SUCCESS CRITERIA FOR PHASE 1 COMPLETION:**

- [ ] **Project Context Isolation implemented** - Users can choose which project contexts to include
- [ ] **Project-aware ingestion** - CLI accepts project identification parameters
- [ ] **Project-scoped knowledge retrieval** - Search and assembly respect project boundaries
- [ ] **Cross-project contamination prevention** - Knowledge from different projects doesn't mix inappropriately
- [x] ✅ ChunkingEngine can detect semantic boundaries in real content
- [ ] KnowledgeQuarantine can isolate and manage conflicting chunks
- [ ] Migration tool can convert existing flat files to CRAFT structure
- [ ] All CLI commands support CRAFT and project-aware parameters
- [ ] Assemble command produces working context for AI assistants with project filtering
- [x] ✅ All new functionality has comprehensive tests (ChunkingEngine fully tested)
- [ ] Documentation updated with new capabilities including project-aware usage

## **AFTER PHASE 1 - PHASE 2 PREVIEW:**

Once Phase 1 is complete, Phase 2 focuses on:
- Multi-resolution knowledge storage (summary → comprehensive)
- Dynamic context window management
- Query analysis and intent recognition
- Advanced relevance scoring

## **DEVELOPMENT APPROACH:**

1. **Start Small:** Implement ChunkingEngine with basic semantic detection first
2. **Test-Driven:** Write tests for each component before implementation
3. **Incremental:** Each component should work independently
4. **Integration:** Test components together as you build them
5. **Documentation:** Update README and docs as you implement features

## **TECHNICAL NOTES:**

- Use existing domain models (KnowledgeChunk, ChunkMetadata) - they're complete
- Leverage existing vector store infrastructure - it's working well
- Follow the Gateway pattern for I/O operations
- Use pydantic for all new data models
- Co-locate tests with implementation files

**The foundation is solid - now it's time to build the intelligent chunking and conflict management layers that will make Context-Mixer truly powerful.**
