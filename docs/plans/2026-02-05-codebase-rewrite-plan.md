# ABI Framework — Comprehensive Rewrite Plan

**Date:** 2026-02-05
**Focus:** Modularity, cleanliness, reduced bloat
**Current state:** 735 .zig files in src/, 303K lines, 917/922 tests pass

---

## 1. Codebase Metrics (Current)

| Area | Files | Lines | Assessment |
|------|-------|-------|------------|
| src/features/ai/ | 255 | 104,330 | **Largest — needs decomposition** |
| src/features/gpu/ | 157 | 67,885 | Large but well-structured (vtable pattern) |
| src/features/database/ | 45 | 24,528 | Reasonable |
| src/features/network/ | 30 | 14,879 | Reasonable |
| src/features/web/ | 20 | 5,543 | Small |
| src/features/observability/ | 21 | 4,414 | Small |
| src/features/analytics/ | 2 | 495 | New, minimal |
| src/core/ | 29 | ~8,000 | Config/registry/flags |
| src/services/ | 184 | ~42,000 | Shared infra + tests |
| tools/ | 52 | ~15,000 | CLI + TUI |
| benchmarks/ | 43 | ~14,000 | Performance harness |
| docs/ | 27 md | 4,793 | API docs + plans |
| Root .md | 16 | 4,414 | Too many, overlap |

**Top 10 largest files (>1,300 lines) — split candidates:**
1. `ai/training/trainable_model.zig` (2,398)
2. `connectors/discord/rest.zig` (1,680)
3. `ai/training/self_learning.zig` (1,642)
4. `database/hnsw.zig` (1,599)
5. `shared/simd.zig` (1,598)
6. `network/raft.zig` (1,553)
7. `tests/observability_test.zig` (1,489)
8. `ai/streaming/server.zig` (1,485)
9. `tests/integration/c_api_test.zig` (1,445)
10. `gpu/backends/vulkan.zig` (1,423)

---

## 2. Structural Problems

### 2.1 src/abi.zig is overloaded
- 572 lines acting as both public API and legacy compatibility layer
- 60+ `pub const` re-exports for backward compatibility
- Functions like `createDefaultFramework` / `createFramework` are deprecated wrappers
- **Fix:** Split into `src/abi.zig` (lean public API) + `src/compat.zig` (legacy re-exports)

### 2.2 AI module is 34% of codebase
- 255 files, 104K lines — more than GPU + database + network combined
- 17 sub-modules, many are stubs
- The abbey/ sub-module alone has meta-learning, theory of mind, neural attention
- **Fix:** Promote AI sub-modules to separate feature modules where they're large enough

### 2.3 Root markdown proliferation
- 16 .md files at root: README, CLAUDE, AGENTS, CONTRIBUTING, PLAN, ROADMAP, TODO, CHANGELOG, QUICKSTART, API_REFERENCE, CODEBASE_IMPROVEMENTS, DEPLOYMENT_GUIDE, SECURITY, PROMPT, GEMINI, CODE_OF_CONDUCT
- Significant overlap (PLAN vs ROADMAP vs TODO vs CODEBASE_IMPROVEMENTS)
- **Fix:** Keep 5 essential files, move rest to docs/

### 2.4 Services grab-bag
- `src/services/` contains: runtime, platform, shared, ha, tasks, tests, connectors
- "shared" has 15+ modules (simd, security, logging, plugins, utils, os, time...)
- connectors/discord doesn't belong with infrastructure
- **Fix:** Reorganize into cleaner layers

### 2.5 Test organization
- Tests scattered: inline (good), `src/services/tests/` (monolith), feature-specific
- No clear test category boundaries
- Property tests duplicated between two systems
- **Fix:** Co-locate tests with modules, keep integration tests separate

---

## 3. Proposed New Structure

```
src/
├── abi.zig                    # Lean public API (< 200 lines)
├── compat.zig                 # Legacy re-exports (deprecated)
│
├── core/                      # Framework infrastructure
│   ├── framework.zig          # Lifecycle state machine
│   ├── config.zig             # Unified config (flatten config/)
│   ├── flags.zig              # Build flags
│   └── registry.zig           # Feature registry (flatten registry/)
│
├── features/                  # Feature modules (each: mod.zig + stub.zig)
│   ├── gpu/                   # GPU compute (keep current structure)
│   │   ├── mod.zig
│   │   ├── stub.zig
│   │   ├── backends/          # VTable implementations
│   │   ├── dsl/               # Kernel DSL
│   │   └── mega/              # Multi-GPU
│   │
│   ├── ai/                    # Core AI (reduced scope)
│   │   ├── mod.zig            # Transformer, prompts, tools
│   │   ├── stub.zig
│   │   ├── core/              # Core AI abstractions
│   │   └── abbey/             # Advanced reasoning
│   │
│   ├── llm/                   # LLM inference (promoted from ai/)
│   │   ├── mod.zig
│   │   ├── stub.zig
│   │   └── ops/               # GPU memory pool, attention, etc.
│   │
│   ├── agents/                # Agent framework (promoted from ai/)
│   │   ├── mod.zig
│   │   └── stub.zig
│   │
│   ├── embeddings/            # Embedding generation (promoted)
│   │   ├── mod.zig
│   │   └── stub.zig
│   │
│   ├── training/              # Model training (promoted)
│   │   ├── mod.zig
│   │   └── stub.zig
│   │
│   ├── database/              # Vector database (keep)
│   ├── network/               # Distributed networking (keep)
│   ├── observability/         # Monitoring/profiling (keep)
│   ├── web/                   # Web/HTTP (keep)
│   └── analytics/             # Analytics/experiments (keep)
│
├── infra/                     # Shared infrastructure (rename from services/)
│   ├── runtime/               # Runtime engine
│   ├── platform/              # Platform detection
│   ├── simd.zig               # SIMD operations
│   ├── time.zig               # Time utilities
│   ├── logging.zig            # Logging
│   ├── security/              # Security modules
│   └── plugins.zig            # Plugin system
│
├── connectors/                # External service connectors (promoted)
│   ├── discord/
│   ├── openai/
│   ├── anthropic/
│   └── ollama/
│
└── tests/                     # Integration/E2E tests only
    ├── mod.zig                # Test root
    ├── parity/                # Stub parity tests
    ├── integration/           # Cross-module integration
    ├── stress/                # Load/stress tests
    └── property/              # Property-based tests
```

### Key changes:
1. **AI decomposition**: ai/, llm/, agents/, embeddings/, training/ become separate features
2. **Services → infra**: Cleaner name, flat structure
3. **Connectors promoted**: Not buried under services/
4. **Tests co-located**: Unit tests stay inline, only integration/cross-cutting in tests/
5. **abi.zig slimmed**: < 200 lines, no legacy wrappers

---

## 4. Root Markdown Consolidation

**Keep (5 files):**
- `README.md` — Project overview + quickstart
- `CLAUDE.md` — AI assistant context
- `CONTRIBUTING.md` — Dev workflow
- `SECURITY.md` — Security practices
- `CODE_OF_CONDUCT.md` — Community standards

**Move to docs/:**
- `AGENTS.md` → `docs/agents.md`
- `DEPLOYMENT_GUIDE.md` → `docs/deployment.md`
- `API_REFERENCE.md` → `docs/api/reference.md` (merge with docs/api/)

**Delete (consolidated into README or docs/):**
- `PLAN.md` (merge into ROADMAP)
- `ROADMAP.md` → `docs/roadmap.md`
- `TODO.md` (use GitHub issues)
- `QUICKSTART.md` (merge into README)
- `CHANGELOG_CONSOLIDATED.md` → `CHANGELOG.md`
- `CODEBASE_IMPROVEMENTS.md` (done, archive)
- `PROMPT.md` (merge into CLAUDE.md)
- `GEMINI.md` (merge into CLAUDE.md or delete)

---

## 5. Benchmark Rewrite

**Current problems:**
- 43 files, heavy abstraction layer
- `system/framework.zig` at 805 lines is over-engineered for a benchmark runner
- `system/baseline_store.zig` at 677 lines for regression tracking
- Competitive comparison framework adds complexity rarely used

**Proposed structure:**
```
benchmarks/
├── main.zig              # Entry point (< 100 lines)
├── runner.zig            # Core benchmark runner (< 300 lines)
├── stats.zig             # Statistical analysis (warm-up, percentiles)
├── baseline.zig          # Regression tracking (simple JSON store)
├── suites/
│   ├── simd.zig          # SIMD benchmarks
│   ├── memory.zig        # Memory allocation
│   ├── concurrency.zig   # Thread/async performance
│   ├── gpu.zig           # GPU kernel execution
│   ├── database.zig      # Vector DB operations
│   ├── ai.zig            # AI inference
│   ├── network.zig       # Network I/O
│   └── crypto.zig        # Crypto operations
└── results/              # Baseline results (gitignored)
```

---

## 6. Test Rewrite

**Current problems:**
- `src/services/tests/` is a monolith with mixed test types
- Property test duplication (proptest.zig legacy + property/mod.zig new)
- Large test files (>1,000 lines)

**Proposed pattern:**
- Unit tests: `*_test.zig` alongside source (already done for many)
- Feature integration: `src/features/<name>/tests/`
- Cross-feature integration: `src/tests/integration/`
- Parity tests: Auto-generated from module declarations
- Property tests: Consolidated into single framework

---

## 7. Execution Order

### Phase 1: Foundation (non-breaking)
1. Slim down `src/abi.zig` — extract legacy re-exports to `compat.zig`
2. Flatten `src/core/config/` and `src/core/registry/`
3. Consolidate root markdown files

### Phase 2: Feature decomposition
4. Promote `ai/llm/` to `features/llm/` (new feature flag)
5. Promote `ai/agents/` to `features/agents/`
6. Promote `ai/embeddings/` to `features/embeddings/`
7. Promote `ai/training/` to `features/training/`
8. Move `services/connectors/` to `src/connectors/`

### Phase 3: Infrastructure cleanup
9. Rename `services/` to `infra/`
10. Flatten `services/shared/` into `infra/`
11. Co-locate unit tests with source files
12. Restructure `src/services/tests/` into `src/tests/`

### Phase 4: Benchmarks & docs
13. Rewrite benchmark runner (simpler, flatter)
14. Rewrite benchmark suites (one per domain)
15. Rebuild docs/ API reference from source

### Phase 5: Verification
16. Run full test suite after each phase
17. Verify all feature flag combinations compile
18. Run benchmarks to confirm no regressions
19. Update CLAUDE.md with new structure
