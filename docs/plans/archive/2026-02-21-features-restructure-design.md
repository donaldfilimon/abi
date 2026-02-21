# Feature Module Restructure Design

**Date:** 2026-02-21
**Scope:** `src/features/` full cleanup, deduplication, and consolidation
**Approach:** Full Restructure (Approach C)

## Goals

1. Delete all obsolete/old files
2. Combine similar implementations to avoid duplication
3. Consolidate AI module hierarchy
4. Fix stub parity gaps and broken tests
5. Unify shared algorithms into `services/shared/`

## Changes by Domain

### Domain 1: AI Module Consolidation (independent)

**1a. Delete old standalone AI directories**
- Commit the already-deleted `src/features/ai_core/`, `ai_inference/`, `ai_reasoning/`, `ai_training/` (git tracks as D)

**1b. Merge flat AI files into proper subdirectories**
Move these flat files into `name/mod.zig` + `name/stub.zig` pairs, eliminating the non-standard `ai/stubs/` directory:

| Current | New |
|---------|-----|
| `ai/agent.zig` + `ai/stubs/agent.zig` | `ai/agents/mod.zig` (merge with existing) + `ai/agents/stub.zig` (already exists) |
| `ai/model_registry.zig` + `ai/stubs/model_registry.zig` | `ai/models/registry.zig` (merge into existing `ai/models/`) |
| `ai/tool_agent.zig` + `ai/stubs/tool_agent.zig` | `ai/tools/tool_agent.zig` + update `ai/tools/stub.zig` |
| `ai/codebase_index.zig` + `ai/stubs/codebase_index.zig` | `ai/explore/codebase_index.zig` + update `ai/explore/stub.zig` |
| `ai/gpu_agent.zig` + `ai/stubs/gpu_agent.zig` | `ai/agents/gpu_agent.zig` + update `ai/agents/stub.zig` |
| `ai/discovery.zig` + `ai/stubs/discovery.zig` | `ai/explore/discovery.zig` + update `ai/explore/stub.zig` |

After: delete `ai/stubs/` directory entirely.

**1c. Rename `ai/core/` to avoid confusion with `abi.ai_core` facade**
- Rename `ai/core/` to `ai/types/` (contains `types.zig` and `config.zig` — primitive types for Abbey)
- Update all imports from `ai/core/mod.zig` to `ai/types/mod.zig`

**1d. Unify AI memory subsystem**
- Create `ai/memory/cognitive/` containing the current `ai/abbey/memory/` files (episodic, semantic, working)
- Keep `ai/memory/` as the single memory entry point with two sub-layers:
  - `ai/memory/chat/` (current short_term, window, summary, long_term, manager, persistence)
  - `ai/memory/cognitive/` (moved from abbey/memory — episodic, semantic, working, MemoryManager)
- Update `ai/abbey/mod.zig` to import from `ai/memory/cognitive/` instead of `abbey/memory/`
- Rename abbey's `MemoryManager` to `CognitiveMemoryManager` to avoid name collision

**1e. Fix stub parity gaps**
- `ai/stubs/model_registry.zig` → fix `ModelInfo` to include real fields with defaults
- `ai/stubs/agent.zig` → change `ConnectorNotAvailable` to `AiDisabled`
- All AI stubs: audit for `error.FeatureDisabled` consistency

**1f. Resolve StreamingGenerator name collision**
- Rename `ai/streaming/generator.zig`'s type to `StreamingPipeline` (high-level orchestrator)
- Keep `ai/llm/generation/streaming.zig`'s `StreamingGenerator` (low-level, LLM-coupled)

### Domain 2: GPU Failover Consolidation (independent)

- Make `gpu/mega/failover.zig` a thin wrapper over `gpu/failover.zig`
- The mega version adds circuit breaker overlay — keep that as an extension, but delegate core state machine to the primary `FailoverManager`
- Remove duplicated `BackendHealth`, `CircuitState` types from mega — import from primary

### Domain 3: Rate Limiter Unification (independent)

- Extract shared algorithms to `services/shared/resilience/rate_limiter.zig`:
  - `TokenBucketCore` — the refill/consume math (value type, no allocator)
  - `SlidingWindowCore` — the bucket histogram logic
  - `FixedWindowCore` — simple counter + reset
- `gateway/rate_limit.zig` wraps core with nanosecond precision, no allocator
- `network/rate_limiter.zig` wraps core with allocator, atomics, queue support, leaky bucket (stays unique)
- Pattern follows the existing `circuit_breaker.zig` unification approach

### Domain 4: Database + Network Cleanup (independent)

- Delete deprecated `database/storage.zig` (v2 reads v1 format via `MIN_READ_VERSION = 1`)
- Delete broken test in `network/failover.zig:215` (`expect(false)`)
- Update `database/mod.zig` to remove any `storage.zig` import (use only `storage_v2.zig`)

### Domain 5: Integration Layer Update (depends on all above)

After all domain changes stabilize:
- Update `src/abi.zig` imports for any moved files
- Update `src/core/framework.zig`, `context_init.zig`, `shutdown.zig` for changed module paths
- Update `src/feature_test_root.zig` for new test locations
- Update `src/core/feature_catalog.zig` parity spec paths
- Remove facades indirection: have `abi.zig` import AI sub-modules directly from their new locations
- Run `zig build validate-flags` and `zig build full-check`

## Dependency Graph

```
Domain 1 (AI)  ──┐
Domain 2 (GPU) ──┤
Domain 3 (Rate)──┼──► Domain 5 (Integration)
Domain 4 (DB)  ──┘
```

Domains 1-4 are independent and can be worked in parallel.
Domain 5 must wait for all to complete.

## Risk Mitigation

- Each domain is self-contained within its feature directory
- No feature imports another feature (enforced by import rules)
- Integration layer (Domain 5) is the only shared-state domain
- Build verification after each domain: `zig build -Denable-<feature>=true` and `=false`
- Final gate: `zig build full-check`

## Files Affected (estimated)

- ~30 files moved/renamed in AI
- ~5 files modified in GPU
- ~3 new files + 2 modified in rate limiter
- ~2 files modified in database/network
- ~6 integration files updated
- Total: ~46 files touched
