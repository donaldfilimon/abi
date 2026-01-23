---
title: "2026-01-17-refactor-phase2"
tags: []
---
# ABI Framework Refactoring - Phase 2
> **Codebase Status:** Synced with repository as of 2026-01-18.

**Date:** 2026-01-17  
**Status:** Ready for Implementation  
**Prerequisite:** Phase 1 (Runtime Consolidation) - Complete

## Overview

Continue the modular architecture refactoring with focus on:
1. Observability module consolidation
2. Core module evaluation
3. Features module cleanup
4. Documentation of legacy patterns

**Scope:** Medium complexity, moderate risk

---

## Task 1: Consolidate Observability Module

### Goal
Unify the three observability implementations into a single coherent module.

### Current State
- `src/observability/mod.zig` - Wrapper with Context struct (minimal)
- `src/shared/observability/` - Tracing infrastructure
- `src/features/monitoring/` - Metrics (Prometheus, OpenTelemetry, StatsD)

### Steps

#### 1.1 Review monitoring usage
```bash
grep -r "features/monitoring" src/ --include="*.zig" | head -20
```
Identify what imports from `features/monitoring/`.

#### 1.2 Update observability/mod.zig
Update to properly export monitoring features:
```zig
// src/observability/mod.zig
const build_options = @import("build_options");

// Tracing infrastructure (always available)
pub const tracing = @import("../shared/observability/tracing.zig");

// Monitoring features (feature-gated)
pub const metrics = if (build_options.enable_profiling)
    @import("../features/monitoring/mod.zig")
else
    @import("stub.zig");
```

#### 1.3 Update imports in dependent files
Replace `@import("../features/monitoring/")` with `@import("observability").metrics`.

#### Verification
```bash
zig build test --summary all
```

---

## Task 2: Evaluate Core Module

### Goal
Determine if `src/core/` should be deprecated or kept minimal.

### Current State
- `src/core/mod.zig` - Re-exports platform + version
- `src/core/profile.zig` - Profiling utilities
- Only 3 files total

### Steps

#### 2.1 Check core module usage
```bash
grep -r "core/" src/ --include="*.zig" | grep -v "src/core/" | head -20
```

#### 2.2 Decision Point
**If heavily used:** Keep as hardware/version introspection utility  
**If minimal usage:** Merge into `src/internal/` and update imports

#### 2.3 If merging
Move `src/core/profile.zig` → `src/internal/profile.zig`
Update `src/internal/mod.zig` to export profile utilities.
Update all imports.
Delete `src/core/`.

#### Verification
```bash
zig build test --summary all
```

---

## Task 3: Clean Up Features Module

### Goal
Clarify migration status and document what remains in features/.

### Current State
- `src/features/ai/` - Full implementation (not migrated)
- `src/features/connectors/` - API connectors (keep)
- `src/features/monitoring/` - Observability (consolidate to observability/)
- `src/features/ha/` - High availability (keep)

### Steps

#### 3.1 Update features/README.md
Document:
- Which features remain here vs migrated
- Connectors and HA are intentionally here
- AI migration is planned for Phase 3

#### 3.2 Add deprecation notice to monitoring
```zig
// src/features/monitoring/mod.zig
//! @deprecated Use src/observability/ instead.
//! This module will be removed in a future version.
```

#### 3.3 Update features/mod.zig
Remove references to migrated modules, keep connectors/ha.

#### Verification
```bash
zig build test --summary all
```

---

## Task 4: Document Legacy Patterns

### Goal
Make intentional backward-compatibility patterns explicit.

### Steps

#### 4.1 Add comments to compute/mod.zig
```zig
//! Backward Compatibility Layer
//!
//! This module re-exports from src/runtime/ for API stability.
//! New code should use `@import("runtime")` directly.
//!
//! @deprecated Prefer src/runtime/mod.zig for new code.
```

#### 4.2 Update CLAUDE.md Architecture section
Add note about legacy patterns and migration status.

#### 4.3 Update src/README.md
Document the current module organization and migration status.

#### Verification
Review documentation for accuracy.

---

## Task 5: Run Full Verification

### Steps

#### 5.1 Run all tests
```bash
zig build test --summary all
```

#### 5.2 Test feature-disabled builds
```bash
zig build -Denable-profiling=false
```

#### 5.3 Verify CLI commands
```bash
zig build run -- --list-features
zig build run -- help
```

---

## Verification Checklist

- [ ] `zig build` succeeds
- [ ] `zig build test --summary all` passes (51+ tests)
- [ ] `zig build -Denable-profiling=false` succeeds
- [ ] `zig build run -- --list-features` works
- [ ] No broken imports
- [ ] Documentation updated

---

## Risk Assessment

| Task | Risk | Mitigation |
|------|------|------------|
| Observability consolidation | Low | Feature-gated, can keep both during transition |
| Core module evaluation | Low | Minimal changes if keeping |
| Features cleanup | Low | Documentation only, no code changes |
| Legacy documentation | None | Documentation only |

---

## Estimated Effort

| Task | Complexity | Files Changed |
|------|------------|---------------|
| Task 1: Observability | Medium | 5-10 |
| Task 2: Core evaluation | Low | 3-5 |
| Task 3: Features cleanup | Low | 2-3 |
| Task 4: Documentation | Low | 3-4 |
| Task 5: Verification | Low | 0 |

**Total:** ~15-25 files, low-medium complexity

---

## Future Phase 3: AI Module Migration

After Phase 2 completion, Phase 3 will tackle the full AI module migration:
- Move `src/features/ai/llm/` → `src/ai/llm/`
- Move `src/features/ai/embeddings/` → `src/ai/embeddings/`
- Move `src/features/ai/training/` → `src/ai/training/`
- Move `src/features/ai/abbey/` → `src/ai/abbey/`

This is a larger effort (~120 files) and will be planned separately.

