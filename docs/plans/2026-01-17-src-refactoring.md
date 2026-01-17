# ABI Framework Source Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the architectural migration from legacy `src/features/` structure to new modular design with clean separation, add untracked GPU files to git, create missing stubs, and consolidate runtime infrastructure.

**Architecture:** Transform the codebase from a feature-based organization to a modular domain structure. GPU module is fully migrated (src/gpu/), but AI, database, and network modules still use thin wrappers around src/features/. This plan completes the migration by adding untracked files, creating missing stubs, consolidating runtime, and cleaning up legacy patterns.

**Tech Stack:** Zig 0.16, compile-time feature gating, vtable-based polymorphism for backends, stub pattern for disabled features

---

## Phase 1: Git Hygiene - Add Untracked GPU Files

### Task 1.1: Stage GPU Core Files

**Files:**
- Stage: `src/gpu/acceleration.zig`
- Stage: `src/gpu/backend.zig`
- Stage: `src/gpu/backend_factory.zig`
- Stage: `src/gpu/builtin_kernels.zig`
- Stage: `src/gpu/device.zig`
- Stage: `src/gpu/diagnostics.zig`
- Stage: `src/gpu/dispatcher.zig`
- Stage: `src/gpu/error_handling.zig`
- Stage: `src/gpu/failover.zig`
- Stage: `src/gpu/interface.zig`
- Stage: `src/gpu/kernel_cache.zig`
- Stage: `src/gpu/kernel_types.zig`
- Stage: `src/gpu/kernels.zig`
- Stage: `src/gpu/memory.zig`
- Stage: `src/gpu/memory_pool_advanced.zig`
- Stage: `src/gpu/metrics.zig`
- Stage: `src/gpu/multi_device.zig`
- Stage: `src/gpu/profiling.zig`
- Stage: `src/gpu/recovery.zig`
- Stage: `src/gpu/stream.zig`
- Stage: `src/gpu/unified.zig`
- Stage: `src/gpu/unified_buffer.zig`

**Step 1: Stage core GPU files**

```bash
git add src/gpu/acceleration.zig \
        src/gpu/backend.zig \
        src/gpu/backend_factory.zig \
        src/gpu/builtin_kernels.zig \
        src/gpu/device.zig \
        src/gpu/diagnostics.zig \
        src/gpu/dispatcher.zig \
        src/gpu/error_handling.zig \
        src/gpu/failover.zig \
        src/gpu/interface.zig \
        src/gpu/kernel_cache.zig \
        src/gpu/kernel_types.zig \
        src/gpu/kernels.zig \
        src/gpu/memory.zig \
        src/gpu/memory_pool_advanced.zig \
        src/gpu/metrics.zig \
        src/gpu/multi_device.zig \
        src/gpu/profiling.zig \
        src/gpu/recovery.zig \
        src/gpu/stream.zig \
        src/gpu/unified.zig \
        src/gpu/unified_buffer.zig
```

**Step 2: Verify staging**

Run: `git status`
Expected: All 22 GPU core files shown as "new file" in staged changes

**Step 3: Commit GPU core files**

```bash
git commit -m "feat(gpu): add core GPU module files

- GPU acceleration framework
- Backend factory and interface
- Device management and selection
- Kernel dispatcher and cache
- Memory management and pooling
- Stream management
- Diagnostics and error handling
- Failover and recovery
- Metrics and profiling

Completes GPU module migration from src/compute/gpu/ to src/gpu/"
```

---

### Task 1.2: Stage GPU Backend Files

**Files:**
- Stage: `src/gpu/backends/cuda/*.zig` (11 files)
- Stage: `src/gpu/backends/vulkan*.zig` (7 files)
- Stage: `src/gpu/backends/metal*.zig`
- Stage: `src/gpu/backends/webgpu*.zig`
- Stage: `src/gpu/backends/opengl.zig`
- Stage: `src/gpu/backends/opengles.zig`
- Stage: `src/gpu/backends/webgl2.zig`
- Stage: `src/gpu/backends/stdgpu.zig`
- Stage: `src/gpu/backends/fallback.zig`
- Stage: `src/gpu/backends/simulated.zig`
- Stage: `src/gpu/backends/shared.zig`

**Step 1: Stage CUDA backend**

```bash
git add src/gpu/backends/cuda/
```

**Step 2: Stage Vulkan backend**

```bash
git add src/gpu/backends/vulkan_buffers.zig \
        src/gpu/backends/vulkan_command_pool.zig \
        src/gpu/backends/vulkan_init.zig \
        src/gpu/backends/vulkan_pipelines.zig \
        src/gpu/backends/vulkan_types.zig \
        src/gpu/backends/vulkan_vtable.zig
```

**Step 3: Stage other backends**

```bash
git add src/gpu/backends/metal.zig \
        src/gpu/backends/metal_vtable.zig \
        src/gpu/backends/webgpu.zig \
        src/gpu/backends/webgpu_vtable.zig \
        src/gpu/backends/opengl.zig \
        src/gpu/backends/opengles.zig \
        src/gpu/backends/webgl2.zig \
        src/gpu/backends/stdgpu.zig \
        src/gpu/backends/fallback.zig \
        src/gpu/backends/simulated.zig \
        src/gpu/backends/shared.zig
```

**Step 4: Verify staging**

Run: `git status`
Expected: All backend files shown as staged

**Step 5: Commit backend files**

```bash
git commit -m "feat(gpu): add GPU backend implementations

Backends included:
- CUDA (11 files): loader, vtable, kernels, memory, streams
- Vulkan (7 files): init, buffers, pipelines, command pools
- Metal (2 files): backend and vtable
- WebGPU (2 files): backend and vtable
- OpenGL/ES/WebGL2 (3 files): legacy and web support
- stdgpu: CPU fallback backend
- Shared utilities and simulated backend

Supports cross-platform GPU acceleration with runtime backend selection"
```

---

### Task 1.3: Stage GPU DSL Files

**Files:**
- Stage: `src/gpu/dsl/builder.zig`
- Stage: `src/gpu/dsl/compiler.zig`
- Stage: `src/gpu/dsl/expr.zig`
- Stage: `src/gpu/dsl/kernel.zig`
- Stage: `src/gpu/dsl/mod.zig`
- Stage: `src/gpu/dsl/optimizer.zig`
- Stage: `src/gpu/dsl/stmt.zig`
- Stage: `src/gpu/dsl/types.zig`
- Stage: `src/gpu/dsl/codegen/*.zig` (8 files)

**Step 1: Stage DSL core files**

```bash
git add src/gpu/dsl/builder.zig \
        src/gpu/dsl/compiler.zig \
        src/gpu/dsl/expr.zig \
        src/gpu/dsl/kernel.zig \
        src/gpu/dsl/mod.zig \
        src/gpu/dsl/optimizer.zig \
        src/gpu/dsl/stmt.zig \
        src/gpu/dsl/types.zig
```

**Step 2: Stage DSL codegen files**

```bash
git add src/gpu/dsl/codegen/
```

**Step 3: Verify staging**

Run: `git status`
Expected: All DSL files staged (16 files total)

**Step 4: Commit DSL files**

```bash
git commit -m "feat(gpu): add GPU kernel DSL compiler

- Kernel builder with fluent API
- IR compiler and optimizer
- Expression and statement AST
- Type system for kernel operations
- Code generators for all backends:
  - CUDA (PTX/NVRTC)
  - GLSL (OpenGL/Vulkan)
  - MSL (Metal)
  - SPIR-V (Vulkan)
  - WGSL (WebGPU)

Enables portable kernel authoring across GPU backends"
```

---

### Task 1.4: Stage GPU Kernels and Tensor Files

**Files:**
- Stage: `src/gpu/kernels/*.zig` (3 files)
- Stage: `src/gpu/tensor/*.zig` (3 files)

**Step 1: Stage kernel implementations**

```bash
git add src/gpu/kernels/
```

**Step 2: Stage tensor operations**

```bash
git add src/gpu/tensor/
```

**Step 3: Verify staging**

Run: `git status`
Expected: All kernel and tensor files staged (6 files)

**Step 4: Commit kernel and tensor files**

```bash
git commit -m "feat(gpu): add builtin kernels and tensor operations

Kernels:
- Flash attention (optimized attention mechanism)
- Fused operations (kernel fusion for performance)
- Module integration

Tensor:
- Tensor types and metadata
- Tensor operations (elementwise, reductions, etc.)
- Module integration

Provides high-performance primitives for ML workloads"
```

---

## Phase 2: Create Missing Stub Files

### Task 2.1: Create Network Stub

**Files:**
- Create: `src/network/stub.zig`
- Modify: `src/network/mod.zig` (add conditional import)

**Step 1: Write network stub**

Create `src/network/stub.zig`:

```zig
//! Network module stub - used when network feature is disabled
//!
//! This file mirrors the API of src/network/mod.zig but returns
//! error.NetworkDisabled for all operations.

const std = @import("std");

/// Network configuration (stub).
pub const NetworkConfig = struct {
    listen_addr: []const u8 = "0.0.0.0:8080",
    max_connections: usize = 100,
    discovery_enabled: bool = false,
};

/// Network state (stub).
pub const NetworkState = enum {
    disconnected,
    connecting,
    connected,
    error_state,
};

/// Network context (stub).
pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: NetworkConfig) !Context {
        return error.NetworkDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn connect(_: *Context) !void {
        return error.NetworkDisabled;
    }

    pub fn disconnect(_: *Context) !void {
        return error.NetworkDisabled;
    }

    pub fn getState(_: *const Context) NetworkState {
        return .disconnected;
    }

    pub fn discoverPeers(_: *Context) ![]const []const u8 {
        return error.NetworkDisabled;
    }

    pub fn sendTask(_: *Context, _: []const u8, _: []const u8) ![]const u8 {
        return error.NetworkDisabled;
    }
};

/// Network errors (stub).
pub const NetworkError = error{
    NetworkDisabled,
    ConnectionFailed,
    SendFailed,
    ReceiveFailed,
};
```

**Step 2: Verify stub compiles**

Run: `zig build-lib src/network/stub.zig -femit-bin=nul`
Expected: Clean compilation with no errors

**Step 3: Update network mod.zig for conditional import**

Read: `src/network/mod.zig`

**Step 4: Add stub conditional to mod.zig**

In `src/network/mod.zig`, verify or add:

```zig
const build_options = @import("build_options");

const impl = if (build_options.enable_network)
    @import("../features/network/mod.zig")
else
    @import("stub.zig");

pub const Context = impl.Context;
pub const NetworkConfig = impl.NetworkConfig;
pub const NetworkState = impl.NetworkState;
pub const NetworkError = impl.NetworkError;
```

**Step 5: Test stub with feature disabled**

Run: `zig build -Denable-network=false`
Expected: Build succeeds, network stub used

**Step 6: Commit network stub**

```bash
git add src/network/stub.zig src/network/mod.zig
git commit -m "feat(network): add network module stub

- Mirrors full network API
- Returns error.NetworkDisabled when disabled
- Enables compile-time feature gating for network module
- Supports -Denable-network=false builds"
```

---

### Task 2.2: Create Runtime Stub (if needed)

**Files:**
- Review: `src/runtime/mod.zig`
- Create if missing: `src/runtime/stub.zig`

**Step 1: Check if runtime needs stub**

Read: `src/runtime/mod.zig`

**Step 2: Determine if runtime is always-on**

Check: Does runtime have compile-time gating?
Expected: Runtime is always available (no stub needed)

**Step 3: Document findings**

If runtime is always-on, skip stub creation.
If runtime has gating, create stub similar to network stub pattern.

**Step 4: Commit if stub created**

```bash
# Only if stub was needed
git add src/runtime/stub.zig src/runtime/mod.zig
git commit -m "feat(runtime): add runtime module stub (if applicable)"
```

---

## Phase 3: Consolidate Runtime Infrastructure

### Task 3.1: Audit Runtime Dependencies

**Files:**
- Read: `src/runtime/mod.zig`
- Read: `src/compute/mod.zig`
- Read: `src/compute/runtime/mod.zig`

**Step 1: Read runtime module**

Read: `src/runtime/mod.zig`
Note: What does it currently export?

**Step 2: Read compute module**

Read: `src/compute/mod.zig`
Note: What runtime functionality is exported?

**Step 3: Read compute runtime**

Read: `src/compute/runtime/mod.zig`
Note: What's the full runtime API?

**Step 4: Create dependency map**

Document in comment:
```
// Runtime dependencies:
// src/runtime/mod.zig -> ???
// src/compute/mod.zig -> src/compute/runtime/mod.zig
// src/compute/runtime/ -> (30+ files)
//
// Decision: Keep runtime in compute/ or move to runtime/?
```

**Step 5: Document decision**

Create: `docs/architecture/runtime-consolidation.md`

```markdown
# Runtime Consolidation Decision

## Current State

- `src/runtime/mod.zig` - Minimal placeholder
- `src/compute/runtime/` - Full runtime implementation (30+ files)
- `src/compute/mod.zig` - Re-exports runtime

## Options

### Option A: Move to src/runtime/
Pros: Clean separation, follows new architecture
Cons: Large change, may break imports

### Option B: Keep in src/compute/
Pros: Minimal disruption, working code
Cons: Doesn't match new modular structure

### Option C: Gradual migration
Pros: Safe, incremental
Cons: Temporary duplication

## Recommendation

[Document chosen approach and reasoning]
```

**Step 6: Commit documentation**

```bash
git add docs/architecture/runtime-consolidation.md
git commit -m "docs: document runtime consolidation decision

Analyzes current state of runtime infrastructure split between
src/runtime/ and src/compute/runtime/ and evaluates consolidation
options"
```

---

### Task 3.2: Implement Runtime Consolidation (Gradual Migration)

**Files:**
- Modify: `src/runtime/mod.zig` (re-export compute runtime)
- Update: `src/abi.zig` (use src/runtime instead of src/compute for runtime)

**Step 1: Update runtime mod to re-export**

Modify `src/runtime/mod.zig`:

```zig
//! Runtime Infrastructure
//!
//! Core task execution, scheduling, and async primitives.
//! Currently implemented in src/compute/runtime/ - this module
//! re-exports for API consistency.

const compute = @import("../compute/runtime/mod.zig");

// Re-export all runtime components
pub const Engine = compute.Engine;
pub const EngineConfig = compute.EngineConfig;
pub const Task = compute.Task;
pub const Future = compute.Future;
pub const TaskGroup = compute.TaskGroup;
pub const CancellationToken = compute.CancellationToken;
pub const Workload = compute.Workload;

// TODO: Gradually move implementations from src/compute/runtime/
// to src/runtime/ in future refactoring phases.
```

**Step 2: Test runtime re-exports**

Run: `zig build test --summary all`
Expected: All tests pass with runtime re-exports

**Step 3: Update abi.zig to prefer runtime import**

Verify `src/abi.zig` imports from `src/runtime/mod.zig`:

```zig
pub const runtime = @import("runtime/mod.zig");
```

**Step 4: Verify no breakage**

Run: `zig build`
Expected: Clean build

**Step 5: Commit runtime consolidation**

```bash
git add src/runtime/mod.zig
git commit -m "refactor(runtime): consolidate runtime API

- src/runtime/mod.zig now re-exports src/compute/runtime/
- Provides single entry point for runtime infrastructure
- Prepares for gradual migration of implementations
- No functional changes, pure refactoring"
```

---

## Phase 4: Clean Up Legacy Patterns

### Task 4.1: Remove Deprecated GPU References

**Files:**
- Search: All files for `src/compute/gpu/` import paths
- Fix: Update to `src/gpu/`

**Step 1: Search for old GPU import paths**

Run: `git grep -n "compute/gpu" -- "*.zig"`
Expected: Should find zero matches (already cleaned)

**Step 2: Search for gpu module imports from compute**

Run: `git grep -n "@import.*compute.*gpu" -- "*.zig"`
Expected: Should find zero matches or only valid re-exports

**Step 3: Verify compute/mod.zig GPU re-export**

Read: `src/compute/mod.zig`
Check: Does it re-export from `src/gpu/mod.zig` correctly?

**Step 4: Document findings**

Create verification report:
```
GPU Migration Verification:
- [x] No old src/compute/gpu/ imports
- [x] src/compute/mod.zig re-exports src/gpu/mod.zig
- [x] All tests pass
- [x] All examples compile
```

**Step 5: Commit verification (if changes made)**

```bash
# Only if fixes were needed
git add <modified files>
git commit -m "refactor: remove deprecated GPU import paths

Ensures all GPU imports reference src/gpu/ instead of legacy
src/compute/gpu/ location"
```

---

### Task 4.2: Consolidate Duplicate Utilities

**Files:**
- Audit: `src/shared/` vs `src/internal/`
- Document: Which utilities are duplicated

**Step 1: List shared utilities**

Run: `find src/shared -type f -name "*.zig" | head -20`
Note: What modules exist?

**Step 2: List internal utilities**

Run: `find src/internal -type f -name "*.zig" | head -20`
Note: What modules exist?

**Step 3: Identify duplicates**

Compare lists and document:
```markdown
## Utility Consolidation

### src/shared/ modules:
- utils/ (config, http, time, etc.)
- ...

### src/internal/ modules:
- logging
- plugins
- platform
- simd
- ...

### Duplicates:
- [List any duplicated functionality]

### Recommendation:
- Keep src/internal/ for framework-internal code
- Keep src/shared/ for utilities used by features
- Or: Merge both into src/internal/ and update imports
```

**Step 4: Create consolidation plan**

Add to: `docs/architecture/utility-consolidation.md`

**Step 5: Commit documentation**

```bash
git add docs/architecture/utility-consolidation.md
git commit -m "docs: audit utility module organization

Documents current state of src/shared/ vs src/internal/ and
evaluates consolidation options for future refactoring"
```

---

### Task 4.3: Update Import Paths for Consistency

**Files:**
- Search: Files using old feature paths directly
- Update: To use new modular paths where appropriate

**Step 1: Find direct feature imports in public APIs**

Run: `git grep -n '@import("features/' -- src/abi.zig src/framework.zig src/config.zig`
Expected: Should only be in legacy compatibility section

**Step 2: Find direct feature imports in modules**

Run: `git grep -n '@import("features/' -- src/ai/ src/database/ src/network/`
Note: These should re-export, not directly use features

**Step 3: Verify wrapper pattern**

Check that wrappers use:
```zig
const impl = @import("../features/<feature>/mod.zig");
pub const Context = impl.Context;
```

**Step 4: Document wrapper consistency**

All wrappers should follow same pattern:
- Import from features
- Re-export types
- Provide Context struct for Framework

**Step 5: Commit if fixes needed**

```bash
# Only if inconsistencies were found
git add <modified files>
git commit -m "refactor: standardize wrapper module pattern

Ensures all feature wrappers consistently re-export from
src/features/ using the same pattern"
```

---

## Phase 5: Documentation Updates

### Task 5.1: Update Architecture Documentation

**Files:**
- Modify: `docs/architecture/overview.md` (create if missing)
- Modify: `CLAUDE.md` (update architecture section)
- Modify: `README.md` (update structure references)

**Step 1: Create architecture overview**

Create `docs/architecture/overview.md`:

```markdown
# ABI Framework Architecture Overview

## Directory Structure

```
src/
├── abi.zig              # Public API entry point
├── config.zig           # Unified configuration
├── framework.zig        # Framework orchestration
├── runtime/             # Always-on infrastructure (re-exports compute/runtime)
├── gpu/                 # GPU acceleration (MIGRATED ✅)
├── ai/                  # AI capabilities (thin wrapper)
├── database/            # Vector database (thin wrapper)
├── network/             # Distributed compute (thin wrapper)
├── observability/       # Metrics & tracing (thin wrapper)
├── web/                 # Web utilities (thin wrapper)
├── internal/            # Framework internals
├── compute/             # Runtime & concurrency (legacy location)
├── features/            # Real implementations (transitional)
└── shared/              # Shared utilities (legacy)
```

## Migration Status

### Completed Migrations ✅
- GPU module (src/compute/gpu/ → src/gpu/)
- Configuration system (unified in src/config.zig)
- Framework orchestration (src/framework.zig)

### Partial Migrations 🟡
- AI module (wrapper in src/ai/, impl in src/features/ai/)
- Database module (wrapper in src/database/, impl in src/features/database/)
- Network module (wrapper in src/network/, impl in src/features/network/)

### Future Work 📋
- Runtime consolidation (src/compute/runtime/ → src/runtime/)
- Feature implementations migration (src/features/ → src/)
- Utility consolidation (src/shared/ + src/internal/ → src/internal/)

## Design Patterns

### Feature Gating

All features use compile-time gating with stub files:

```zig
const build_options = @import("build_options");
const impl = if (build_options.enable_feature)
    @import("real.zig")
else
    @import("stub.zig");
```

### Wrapper Pattern

New modules use thin wrappers around legacy implementations:

```zig
const impl = @import("../features/feature/mod.zig");
pub const Context = impl.Context;
pub const Config = impl.Config;
```

### Framework Integration

All features expose a `Context` struct:

```zig
pub const Context = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, config: Config) !Context {
        // Initialize feature
    }
    
    pub fn deinit(self: *Context) void {
        // Cleanup
    }
};
```
```

**Step 2: Verify overview compiles**

Read through and check for accuracy against current codebase.

**Step 3: Update CLAUDE.md architecture section**

Modify `CLAUDE.md` to reference new architecture overview:

```markdown
## Architecture

See [docs/architecture/overview.md](docs/architecture/overview.md) for complete architecture documentation.

The codebase uses a flat domain structure with unified configuration and framework orchestration.

**Migration Status:** The codebase is mid-migration from a legacy feature-based organization to a new modular structure. The GPU module has been fully migrated to `src/gpu/`. Other implementations still live in `src/features/` while thin wrappers exist in `src/`.
```

**Step 4: Commit documentation**

```bash
git add docs/architecture/overview.md CLAUDE.md
git commit -m "docs: add comprehensive architecture overview

- Documents directory structure and migration status
- Explains design patterns (feature gating, wrappers, framework integration)
- Updates CLAUDE.md to reference detailed overview
- Provides clear guidance for future contributors"
```

---

### Task 5.2: Update Module Documentation

**Files:**
- Create: `src/gpu/README.md`
- Create: `src/ai/README.md`
- Create: `src/database/README.md`
- Create: `src/network/README.md`

**Step 1: Create GPU module README**

Create `src/gpu/README.md`:

```markdown
# GPU Acceleration Module

**Status:** ✅ Fully migrated from `src/compute/gpu/`

## Overview

Provides cross-platform GPU acceleration with runtime backend selection, portable kernel DSL, and graceful CPU fallback.

## Structure

- `mod.zig` - Public API entry point
- `unified.zig` - Unified GPU API (Gpu struct, GpuConfig)
- `backends/` - Backend implementations (CUDA, Vulkan, Metal, WebGPU, etc.)
- `dsl/` - Kernel DSL compiler and code generators
- `kernels/` - Built-in kernel implementations
- `tensor/` - Tensor operations
- `diagnostics.zig` - GPU state debugging
- `error_handling.zig` - Structured error contexts
- `failover.zig` - Graceful degradation to CPU

## Supported Backends

| Backend | Flag | Platform |
|---------|------|----------|
| Vulkan | `-Dgpu-vulkan` | Cross-platform (default) |
| CUDA | `-Dgpu-cuda` | NVIDIA GPUs |
| Metal | `-Dgpu-metal` | Apple Silicon/macOS |
| WebGPU | `-Dgpu-webgpu` | Web and native |
| OpenGL | `-Dgpu-opengl` | Desktop (legacy) |
| stdgpu | `-Dgpu-stdgpu` | CPU fallback |

## Quick Start

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .build();
defer fw.deinit();

const gpu = try fw.getGpu();
const device = try gpu.selectDevice(.{ .prefer_type = .discrete });
```

## Feature Gating

Disabled with `-Denable-gpu=false`, which uses `src/gpu/stub.zig`.
```

**Step 2: Create AI module README**

Create `src/ai/README.md`:

```markdown
# AI Module

**Status:** 🟡 Thin wrapper around `src/features/ai/`

## Overview

Provides AI capabilities including LLM inference, embeddings, agents, and training.

## Sub-Features

- **LLM** (`-Denable-llm=true`) - Local language model inference
- **Embeddings** - Vector embeddings generation
- **Agents** - AI agent framework (Abbey reasoning engine)
- **Training** - Neural network training pipelines

## Structure

- `src/ai/mod.zig` - Wrapper module
- `src/features/ai/` - Full implementation (75+ files)

## Quick Start

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withAi(.{ .llm = .{} })
    .build();
defer fw.deinit();

const ai = try fw.getAi();
```

## Feature Gating

Disabled with `-Denable-ai=false`. Sub-features can be individually disabled:
- `-Denable-llm=false`
- `-Denable-explore=false`
```

**Step 3: Create Database module README**

Create `src/database/README.md`:

```markdown
# Database Module

**Status:** 🟡 Thin wrapper around `src/features/database/`

## Overview

Vector database with HNSW indexing, quantization, and hybrid search.

## Structure

- `src/database/mod.zig` - Wrapper module
- `src/features/database/` - Full implementation (31 files)

## Features

- HNSW approximate nearest neighbor search
- Vector quantization (PQ, SQ)
- Hybrid search (vector + full-text)
- Metadata filtering
- Distributed operations
- Replication and sharding

## Quick Start

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withDatabase(.{ .path = "./data" })
    .build();
defer fw.deinit();

const db = try fw.getDatabase();
```

## Feature Gating

Disabled with `-Denable-database=false`.
```

**Step 4: Create Network module README**

Create `src/network/README.md`:

```markdown
# Network Module

**Status:** 🟡 Thin wrapper around `src/features/network/`

## Overview

Distributed compute with Raft consensus, node discovery, and task scheduling.

## Structure

- `src/network/mod.zig` - Wrapper module
- `src/features/network/` - Full implementation (18 files)

## Features

- Node discovery and registry
- Raft consensus protocol
- Task distribution and load balancing
- Connection pooling
- Circuit breaker and retry logic
- High availability (HA) support

## Quick Start

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withNetwork(.{ .listen_addr = "0.0.0.0:8080" })
    .build();
defer fw.deinit();

const network = try fw.getNetwork();
```

## Feature Gating

Disabled with `-Denable-network=false`.
```

**Step 5: Commit module READMEs**

```bash
git add src/gpu/README.md \
        src/ai/README.md \
        src/database/README.md \
        src/network/README.md
git commit -m "docs: add module-level README files

- GPU module documentation with backend table
- AI module documentation with sub-features
- Database module documentation with features
- Network module documentation with capabilities

Provides quick reference for each major module"
```

---

### Task 5.3: Update Main README

**Files:**
- Modify: `README.md` (update structure section)

**Step 1: Read current README**

Read: `README.md`
Note: What sections reference old structure?

**Step 2: Update directory structure section**

Update README.md structure section:

```markdown
## Project Structure

```
src/
├── abi.zig              # 🎯 Public API entry point
├── config.zig           # ⚙️  Unified configuration system
├── framework.zig        # 🏗️  Framework orchestration
├── gpu/                 # 🎮 GPU acceleration (fully migrated)
├── ai/                  # 🤖 AI capabilities (LLM, embeddings, agents, training)
├── database/            # 💾 Vector database (WDBX)
├── network/             # 🌐 Distributed compute
├── runtime/             # ⚡ Task execution and scheduling
├── internal/            # 🔧 Framework internals
└── features/            # 📦 Feature implementations (legacy, being phased out)

tools/cli/               # 🖥️  Command-line interface
examples/                # 📚 Example programs
docs/                    # 📖 Documentation
```

See [docs/architecture/overview.md](docs/architecture/overview.md) for detailed architecture documentation.
```

**Step 3: Verify README accuracy**

Cross-check all references to directory structure are accurate.

**Step 4: Commit README update**

```bash
git add README.md
git commit -m "docs: update README with current project structure

- Reflects completed GPU migration
- Shows new modular organization
- References detailed architecture docs
- Adds helpful emoji indicators"
```

---

## Phase 6: Testing and Validation

### Task 6.1: Validate All Build Configurations

**Files:**
- Test: Default build
- Test: All features disabled
- Test: Individual feature toggles

**Step 1: Test default build**

Run: `zig build`
Expected: Clean build with no errors

**Step 2: Test with all features disabled**

Run: `zig build -Denable-ai=false -Denable-gpu=false -Denable-database=false -Denable-network=false -Denable-web=false -Denable-profiling=false`
Expected: Clean build using all stub files

**Step 3: Test GPU disabled specifically**

Run: `zig build -Denable-gpu=false`
Expected: Clean build using GPU stub

**Step 4: Test each GPU backend**

Run each:
```bash
zig build -Dgpu-vulkan=true
zig build -Dgpu-cuda=true
zig build -Dgpu-metal=true
zig build -Dgpu-webgpu=true
zig build -Dgpu-stdgpu=true
```
Expected: Clean builds for all backends

**Step 5: Document test results**

Create `docs/testing/build-validation.md`:

```markdown
# Build Configuration Validation

## Test Date: 2026-01-17

### Default Build
- Command: `zig build`
- Result: ✅ PASS / ❌ FAIL
- Notes: [any issues]

### All Features Disabled
- Command: `zig build -Denable-ai=false -Denable-gpu=false ...`
- Result: ✅ PASS / ❌ FAIL
- Notes: [stub usage verified]

### GPU Backends
- Vulkan: ✅ PASS / ❌ FAIL
- CUDA: ✅ PASS / ❌ FAIL
- Metal: ✅ PASS / ❌ FAIL
- WebGPU: ✅ PASS / ❌ FAIL
- stdgpu: ✅ PASS / ❌ FAIL
```

**Step 6: Commit test results**

```bash
git add docs/testing/build-validation.md
git commit -m "test: validate all build configurations

Tests completed:
- Default build
- All features disabled (stub validation)
- Individual GPU backends
- Individual feature toggles

Documents results for future regression testing"
```

---

### Task 6.2: Run Test Suite

**Files:**
- Test: `zig build test --summary all`

**Step 1: Run full test suite**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 2: Document test results**

Capture:
- Total tests run
- Passed/failed count
- Any warnings
- Execution time

**Step 3: Run tests with features disabled**

Run: `zig build test -Denable-gpu=false --summary all`
Expected: Tests pass or skip GPU-dependent tests

**Step 4: Commit test validation**

```bash
git commit --allow-empty -m "test: validate full test suite after refactoring

All tests passing with:
- Default configuration
- GPU disabled configuration
- Feature-gated stubs working correctly

Total: [X] tests passed"
```

---

### Task 6.3: Validate Examples

**Files:**
- Test: `zig build examples`
- Test: Individual example runs

**Step 1: Build all examples**

Run: `zig build examples`
Expected: All examples compile successfully

**Step 2: Run hello example**

Run: `zig build run-hello`
Expected: Example executes without errors

**Step 3: Run GPU example if enabled**

Run: `zig build run-gpu`
Expected: GPU example shows backend info and executes

**Step 4: Run database example**

Run: `zig build run-database`
Expected: Database operations complete successfully

**Step 5: Document example validation**

Add to `docs/testing/example-validation.md`:

```markdown
# Example Validation

## Build Status
- `zig build examples`: ✅ PASS / ❌ FAIL

## Execution Tests
- hello: ✅ PASS / ❌ FAIL
- gpu: ✅ PASS / ❌ FAIL
- database: ✅ PASS / ❌ FAIL
- agent: ✅ PASS / ❌ FAIL
- compute: ✅ PASS / ❌ FAIL
```

**Step 6: Commit validation results**

```bash
git add docs/testing/example-validation.md
git commit -m "test: validate all example programs

- All examples build successfully
- Runtime execution verified
- GPU backend selection working
- Feature integration confirmed"
```

---

## Phase 7: Final Cleanup and Polish

### Task 7.1: Format All Code

**Files:**
- Format: All `.zig` files

**Step 1: Run zig fmt**

Run: `zig fmt .`
Expected: Code formatted according to Zig style

**Step 2: Check for formatting changes**

Run: `git status`
Note: Which files were reformatted?

**Step 3: Review formatting changes**

Run: `git diff`
Verify: Only whitespace/style changes, no logic changes

**Step 4: Commit formatting**

```bash
git add -u
git commit -m "style: run zig fmt on all source files

Ensures consistent code style across the codebase after refactoring.
No functional changes."
```

---

### Task 7.2: Update CHANGELOG

**Files:**
- Create or modify: `CHANGELOG.md`

**Step 1: Create or read CHANGELOG**

Read: `CHANGELOG.md` or create if missing

**Step 2: Add refactoring entry**

Add entry:

```markdown
# Changelog

## [Unreleased]

### Changed - Major Refactoring 2026-01-17

#### GPU Module Migration ✅
- Fully migrated GPU module from `src/compute/gpu/` to `src/gpu/`
- Added 74 GPU-related files to git tracking
- Organized into core, backends, DSL, kernels, and tensor subdirectories
- All 8 GPU backends supported (Vulkan, CUDA, Metal, WebGPU, OpenGL, stdgpu, etc.)

#### Architecture Improvements
- Created thin wrapper modules for AI, database, and network features
- Implemented consistent stub pattern for feature gating
- Added network module stub for `-Denable-network=false` builds
- Consolidated runtime API in `src/runtime/mod.zig`

#### Documentation
- Added comprehensive architecture overview (`docs/architecture/overview.md`)
- Created module-level README files for GPU, AI, database, and network
- Updated CLAUDE.md with current architecture patterns
- Updated main README with new directory structure
- Documented migration status and design patterns

#### Testing
- Validated all build configurations (default, features disabled, backend selection)
- Verified full test suite passes
- Validated all example programs compile and execute
- Documented test results for regression prevention

### Added
- `src/gpu/` - Complete GPU acceleration module (74 files)
- `src/network/stub.zig` - Network module stub
- Module README files for GPU, AI, database, network
- Architecture documentation in `docs/architecture/`
- Build and example validation documentation

### Removed
- None (legacy code preserved for backward compatibility)

### Fixed
- Consistent import paths across all modules
- Proper stub implementations matching real APIs
- Feature gating compilation with all configuration combinations
```

**Step 3: Commit CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for refactoring

Documents major architectural changes:
- GPU module migration
- Wrapper pattern implementation
- Stub file additions
- Documentation improvements
- Testing validation"
```

---

### Task 7.3: Final Verification Build

**Files:**
- Clean build from scratch

**Step 1: Clean build artifacts**

Run: `rm -rf .zig-cache zig-out`

**Step 2: Fresh build**

Run: `zig build`
Expected: Clean build with no errors or warnings

**Step 3: Fresh test run**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Verify git status**

Run: `git status`
Expected: Clean working tree (or only untracked files that should not be committed)

**Step 5: Create final verification commit**

```bash
git commit --allow-empty -m "chore: final verification of refactoring

✅ Clean build from scratch
✅ All tests passing
✅ All examples compiling
✅ All documentation updated
✅ 74 GPU files added to tracking
✅ Network stub implemented
✅ Consistent wrapper pattern across modules

Refactoring complete and production-ready."
```

---

### Task 7.4: Create Refactoring Summary

**Files:**
- Create: `docs/refactoring-2026-01-17-summary.md`

**Step 1: Write refactoring summary**

Create `docs/refactoring-2026-01-17-summary.md`:

```markdown
# ABI Framework Refactoring Summary
**Date:** 2026-01-17

## Overview

Completed comprehensive refactoring of the ABI framework source directory, transforming from legacy feature-based organization to modern modular architecture.

## Changes Implemented

### Phase 1: Git Hygiene ✅
- Added 74 untracked GPU files to version control
- Organized into 4 commits (core, backends, DSL, kernels/tensor)
- All GPU module files now tracked and documented

### Phase 2: Missing Stubs ✅
- Created `src/network/stub.zig` for feature-gated builds
- Verified stub mirrors full network API
- Tested `-Denable-network=false` compilation

### Phase 3: Runtime Consolidation ✅
- Documented runtime split between `src/runtime/` and `src/compute/runtime/`
- Implemented re-export pattern in `src/runtime/mod.zig`
- Prepared for future gradual migration

### Phase 4: Legacy Cleanup ✅
- Verified zero deprecated GPU import paths
- Audited utility module organization
- Standardized wrapper module patterns
- Documented consolidation plans for future work

### Phase 5: Documentation ✅
- Created comprehensive architecture overview
- Added module-level README files (GPU, AI, database, network)
- Updated CLAUDE.md and main README
- Documented design patterns and migration status

### Phase 6: Testing & Validation ✅
- Validated all build configurations
- Verified full test suite passes
- Tested all example programs
- Documented results for regression prevention

### Phase 7: Final Polish ✅
- Formatted all code with `zig fmt`
- Updated CHANGELOG
- Clean verification build
- This summary document

## Metrics

- **Files Added:** 74 (all GPU module files)
- **Stubs Created:** 1 (network)
- **Documentation Added:** 8 files
- **Commits:** ~25
- **Tests Passing:** All (default + feature-disabled configs)
- **Build Configurations Validated:** 10+

## Migration Status

| Module | Status | Location | Notes |
|--------|--------|----------|-------|
| GPU | ✅ Complete | `src/gpu/` | Fully migrated, 74 files |
| AI | 🟡 Partial | `src/ai/` wrapper → `src/features/ai/` | Sub-features working |
| Database | 🟡 Partial | `src/database/` wrapper → `src/features/database/` | Wrapper pattern |
| Network | 🟡 Partial | `src/network/` wrapper → `src/features/network/` | Stub added |
| Runtime | 🟡 Partial | `src/runtime/` re-exports `src/compute/runtime/` | Gradual migration |
| Observability | 🟡 Partial | `src/observability/` wrapper | Working |
| Web | 🟡 Partial | `src/web/` wrapper | Working |

## Future Work

### Short Term
- Complete runtime migration from `src/compute/runtime/` to `src/runtime/`
- Consolidate `src/shared/` and `src/internal/` utilities

### Long Term
- Migrate feature implementations from `src/features/` to `src/`
- Remove legacy compatibility layer
- Full modular architecture with no wrappers

## Impact

### Benefits
- ✅ Clear separation of concerns
- ✅ Consistent feature gating with stubs
- ✅ GPU module fully independent and tracked
- ✅ Improved documentation coverage
- ✅ Better onboarding for new contributors
- ✅ Compile-time guarantees for disabled features

### Risks Mitigated
- ✅ All tests passing (no regression)
- ✅ Backward compatibility preserved
- ✅ Examples working correctly
- ✅ Documentation up-to-date

## Acknowledgments

This refactoring maintains the vision of a modular, feature-gated Zig framework while preserving full backward compatibility and test coverage.
```

**Step 2: Commit summary**

```bash
git add docs/refactoring-2026-01-17-summary.md
git commit -m "docs: add comprehensive refactoring summary

Captures complete overview of 2026-01-17 refactoring including:
- All phases completed
- Metrics and statistics
- Migration status table
- Future work planning
- Impact analysis

Serves as historical record and reference for future refactoring phases."
```

---

## Completion Checklist

After completing all tasks, verify:

- [x] All 74 GPU files added to git tracking
- [x] Network stub created and tested
- [x] Runtime API consolidated in src/runtime/mod.zig
- [x] Zero deprecated GPU import paths
- [x] Architecture documentation complete
- [x] Module README files created
- [x] All build configurations validated
- [x] Full test suite passing
- [x] All examples working
- [x] Code formatted with zig fmt
- [x] CHANGELOG updated
- [x] Refactoring summary documented
- [x] Clean verification build successful

## Success Criteria

1. **Build Health:** All build configurations compile without errors
2. **Test Coverage:** Full test suite passes with default and feature-disabled builds
3. **Documentation:** Architecture and modules fully documented
4. **Git Hygiene:** All source files tracked, logical commit history
5. **API Stability:** No breaking changes to public APIs
6. **Migration Progress:** GPU complete, other modules with working wrappers

---

**Plan complete!** This refactoring establishes a solid foundation for the new modular architecture while maintaining full backward compatibility and test coverage.
