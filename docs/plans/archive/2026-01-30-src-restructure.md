# ABI src/ Directory Restructure Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `src/` into a clean layered Zig 0.16 layout with consistent module boundaries and feature-gate stubs.

**Architecture:** Flat domain structure with `abi.zig` as sole public entry point, `framework/` for lifecycle orchestration, and domain modules (`ai/`, `gpu/`, `database/`, `network/`, `observability/`, `web/`, `cli/`, `integrations/`) each with `mod.zig` entry points and `stub.zig` for disabled builds.

**Tech Stack:** Zig 0.16, comptime feature detection, parent-export import patterns

---

## Current State Analysis

**Total Files:** 636+ Zig files across 17+ modules
**Well-Organized (No Change):** config/, runtime/, ai/, gpu/, database/, network/, observability/, shared/, connectors/, web/, cloud/, ha/, registry/, tasks/, tests/
**Needs Restructuring:**
- `src/main.zig` → `src/cli/main.zig`
- `tools/cli/` → `src/cli/`
- `src/cpu.zig` → `src/platform/cpu.zig`
- `src/io.zig` → `src/shared/io.zig`
- `src/connectors/` → `src/integrations/connectors/`
- `src/cloud/` → `src/integrations/cloud/`

---

## Target Structure

```
src/
├── abi.zig                    # Public API entry point (KEEP)
├── flags.zig                  # Feature flags (KEEP)
├── framework.zig → framework/ # Move to directory
│   ├── mod.zig               # Framework entry point
│   ├── lifecycle.zig         # Lifecycle management
│   └── orchestration.zig     # Feature orchestration
├── config/                    # Configuration (9 files - KEEP)
├── platform/                  # NEW - Platform abstractions
│   ├── mod.zig
│   ├── cpu.zig               # From src/cpu.zig
│   ├── detection.zig         # OS/arch detection
│   └── simd.zig              # SIMD capabilities
├── shared/                    # Shared utilities (48 files)
│   ├── mod.zig
│   ├── io.zig                # From src/io.zig
│   └── ... (existing)
├── runtime/                   # Runtime infrastructure (25 files - KEEP)
├── ai/                        # AI features (236 files - KEEP)
├── gpu/                       # GPU acceleration (145 files - KEEP)
├── database/                  # Vector database (41 files - KEEP)
├── network/                   # Distributed networking (29 files - KEEP)
├── observability/             # Metrics/tracing (11 files - KEEP)
├── web/                       # Web utilities (6 files - KEEP)
├── cli/                       # NEW - CLI commands
│   ├── mod.zig
│   ├── main.zig              # From src/main.zig
│   ├── commands/             # From tools/cli/commands/
│   └── tui/                  # From tools/cli/tui/
├── integrations/              # NEW - External integrations
│   ├── mod.zig
│   ├── connectors/           # From src/connectors/
│   └── cloud/                # From src/cloud/
├── ha/                        # High availability (4 files - KEEP)
├── registry/                  # Feature registry (4 files - KEEP)
├── tasks/                     # Task management (6 files - KEEP)
└── tests/                     # Test infrastructure (50+ files - KEEP)
```

---

## Task 1: Create platform/ Module

**Files:**
- Create: `src/platform/mod.zig`
- Move: `src/cpu.zig` → `src/platform/cpu.zig`
- Move: `src/shared/platform.zig` → `src/platform/detection.zig`
- Create: `src/platform/stub.zig`

**Step 1: Create platform/mod.zig**

```zig
//! Platform Detection and Abstraction
//!
//! Provides OS, architecture, and capability detection for cross-platform code.

const std = @import("std");
const builtin = @import("builtin");

pub const cpu = @import("cpu.zig");
pub const detection = @import("detection.zig");

// Re-export common types
pub const Os = detection.Os;
pub const Arch = detection.Arch;
pub const PlatformInfo = detection.PlatformInfo;

/// Get current platform information
pub fn getPlatformInfo() PlatformInfo {
    return PlatformInfo.detect();
}

/// Check if current platform supports threading
pub fn supportsThreading() bool {
    return detection.is_threaded_target;
}

/// Get CPU count in a platform-safe manner
pub fn getCpuCount() usize {
    return detection.getCpuCountSafe();
}

test "platform module" {
    const info = getPlatformInfo();
    try std.testing.expect(info.max_threads >= 1);
}
```

**Step 2: Move cpu.zig to platform/**

Run: `mv src/cpu.zig src/platform/cpu.zig`

**Step 3: Move platform.zig to platform/detection.zig**

Run: `mv src/shared/platform.zig src/platform/detection.zig`

**Step 4: Update imports in platform/cpu.zig**

Change any `@import("../shared/platform.zig")` to `@import("detection.zig")`

**Step 5: Create platform/stub.zig**

```zig
//! Platform stub for disabled builds
pub const Os = enum { unknown };
pub const Arch = enum { unknown };
pub const PlatformInfo = struct {
    os: Os = .unknown,
    arch: Arch = .unknown,
    max_threads: u32 = 1,
    pub fn detect() PlatformInfo { return .{}; }
};
pub fn getPlatformInfo() PlatformInfo { return .{}; }
pub fn supportsThreading() bool { return false; }
pub fn getCpuCount() usize { return 1; }
```

**Step 6: Run tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 7: Commit**

```bash
git add src/platform/
git commit -m "refactor: create platform/ module with cpu and detection"
```

---

## Task 2: Move io.zig to shared/

**Files:**
- Move: `src/io.zig` → `src/shared/io.zig`
- Modify: `src/shared/mod.zig`

**Step 1: Move io.zig**

Run: `mv src/io.zig src/shared/io.zig`

**Step 2: Update shared/mod.zig exports**

Add to `src/shared/mod.zig`:
```zig
pub const io = @import("io.zig");
```

**Step 3: Update all imports of io.zig**

Run: `rg '@import\("io\.zig"\)' src/ --files-with-matches`

Update each file to use `@import("shared/io.zig")` or through shared module.

**Step 4: Run tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 5: Commit**

```bash
git add src/shared/io.zig src/shared/mod.zig
git commit -m "refactor: move io.zig to shared/"
```

---

## Task 3: Create framework/ Directory

**Files:**
- Create: `src/framework/mod.zig`
- Move: `src/framework.zig` → `src/framework/orchestration.zig`
- Create: `src/framework/lifecycle.zig`
- Create: `src/framework/stub.zig`

**Step 1: Create framework directory**

Run: `mkdir -p src/framework`

**Step 2: Move framework.zig**

Run: `mv src/framework.zig src/framework/orchestration.zig`

**Step 3: Create framework/mod.zig**

```zig
//! Framework Module
//!
//! Provides framework lifecycle management and feature orchestration.

pub const orchestration = @import("orchestration.zig");
pub const lifecycle = @import("lifecycle.zig");

// Re-export main types
pub const Framework = orchestration.Framework;
pub const FrameworkBuilder = orchestration.FrameworkBuilder;
pub const State = orchestration.State;

pub fn init(allocator: std.mem.Allocator, config: anytype) !Framework {
    return orchestration.Framework.init(allocator, config);
}
```

**Step 4: Create framework/lifecycle.zig**

Extract lifecycle management code from orchestration.zig or create new:
```zig
//! Framework Lifecycle Management
const std = @import("std");

pub const LifecycleState = enum {
    uninitialized,
    initializing,
    running,
    stopping,
    stopped,
    failed,
};

pub const LifecycleHook = struct {
    on_init: ?*const fn() anyerror!void = null,
    on_start: ?*const fn() anyerror!void = null,
    on_stop: ?*const fn() void = null,
    on_error: ?*const fn(anyerror) void = null,
};
```

**Step 5: Update src/abi.zig imports**

Change `@import("framework.zig")` to `@import("framework/mod.zig")`

**Step 6: Run tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 7: Commit**

```bash
git add src/framework/
git commit -m "refactor: convert framework.zig to framework/ directory"
```

---

## Task 4: Create cli/ Module (Move from tools/)

**Files:**
- Create: `src/cli/mod.zig`
- Move: `src/main.zig` → `src/cli/main.zig`
- Move: `tools/cli/commands/` → `src/cli/commands/`
- Move: `tools/cli/tui/` → `src/cli/tui/`
- Create: `src/cli/stub.zig`

**Step 1: Create cli directory structure**

Run: `mkdir -p src/cli/commands src/cli/tui`

**Step 2: Move main.zig**

Run: `mv src/main.zig src/cli/main.zig`

**Step 3: Move commands**

Run: `mv tools/cli/commands/*.zig src/cli/commands/`

**Step 4: Move TUI**

Run: `mv tools/cli/tui/*.zig src/cli/tui/`

**Step 5: Create cli/mod.zig**

```zig
//! CLI Module
//!
//! Command-line interface for the ABI framework.

pub const commands = @import("commands/mod.zig");
pub const tui = @import("tui/mod.zig");

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    return commands.dispatch(allocator, args);
}
```

**Step 6: Update build.zig**

Change CLI executable source from `tools/cli/main.zig` to `src/cli/main.zig`

**Step 7: Update all CLI command imports**

Update paths in each command file from `@import("../../src/...")` to `@import("../...")`

**Step 8: Run CLI smoke test**

Run: `zig build run -- --help`
Expected: Help output displays correctly

**Step 9: Run full tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 10: Commit**

```bash
git add src/cli/ build.zig
git rm -r tools/cli/commands tools/cli/tui tools/cli/main.zig
git commit -m "refactor: move CLI from tools/ to src/cli/"
```

---

## Task 5: Create integrations/ Module

**Files:**
- Create: `src/integrations/mod.zig`
- Move: `src/connectors/` → `src/integrations/connectors/`
- Move: `src/cloud/` → `src/integrations/cloud/`
- Create: `src/integrations/stub.zig`

**Step 1: Create integrations directory**

Run: `mkdir -p src/integrations`

**Step 2: Move connectors**

Run: `mv src/connectors src/integrations/connectors`

**Step 3: Move cloud**

Run: `mv src/cloud src/integrations/cloud`

**Step 4: Create integrations/mod.zig**

```zig
//! Integrations Module
//!
//! External service connectors and cloud function adapters.

const build_options = @import("build_options");

pub const connectors = @import("connectors/mod.zig");
pub const cloud = @import("cloud/mod.zig");

// Re-export common connectors
pub const openai = connectors.openai;
pub const ollama = connectors.ollama;
pub const anthropic = connectors.anthropic;
pub const huggingface = connectors.huggingface;

// Re-export cloud adapters
pub const aws_lambda = cloud.aws_lambda;
pub const azure_functions = cloud.azure_functions;
pub const gcp_functions = cloud.gcp_functions;
```

**Step 5: Update src/abi.zig**

Change:
```zig
pub const connectors = @import("connectors/mod.zig");
pub const cloud = @import("cloud/mod.zig");
```
To:
```zig
pub const integrations = @import("integrations/mod.zig");
pub const connectors = integrations.connectors;
pub const cloud = integrations.cloud;
```

**Step 6: Run tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 7: Commit**

```bash
git add src/integrations/
git commit -m "refactor: consolidate connectors and cloud into integrations/"
```

---

## Task 6: Update abi.zig with New Structure

**Files:**
- Modify: `src/abi.zig`

**Step 1: Update all imports to new paths**

```zig
// Core
pub const framework = @import("framework/mod.zig");
pub const config = @import("config/mod.zig");
pub const platform = @import("platform/mod.zig");
pub const shared = @import("shared/mod.zig");
pub const runtime = @import("runtime/mod.zig");

// Domain modules
pub const ai = @import("ai/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const database = @import("database/mod.zig");
pub const network = @import("network/mod.zig");
pub const observability = @import("observability/mod.zig");
pub const web = @import("web/mod.zig");

// CLI (internal, not part of library API)
const cli = @import("cli/mod.zig");

// Integrations
pub const integrations = @import("integrations/mod.zig");
pub const connectors = integrations.connectors;
pub const cloud = integrations.cloud;

// Support modules
pub const ha = @import("ha/mod.zig");
pub const registry = @import("registry/mod.zig");
pub const tasks = @import("tasks/mod.zig");
```

**Step 2: Run full validation**

Run: `zig fmt . && zig build test --summary all`
Expected: Format clean, 787+ tests pass

**Step 3: Commit**

```bash
git add src/abi.zig
git commit -m "refactor: update abi.zig with new module structure"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/intro.md`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md Architecture section**

Update the directory tree to reflect new structure.

**Step 2: Update docs/intro.md**

Update module descriptions and import examples.

**Step 3: Update CONTRIBUTING.md**

Update file navigation and module organization sections.

**Step 4: Update README.md**

Update quick start examples if they reference old paths.

**Step 5: Commit**

```bash
git add CLAUDE.md docs/intro.md CONTRIBUTING.md README.md
git commit -m "docs: update documentation for new src/ structure"
```

---

## Task 8: Final Validation

**Step 1: Format all code**

Run: `zig fmt .`
Expected: No formatting changes

**Step 2: Run full test suite**

Run: `zig build test --summary all`
Expected: 787+ tests pass, 5 skipped

**Step 3: Run CLI smoke tests**

Run: `zig build cli-tests`
Expected: All CLI commands work

**Step 4: Run benchmarks**

Run: `zig build bench-competitive`
Expected: Benchmarks complete without errors

**Step 5: Build all backends**

Run: `for backend in auto metal vulkan cuda stdgpu webgpu opengl fpga none; do zig build -Dgpu-backend=$backend; done`
Expected: All backends compile

**Step 6: Final commit**

```bash
git add -A
git commit -m "refactor: complete src/ restructure to layered Zig 0.16 layout"
git push origin main
```

---

## Rollback Plan

If issues arise:
1. `git revert HEAD~N` to revert commits
2. Or restore from backup branch: `git checkout -b backup-pre-restructure HEAD~N`

---

## Success Criteria

- [ ] All 787+ tests pass
- [ ] All CLI commands work
- [ ] All GPU backends compile
- [ ] Documentation updated
- [ ] No import errors
- [ ] Clean `zig fmt .` output
