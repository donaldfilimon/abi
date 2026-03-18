---
title: ABI Framework
purpose: Main project entry point and documentation map
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# ABI Framework

ABI is a Zig 0.16 framework for AI services, semantic storage, GPU acceleration,
distributed runtime features, and a large multi-command CLI. The public package
entrypoint is `src/root.zig`, exposed to consumers as `@import("abi")`.

## What ABI includes

- `abi.App` / `abi.AppBuilder` for framework setup and feature wiring
- `abi.database` for the semantic store and vector search surface
- `abi.ai` for agents, profiles, training, reasoning, and LLM support
- `abi.inference` for engine, scheduler, sampler, and paged KV cache primitives
- `abi.gpu` / `abi.Gpu` / `abi.GpuBackend` for unified compute backends
- `abi.runtime`, `abi.platform`, `abi.connectors`, `abi.mcp`, `abi.acp`, `abi.tasks` for services
- `abi.foundation` for shared utilities (SIMD, logging, security, time)
- `abi` CLI for operational workflows, diagnostics, docs generation, and local tooling

## Current baseline

| Item | Value |
|------|-------|
| Zig pin | `0.16.0-dev.2905+5d71e3051` |
| Package root | `src/root.zig` |
| Main validation gate | `zig build full-check` |
| Full release gate | `zig build verify-all` |
| Docs generator | `zig build gendocs` |
| CLI registry refresh | `zig build refresh-cli-registry` |
| Darwin 26.4 full-validation toolchain | Canonical cached host-built Zig from `./tools/scripts/bootstrap_host_zig.sh` |

## Quick start

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
zig build run -- --help
```

On macOS 26.4 / Darwin 25.x, stock prebuilt Zig on this host is linker-blocked
before `build.zig` runs. ABI's supported full-validation path is the pinned
host-built Zig produced by `./tools/scripts/bootstrap_host_zig.sh`, then
prepended to `PATH` before running direct `zig build` gates:

```bash
./tools/scripts/bootstrap_host_zig.sh
export PATH="$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin:$PATH"
hash -r
zig build toolchain-doctor
zig build full-check
zig build check-docs
```

If you are temporarily on a blocked stock toolchain, use
`./tools/scripts/run_build.sh` only as fallback evidence while replacing the
toolchain.

## Library examples

### Minimal app

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    _ = init;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var app = try abi.App.initDefault(allocator);
    defer app.deinit();

    std.debug.print("ABI {s}\n", .{abi.version()});
}
```

### Semantic store

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    _ = init;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const store = abi.database.semantic_store;

    var handle = try store.openStore(allocator, "vectors-db");
    defer store.closeStore(&handle);

    try store.storeVector(&handle, 1, &[_]f32{ 0.1, 0.2, 0.3 }, "doc-1");
    try store.storeVector(&handle, 2, &[_]f32{ 0.3, 0.2, 0.1 }, "doc-2");

    const results = try store.searchStore(&handle, allocator, &[_]f32{ 0.1, 0.2, 0.25 }, 5);
    defer allocator.free(results);

    for (results) |result| {
        std.debug.print("{d}: {d:.4}\n", .{ result.id, result.score });
    }
}
```

### Local echo agent

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    _ = init;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var agent = try abi.ai.agents.Agent.init(allocator, .{
        .name = "assistant",
        .backend = .echo,
        .enable_history = true,
    });
    defer agent.deinit();

    const response = try agent.chat("Explain comptime in one sentence.", allocator);
    defer allocator.free(response);

    std.debug.print("{s}\n", .{response});
}
```

## CLI quick reference

The CLI registry snapshot is generated from the command modules in
`tools/cli/commands/`. Common entrypoints:

```bash
# Runtime / diagnostics
abi --help
abi system-info
abi doctor
abi status

# Database
abi db stats
abi db add --id 1 --embed "hello world"
abi db query --embed "hello" --top-k 5

# AI
abi agent
abi llm chat model.gguf
abi train help
abi ralph status

# Infrastructure
abi gpu summary
abi network status
abi ui

# Tooling
abi gendocs --check
zig build refresh-cli-registry
zig build check-cli-registry
```

## Public surface map

ABI's public package surface is intentionally small at the top level and
organized by domain.

| Surface | Purpose |
|---------|---------|
| `abi.App` / `abi.AppBuilder` | Framework lifecycle and feature orchestration |
| `abi.database` | Semantic store, search, backup, restore, diagnostics |
| `abi.ai` | Agents, profiles, LLM, training, reasoning |
| `abi.inference` | Engine, scheduler, sampler, and paged KV cache primitives |
| `abi.gpu` | GPU feature namespace |
| `abi.Gpu` / `abi.GpuBackend` | Direct unified GPU runtime access |
| `abi.foundation` / `abi.runtime` | Shared foundations, time/sync/SIMD, and always-on runtime primitives |
| `abi.connectors` / `abi.ha` / `abi.tasks` / `abi.lsp` / `abi.mcp` / `abi.acp` | Service and integration surfaces |

### Notes on migration surfaces

- `abi.ai.profiles` is the canonical behavior-profile namespace.
- The public named `wdbx` package surface has been removed; use `abi.database`.
- `src/root.zig` is the canonical package root for the `abi` module. `src/abi.zig` is a legacy internal file (not imported by any code).

## Project structure

```
abi/
├── src/                  # Framework source (single "abi" module)
│   ├── root.zig          # Public package entrypoint (@import("abi"))
│   ├── abi.zig           # Internal composition layer
│   ├── core/             # Always-on framework internals
│   ├── features/         # 19 comptime-gated modules (mod/stub/types pattern)
│   ├── services/         # Runtime services, connectors, security
│   └── inference/        # ML inference: sampler, scheduler, KV cache
├── build/                # Modular build system (options, flags, modules)
├── build.zig             # Build root — all steps and gates
├── tools/                # CLI, gendocs, validation scripts
│   ├── cli/              # ABI CLI implementation
│   ├── gendocs/          # Documentation generator
│   └── scripts/          # Build helpers and consistency checks
├── tests/                # Integration test roots
├── examples/             # 35 standalone example programs
├── benchmarks/           # Performance benchmark suites
├── bindings/             # C and WASM language bindings
└── docs/                 # Maintained + generated documentation
```

Each feature module under `src/features/<name>/` follows the **mod/stub contract**:
`mod.zig` (real implementation), `stub.zig` (API-compatible no-ops), and `types.zig`
(shared types). See [docs/PATTERNS.md](docs/PATTERNS.md) for details.

## Build, test, and validation

### Core commands

```bash
zig build
zig build test --summary all
zig build feature-tests --summary all
zig build full-check
zig build verify-all
```

### Formatting and deterministic checks

```bash
zig build fix
zig build lint
./tools/scripts/fmt_repo.sh --check
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
```

Do not run `zig fmt .` at the repo root. The repo vendors upstream Zig fixtures
that intentionally contain invalid compile-error cases.

### Docs and generated artifacts

```bash
zig build gendocs
zig build gendocs -- --check --no-wasm --untracked-md
zig build check-docs
zig build refresh-cli-registry
zig build check-cli-registry
```

Generated docs live under `docs/api/` and `docs/plans/`. Structural edits should
go through `tools/gendocs/`, not direct manual edits to generated pages.
On macOS 26.4, full local `zig build gendocs` / `zig build check-docs` support
requires the pinned host-built Zig from `./tools/scripts/bootstrap_host_zig.sh`
to be first on `PATH`. If stock prebuilt Zig is linker-blocked, use
`./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence
while obtaining the bootstrapped compiler.

## Feature flags

All features default to enabled. Disable features with `-Dfeat-<name>=false`.

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

Feature definitions live in `build/options.zig`, and the source of truth for the
catalog lives in `src/core/feature_catalog.zig`.

## Toolchain notes

ABI is pinned to the Zig version in `.zigversion`. When repinning Zig, update the
pin atomically with:

- `.zigversion`
- `build.zig.zon`
- `tools/scripts/baseline.zig`
- `README.md`
- related CI / toolchain docs

This bootstrap wave does **not** repin ABI. `.zigversion`, `build.zig.zon`,
`tools/scripts/baseline.zig`, and CI remain pinned and unchanged.

On macOS 26.4, the supported local path is:

```bash
./tools/scripts/bootstrap_host_zig.sh
export PATH="$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin:$PATH"
hash -r
zig build toolchain-doctor
zig build check-zig-version
zig build full-check
zig build check-docs
zig build gendocs -- --check --no-wasm --untracked-md
```

For macOS linker issues, see [docs/ZIG_MACOS_LINKER_RESEARCH.md](docs/ZIG_MACOS_LINKER_RESEARCH.md).
For blocked Darwin hosts, use `run_build.sh`, compile-only checks, or Linux CI for binary-emitting gates.

## Benchmarks

The `benchmarks/` directory contains comprehensive performance suites covering
SIMD, memory, concurrency, database, network, crypto, AI, and GPU workloads.

```bash
zig build benchmarks                       # Run all suites
zig build benchmarks -- --suite=simd       # Run specific suite
zig build benchmarks -- --quick            # Fast CI-friendly run
zig build bench-competitive                # Industry comparisons
```

See [benchmarks/README.md](benchmarks/README.md) for suite details and
[benchmarks/STRUCTURE.md](benchmarks/STRUCTURE.md) for directory layout.

## Documentation map

### Workflow & Guidelines
- [AGENTS.md](AGENTS.md) — Contributor workflow contract (start here)
- [CLAUDE.md](CLAUDE.md) — Claude AI agent instructions
- [GEMINI.md](GEMINI.md) — Gemini CLI instructions
- [SECURITY.md](SECURITY.md) — Vulnerability reporting

### Technical Documentation
- [docs/README.md](docs/README.md) — Documentation guide (maintained vs generated)
- [docs/STRUCTURE.md](docs/STRUCTURE.md) — Full directory tree reference
- [docs/PATTERNS.md](docs/PATTERNS.md) — Zig 0.16 codebase patterns
- [docs/ABI_WDBX_ARCHITECTURE.md](docs/ABI_WDBX_ARCHITECTURE.md) — WDBX architecture
- [docs/ZIG_MACOS_LINKER_RESEARCH.md](docs/ZIG_MACOS_LINKER_RESEARCH.md) — Darwin linker notes

### Examples & Benchmarks
- [examples/README.md](examples/README.md) — Categorized example index
- [benchmarks/README.md](benchmarks/README.md) — Benchmark suite guide
- [benchmarks/STRUCTURE.md](benchmarks/STRUCTURE.md) — Benchmark directory layout

## Contributing

Before non-trivial changes:

1. Read `AGENTS.md`, `CLAUDE.md`, `tasks/todo.md`, and `tasks/lessons.md`.
2. Plan multi-file work in `tasks/todo.md`.
3. Keep feature-module `mod.zig` and `stub.zig` public surfaces aligned.
4. Run the strongest validation this environment supports and record blockers precisely.

Start with [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md).
