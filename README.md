# ABI Framework

ABI is a Zig 0.16 framework for AI services, semantic storage, GPU acceleration,
distributed runtime features, and a large multi-command CLI. The public package
entrypoint is `src/root.zig`, exposed to consumers as `@import("abi")`.

## What ABI includes

- `abi.App` / `abi.AppBuilder` for framework setup and feature wiring
- `abi.features.database` for the semantic store and vector search surface
- `abi.features.ai` for agents, profiles, training, reasoning, and LLM support
- `abi.Gpu` / `abi.GpuBackend` for unified compute backends
- `abi.services.*` for platform, connectors, MCP, ACP, tasks, and shared runtime services
- `abi` CLI for operational workflows, diagnostics, docs generation, and local tooling

## Current baseline

| Item | Value |
|------|-------|
| Zig pin | `0.16.0-dev.1503+738d2be9d` |
| Package root | `src/root.zig` |
| Main validation gate | `zig build full-check` |
| Full release gate | `zig build verify-all` |
| Docs generator | `zig build gendocs` |
| CLI registry refresh | `zig build refresh-cli-registry` |

## Quick start

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
zig build run -- --help
```

If you are on macOS 26+ and stock Zig cannot link the build runner, use one of
the repo-supported paths instead:

```bash
./tools/scripts/run_build.sh test --summary all
zig fmt --check build.zig build src tools examples
abi bootstrap-zig install
abi bootstrap-zig status
```

`abi toolchain ...` still exists as a compatibility alias for `abi bootstrap-zig ...`.

## Library examples

### Minimal app

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const store = abi.features.database.semantic_store;

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var agent = try abi.features.ai.agents.Agent.init(allocator, .{
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
abi bootstrap-zig status
abi gendocs --check
zig build refresh-cli-registry
zig build check-cli-registry
```

## Public surface map

ABI's public package surface is intentionally small at the top level and broad
under feature and service namespaces.

| Surface | Purpose |
|---------|---------|
| `abi.App` / `abi.AppBuilder` | Framework lifecycle and feature orchestration |
| `abi.features.database` | Semantic store, search, backup, restore, diagnostics |
| `abi.features.ai` | Agents, profiles, LLM, training, reasoning |
| `abi.features.gpu` | GPU feature namespace |
| `abi.Gpu` / `abi.GpuBackend` | Direct unified GPU runtime access |
| `abi.services.*` | Shared runtime services and integration surfaces |

### Notes on migration surfaces

- `abi.features.ai.profiles` is the canonical behavior-profile namespace.
- `abi.features.ai.personas` remains as a compatibility alias during the phase 4 transition.
- The public named `wdbx` package surface has been removed; use `abi.features.database`.
- `src/abi.zig` is the internal composition layer. External consumers should target `@import("abi")`, which resolves to `src/root.zig`.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/root.zig` | Public package root |
| `src/abi.zig` | Internal composition layer |
| `src/features/` | Comptime-gated feature modules with `mod.zig` / `stub.zig` parity |
| `src/core/` | Always-on framework internals |
| `src/services/` | Runtime services shared across features and tools |
| `tools/cli/` | ABI CLI implementation |
| `tools/gendocs/` | Documentation generator and templates |
| `docs/` | Maintained docs plus generated API/plan output |
| `build/` | Modular Zig build graph and validation steps |

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
zig fmt --check build.zig build src tools examples
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

For macOS linker issues, see [docs/ZIG_MACOS_LINKER_RESEARCH.md](docs/ZIG_MACOS_LINKER_RESEARCH.md).
For the repo-local bootstrap bridge, use `abi bootstrap-zig ...`.

## Documentation map

- [docs/README.md](docs/README.md) - docs tree layout and generation workflow
- [docs/FAQ-agents.md](docs/FAQ-agents.md) - repo workflow FAQ for agents and contributors
- [docs/guides/cursor_rules.md](docs/guides/cursor_rules.md) - Cursor-specific ABI rules
- [docs/ZIG_MACOS_LINKER_RESEARCH.md](docs/ZIG_MACOS_LINKER_RESEARCH.md) - Darwin linker failure notes
- [docs/ABI_WDBX_ARCHITECTURE.md](docs/ABI_WDBX_ARCHITECTURE.md) - semantic-store architecture notes

## Contributing

Before non-trivial changes:

1. Read `AGENTS.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `tasks/todo.md`, and `tasks/lessons.md`.
2. Plan multi-file work in `tasks/todo.md`.
3. Keep feature-module `mod.zig` and `stub.zig` public surfaces aligned.
4. Run the strongest validation this environment supports and record blockers precisely.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).
