# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Before Making Changes

This codebase frequently has work-in-progress changes. Before modifying:
1. Run `git status` and `git diff --stat` to understand existing changes
2. Ask the user about uncommitted work before adding to it

## TL;DR for Common Tasks

```bash
# Most common workflow
zig build test --summary all && zig fmt .   # Test + format (always run before commits)

# Single file test (faster iteration — use zig test, NOT zig build test)
zig test src/path/to/file.zig --test-filter "specific test"

# Build with specific features
zig build -Denable-ai=true -Dgpu-backend=vulkan

# Full CI gate
zig build full-check    # Format + tests + feature-tests + flag validation + CLI smoke
```

## Build Commands

```bash
zig build                              # Build the project
zig build test --summary all           # Main tests: 1290 pass, 6 skip (1296 total)
zig build feature-tests --summary all  # Feature tests: 2360 pass (2365 total)
zig build full-check                   # Format + tests + feature-tests + flag validation + CLI smoke
zig build verify-all                   # Release gate (full-check + consistency + examples + wasm)
zig build validate-flags               # Check 34 feature flag combinations
zig fmt .                              # Format code
zig build lint                         # Check formatting (CI uses this)
zig build run -- --help                # CLI help
zig build examples                     # Build all examples
zig build check-wasm                   # Check WASM compilation
```

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` syntax | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| Stub/Real module sync | Changes to `mod.zig` must be mirrored in `stub.zig` with identical signatures |
| No `@import("abi")` inside `src/features/` | Causes circular imports — use relative imports only |
| v0.4.0 facade removal | `abi.inference` → `abi.ai.llm`, `abi.training` → `abi.ai.training`, `abi.reasoning` → `abi.ai.orchestration` |
| Feature disabled errors | Rebuild with `-Denable-<feature>=true` |
| `observability` flag name | Uses `-Denable-profiling`, NOT `-Denable-observability` |
| `ai.orchestration` flag name | Uses `-Denable-reasoning`, NOT `-Denable-orchestration` |
| libc linking | CLI and examples require libc for environment variable access |
| Import paths | Always use `@import("abi")` for public API, not direct file paths |
| `build.zig` file checks | Use existing `pathExists()` helper, not `std.fs.cwd()` |
| WASM limitations | `database`, `network`, `gpu` auto-disabled; no `std.Io.Threaded` |
| macOS 26 Xcode-beta | Set `DEVELOPER_DIR=/Library/Developer/CommandLineTools` in `~/.zshenv` |

**Zig 0.16 gotchas** are in `.claude/rules/zig.md` (auto-loaded for `.zig` files). Key ones: `std.Io.Dir.cwd()` not `std.fs.cwd()`, `.empty` init for ArrayList/HashMap, `{t}` format specifier for enums/errors, `catch {` not `catch |_| {`.

## Architecture

Three-tier structure: **core** (always available), **services** (always available), **features** (comptime-gated via `build_options`).

```
src/
├── abi.zig                    # Public API entry point — all imports go through here
├── core/                      # Always available: framework infrastructure
│   ├── config/mod.zig         # Unified Config struct with builder pattern
│   ├── framework.zig          # Framework orchestration (init, shutdown, builder)
│   ├── registry/mod.zig       # Feature registry (comptime, runtime, dynamic)
│   ├── errors.zig             # Composable error hierarchy
│   └── feature_catalog.zig    # Canonical feature list (source of truth for flags/parity)
├── services/                  # Always available: shared infrastructure
│   ├── runtime/               # Work-stealing task execution, scheduling, concurrency
│   ├── platform/              # OS/arch detection, SIMD, CPU features
│   ├── shared/                # Utilities (SIMD, logging, security, crypto, JSON, etc.)
│   ├── connectors/            # External AI providers (OpenAI, Anthropic, Ollama, etc.)
│   ├── ha/                    # High availability (replication, backup, PITR)
│   ├── tasks/                 # Task management
│   ├── lsp/                   # ZLS client utilities
│   ├── mcp/                   # Model Context Protocol server (WDBX)
│   ├── acp/                   # Agent Communication Protocol
│   └── tests/mod.zig          # Main test root (1290 pass, 6 skip)
├── features/                  # Comptime-gated: each has mod.zig + stub.zig
│   ├── ai/                    # AI (24 submodules: llm, agents, training, streaming, etc.)
│   ├── gpu/                   # GPU acceleration (Vulkan, CUDA, Metal, etc.)
│   ├── database/              # Vector database (WDBX with HNSW/IVF-PQ)
│   ├── network/               # Distributed compute and Raft consensus
│   ├── observability/         # Metrics, tracing, profiling, pages
│   ├── web/                   # Web/HTTP utilities
│   ├── cloud/                 # Cloud function adapters (AWS, Azure, GCP)
│   ├── analytics/             # Event tracking
│   ├── auth/                  # Authentication and security
│   ├── messaging/             # Event bus
│   ├── cache/                 # In-memory caching
│   ├── storage/               # Unified file/object storage
│   ├── search/                # Full-text search
│   ├── gateway/               # API gateway (routing, rate limiting, circuit breaker)
│   ├── mobile/                # Mobile platform (Android/iOS)
│   └── benchmarks/            # Performance benchmarking
├── feature_test_root.zig      # Feature test root (2360 pass, 5 skip)
tools/
├── cli/                       # CLI executable
│   ├── main.zig               # Entry point (uses Init.Minimal)
│   ├── commands/              # 35 command implementations
│   │   ├── llm/               # LLM subcommands (chat, demo, list)
│   │   ├── train/             # Training subcommands (info, monitor, self)
│   │   ├── ralph/             # Ralph agent subcommands (init, run, status, skills)
│   │   ├── bench/             # Benchmark subcommands (micro, suites, training)
│   │   └── ui/                # TUI subcommands (brain, gpu, network, streaming, etc.)
│   ├── tui/                   # TUI panels and rendering (async_loop, widgets, terminal)
│   └── utils/output.zig       # CLI output formatting (NO_COLOR, colors, structured output)
├── scripts/                   # Build and consistency scripts
│   └── baseline.zig           # Test count baselines (source of truth)
build/
├── options.zig                # BuildOptions struct (22 feature flags)
├── modules.zig                # Module creation helpers
├── flags.zig                  # Flag parsing and validation
├── targets.zig                # Build target configuration
├── gpu.zig                    # GPU backend selection
└── link.zig                   # Platform-specific linking (Metal, libc, etc.)
```

### Key Architectural Concepts

**Comptime feature gating:** Each feature in `src/features/` has `mod.zig` (real) and `stub.zig` (disabled). `src/abi.zig` selects which to import based on `build_options.enable_<feature>`. Both must export identical public APIs.

**Feature catalog:** `src/core/feature_catalog.zig` is the canonical source of truth for all features, their compile flags, parent-child relationships, and mod/stub paths.

**Import rules:**
- External code: `@import("abi")` — never import source files directly
- Inside `src/features/`: relative imports only (no `@import("abi")` — causes circular imports)
- AI submodule internals: import from `../../core/mod.zig` for shared types

**Table-driven build:** `build.zig` uses `BuildTarget` struct arrays for examples and benchmarks. Add entries to `example_targets` or `benchmark_targets` rather than duplicating build code.

**Two test roots:**
- `zig build test` → `src/services/tests/mod.zig` (main: 1290 pass, 6 skip (1296 total))
- `zig build feature-tests` → `src/feature_test_root.zig` (inline: 2360 pass (2365 total))
- Baselines tracked in `tools/scripts/baseline.zig`

## Feature Flags

All flags default to `true` except `enable_mobile`.

| Flag | Feature Module | Notes |
|------|---------------|-------|
| `-Denable-gpu` | `features/gpu/` | |
| `-Denable-ai` | `features/ai/` | Parent of llm, training, explore, vision |
| `-Denable-llm` | AI sub-feature | Requires `-Denable-ai` |
| `-Denable-training` | AI sub-feature | Requires `-Denable-ai` |
| `-Denable-reasoning` | `ai/orchestration/` | Maps to orchestration, not "reasoning" |
| `-Denable-database` | `features/database/` | |
| `-Denable-network` | `features/network/` | |
| `-Denable-profiling` | `features/observability/` | Maps to observability, not "profiling" |
| `-Denable-web` | `features/web/` | |
| `-Denable-analytics` | `features/analytics/` | |
| `-Denable-cloud` | `features/cloud/` | |
| `-Denable-auth` | `features/auth/` | |
| `-Denable-messaging` | `features/messaging/` | |
| `-Denable-cache` | `features/cache/` | |
| `-Denable-storage` | `features/storage/` | |
| `-Denable-search` | `features/search/` | |
| `-Denable-gateway` | `features/gateway/` | |
| `-Denable-pages` | `observability/pages/` | |
| `-Denable-benchmarks` | `features/benchmarks/` | |
| `-Denable-mobile` | `features/mobile/` | Default: **false** |
| `-Denable-explore` | AI sub-feature | Internal flag (derived from ai) |
| `-Denable-vision` | AI sub-feature | Internal flag (derived from ai) |

### GPU Backends

```bash
zig build -Dgpu-backend=vulkan              # Single backend
zig build -Dgpu-backend=cuda,vulkan         # Multiple (comma-separated)
zig build -Dgpu-backend=auto                # Auto-detect
zig build -Dgpu-backend=none                # Disable all
```

Available: `none`, `auto`, `cuda`, `vulkan`, `stdgpu`, `metal`, `webgpu`, `opengl`, `opengles`, `webgl2`, `fpga`

## Common Workflows

### Adding a new public API function
1. Add to `src/features/<feature>/mod.zig`
2. Mirror identical signature in `src/features/<feature>/stub.zig`
3. Run `zig build test` to verify both paths compile

### Adding a new feature module
1. Create `src/features/<name>/mod.zig` and `stub.zig` (identical public APIs)
2. Add to `src/core/feature_catalog.zig` metadata array
3. Add `enable_<name>` flag to `build/options.zig` `BuildOptions` struct
4. Add conditional import in `src/abi.zig`
5. Add config entry to `src/core/config/mod.zig` if needed
6. Register in `src/core/registry/` for runtime toggling

### Adding a new example
1. Add entry to `example_targets` array in `build.zig`
2. Create the example file in `examples/`
3. Run `zig build examples` to verify

### Writing tests
- Use `error.SkipZigTest` for hardware-gated tests (GPU, network)
- Test discovery: `test { _ = @import(...); }` — `comptime {}` does NOT work
- End every file with: `test { std.testing.refAllDecls(@This()); }`
- For I/O in tests: use `std.testing.io` and `std.testing.allocator`

### Verifying stub parity after API changes
```bash
zig build -Denable-<feature>=true    # Real module compiles
zig build -Denable-<feature>=false   # Stub module compiles
```

### GPU backend verification
```bash
for backend in auto metal vulkan cuda stdgpu webgpu opengl fpga none; do
  zig build -Dgpu-backend=$backend || echo "FAIL: $backend"
done
```

## Configuration System

```zig
const abi = @import("abi");

// Builder pattern
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
    .withDatabase(.{ .path = "./data" })
    .build();
defer fw.deinit();

// Or use defaults
var fw = try abi.initDefault(allocator);
defer fw.deinit();
```

Modular configs in `src/core/config/`: `ai.zig`, `gpu.zig`, `database.zig`, `network.zig`, `observability.zig`, `web.zig`, `cloud.zig`, `plugin.zig`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ABI_GPU_BACKEND` | GPU backend override (auto, cuda, vulkan, metal, none) |
| `ABI_LLM_MODEL_PATH` | Path to LLM model file |
| `ABI_OPENAI_API_KEY` | OpenAI API key |
| `ABI_ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_HF_API_TOKEN` | HuggingFace token |
| `ABI_MASTER_KEY` | 32-byte key for secrets encryption (required in production) |
| `ABI_DB_PATH` | Database file path (default: `abi.db`) |
| `DEVELOPER_DIR` | macOS: override Xcode SDK path (set to `/Library/Developer/CommandLineTools` for macOS 26) |

## Developer CLI Commands

Quick-reference for the developer-focused CLI commands:

```bash
abi doctor             # Health check: Zig, framework, GPU, API keys, features
abi clean              # Remove .zig-cache/ (add --state for .ralph/, --all --force for models)
abi env                # List ABI_* environment variables (redacted)
abi env validate       # Check AI providers, GPU, master key
abi env export         # Print export commands for shell sourcing
abi init <name>        # Scaffold project (templates: default, llm-app, agent, training)
```

## Zig 0.16 Quick Reference

**Pinned version:** `0.16.0-dev.2653+784e89fd4` (see `.zigversion`)

Full Zig 0.16 patterns are in `.claude/rules/zig.md` (auto-loaded). Essential patterns:

```zig
// I/O backend (required for all file/network ops)
var io_backend = std.Io.Threaded.init(allocator, .{});
defer io_backend.deinit();
const io = io_backend.io();

// File read/write
const dir = std.Io.Dir.cwd();
const data = try dir.readFileAlloc(io, "file.txt", allocator, .limited(1 << 20));
try dir.writeFile(io, .{ .sub_path = "out.txt", .data = content });

// Collections: .empty init, allocator per-call
var list: std.ArrayListUnmanaged(T) = .empty;
defer list.deinit(allocator);
try list.append(allocator, item);
```

## CLI Output Convention

All CLI commands must use `tools/cli/utils/output.zig` instead of raw `std.debug.print`. This provides:
- Colored error/warning/info/success prefixes
- Structured key-value formatting
- `NO_COLOR` environment variable compliance

```zig
const utils = @import("../utils/mod.zig");   // from commands/*.zig
const utils = @import("../../utils/mod.zig"); // from commands/llm/*.zig, commands/train/*.zig, etc.

// Instead of: std.debug.print("Error: {s}\n", .{msg});
utils.output.printError("something failed: {s}", .{msg});
utils.output.printWarning("Check config", .{});
utils.output.printInfo("Processing...", .{});
utils.output.printSuccess("Done!", .{});
utils.output.printHeader("Section Title");
utils.output.printKeyValue("Name", value);
utils.output.printKeyValueFmt("Count", "{d}", .{n});
```

**Exceptions:** `completions.zig` (shell eval output must stay on stderr) and `DebugWriter` callbacks.

## Code Style

| Rule | Convention |
|------|------------|
| Indentation | 4 spaces, no tabs |
| Types | `PascalCase` |
| Functions/Variables | `camelCase` |
| Imports | Explicit only (no `usingnamespace`) |
| Cleanup | Prefer `defer`/`errdefer` |
| ArrayList | Prefer `std.ArrayListUnmanaged` with explicit allocator passing |
| CLI output | Use `utils.output.*`, never raw `std.debug.print` in commands |

## Quick File Navigation

| Task | Key Files |
|------|-----------|
| Add CLI command | `tools/cli/commands/` + register in `tools/cli/main.zig` |
| Add feature module | `src/features/<name>/` + `feature_catalog.zig` + `build/options.zig` + `src/abi.zig` |
| Add connector | `src/services/connectors/<provider>.zig` |
| Add GPU backend | `src/features/gpu/backends/` + `src/features/gpu/dsl/codegen/configs/` |
| Modify public API | `src/abi.zig` (entry point) |
| Add example | `examples/` + `example_targets` array in `build.zig` |
| Test baselines | `tools/scripts/baseline.zig` |
| Feature catalog | `src/core/feature_catalog.zig` |

## Consistency Checks

`zig build full-check` and `zig build verify-all` validate marker strings across docs. These exact literals must stay aligned with `tools/scripts/baseline.zig` and `.zigversion`:

| Marker | Where Validated |
|--------|----------------|
| `0.16.0-dev.2653+784e89fd4` | README.md, CONTRIBUTING.md, CLAUDE.md |
| `1290 pass, 6 skip (1296 total)` | CLAUDE.md, `.claude/rules/zig.md` |
| `2360 pass (2365 total)` | CLAUDE.md, `.claude/rules/zig.md` |

If test counts change, run `/baseline-sync` to update all markers.

## Claude Code Tooling

### Skills (invoke with `/skill-name`)

| Skill | Purpose |
|-------|---------|
| `/ci-gate` | Run quality gates (quick, full, verify) with failure diagnosis |
| `/parity-check` | Verify mod.zig/stub.zig signature parity across all features |
| `/baseline-sync` | Update test baselines after test count changes |
| `/new-feature` | Scaffold a feature module through all 9 integration points |
| `/cli-add-command` | Scaffold a new CLI command with registration |
| `/zig-build` | Build, test, format pipeline |
| `/zig-migrate` | Apply Zig 0.16 migration patterns to old code |
| `/zig-std` | Look up Zig 0.16 std lib APIs from actual source |
| `/connector-add` | Scaffold a new LLM provider connector |
| `/super-ralph` | Run autonomous Ralph agent loop |

### Agents (auto-triggered via Task tool)

| Agent | Trigger |
|-------|---------|
| `stub-parity` | After editing mod.zig or stub.zig |
| `feature-module` | When scaffolding new features |
| `ci-triage` | After build/test/CI failures |
| `zig-expert` | Deep Zig 0.16 reasoning with ABI context |
| `security-auditor` | Security review of code changes |
| `doc-reviewer` | Verify docs match codebase |
| `test-coverage` | Map test coverage gaps |

### Hooks (auto-run in `.claude/settings.json`)

- **PreToolUse:** Blocks `@import("abi")` in `features/` (circular import prevention)
- **PostToolUse:** Auto-formats `.zig` files, warns on mod/stub edits, detects baseline drift, guards test discovery patterns

## Post-Edit Checklist

```bash
zig fmt .                              # Format code (also auto-runs via hook)
zig build test --summary all           # Run main tests
zig build feature-tests --summary all  # Run feature tests (if touching features/)
zig build lint                         # Verify formatting passes CI
```

If modifying feature module APIs:
1. Update both `mod.zig` and `stub.zig` with identical signatures
2. Verify: `zig build -Denable-<feature>=true && zig build -Denable-<feature>=false`
3. Or run `/parity-check` to audit all modules at once

If changing test counts: run `/baseline-sync` to update markers.
