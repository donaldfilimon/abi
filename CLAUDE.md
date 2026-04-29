# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Would an agent likely miss this without help? See AGENTS.md for General Next Steps guidance.

## Project Overview

ABI is a Zig 0.17-dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion`. `tools/zigly` is the repo entrypoint and prefers `~/.zvm/bin/zig` when its actual version matches the pin:

```bash
zvm use --sync              # Sync ZVM with the repo pin
tools/zigly --status        # Print the pinned zig path
tools/zigly --link          # Symlink zig + zls into ~/.local/bin
tools/zigly --bootstrap     # One-command project setup
```

## Build Commands

### macOS 26.4+ (Darwin 25.x)

Zig's internal LLD linker cannot link on macOS 26.4+. Use `./build.sh` which auto-relinks with Apple's native linker:

```bash
./build.sh                         # Build library
./build.sh test --summary all      # Run all tests
./build.sh cli                     # Build CLI binary
./build.sh check                   # Full gate (lint + test + parity)
./build.sh check-parity            # Verify mod/stub declaration parity
```

### Linux / Older macOS

```bash
zig build                          # Build static library
zig build test --summary all       # Run tests
zig build test -- --test-filter "pattern"  # Run single test by name
zig build check                    # Lint + test + stub parity
zig build lint                     # Check formatting
zig build fix                      # Auto-format
zig build check-parity             # Verify mod/stub declaration parity
zig build feature-tests            # Run feature integration tests
zig build full-check               # Full validation gate
zig build verify-all               # Release verification
zig build cross-check              # Verify cross-compilation
zig build lib                      # Build static library artifact
zig build mcp                      # Build MCP stdio server
zig build cli                      # Build CLI binary
zig build typecheck                # Compile-only validation
```

### Focused Test Lanes (27 total)

Each runs unit + integration tests for a specific feature:

```bash
zig build acp-tests agents-tests auth-tests cache-tests cloud-tests
zig build compute-tests connectors-tests database-tests desktop-tests
zig build documents-tests gateway-tests gpu-tests ha-tests inference-tests
zig build lsp-tests messaging-tests multi-agent-tests network-tests
zig build observability-tests orchestration-tests pipeline-tests pitr-tests
zig build search-tests secrets-tests storage-tests tasks-tests web-tests
```

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle
- `src/features/` — 21 feature directories under mod/stub/types pattern
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord)
- `src/protocols/` — MCP, LSP, ACP, HA protocol implementations
- `src/tasks/` — Task management, async job queues
- `src/inference/` — ML inference: engine, scheduler, sampler, KV cache
- `src/main.zig` — CLI entry point (builds as `abi` binary)
- `src/mcp_main.zig` — MCP stdio server entry point
- `src/ffi.zig` — C-ABI FFI endpoints for linking as static library
- `build/` — Build helpers: flags, cross-compilation, linking, validation
- `test/` — Integration tests via `test/mod.zig` (uses `@import("abi")`)

### The Mod/Stub Pattern

Every feature under `src/features/<name>/` follows a contract:

- `mod.zig` — Real implementation
- `stub.zig` — API-compatible no-ops (same public surface, zero-cost when disabled)
- `types.zig` — Shared types used by both mod and stub

In `src/root.zig`, each feature uses comptime selection:

```zig
pub const gpu = if (build_options.feat_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");
```

**Critical rule**: When modifying a feature's public API, **both `mod.zig` and `stub.zig` must be updated in sync**. Run `zig build check-parity` to verify. The parity checker lives at `src/feature_parity_tests.zig`.

### AI Feature Structure

The `ai` feature (`src/features/ai/`) contains 33+ sub-directories:

- **Inference:** `llm/`, `embeddings/`, `vision/`, `models/`, `streaming/`
- **Reasoning:** `abbey/`, `aviva/`, `abi/`, `constitution/`, `eval/`, `reasoning/`
- **Agents:** `agents/`, `tools/`, `multi_agent/`, `coordination/`, `orchestration/`
- **Learning:** `training/`, `memory/`, `federated/`
- **Support:** `templates/`, `prompts/`, `documents/`, `profiles/`, `context_engine/`
- **Pipeline:** `pipeline/` — Composable prompt DSL with WDBX-backed steps

### Multi-Profile Pipeline (Abbey-Aviva-Abi)

The full pipeline is wired end-to-end in `src/features/ai/profile/router.zig`:

```
User Input -> Abi Analysis (sentiment + policy + rules)
  -> AdaptiveModulator (EMA user preference learning)
  -> Routing Decision (single / parallel / consensus)
  -> Profile Execution (Abbey / Aviva / Abi)
  -> Constitution Validation (6 principles)
  -> WDBX Memory Storage (cryptographic block chain)
  -> Response
```

### Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui` (both false). Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

Build options are exposed via `@import("build_options")` with fields like `feat_gpu`, `feat_ai`, `gpu_metal`, etc.

### GPU Backend Status

| Backend  | Status     | Notes                                |
| -------- | ---------- | ------------------------------------ |
| Metal    | Functional | macOS only, MPS acceleration         |
| CUDA     | Functional | NVIDIA GPUs, dynamic library loading |
| Vulkan   | Functional | Cross-platform, full pipeline        |
| stdgpu   | Functional | CPU-based SPIR-V emulation (default) |
| WebGPU   | Partial    | API structure present                |
| OpenGL   | Partial    | Compute shaders (GL 4.3+)            |
| WebGL2   | Stub       | No compute shader support            |
| FPGA/TPU | Stub       | Simulation mode only                 |

## Import Rules

- **Within `src/`**: Use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside the module — causes circular import error.
- **From `test/`**: Use `@import("abi")` and `@import("build_options")` — these are wired as named module imports by build.zig.
- **Cross-feature imports**: Never import another feature's `mod.zig` directly. Use conditional: `const obs = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports.

## Zig 0.16 Gotchas

- `ArrayListUnmanaged` init: use `.empty` not `.{}`
- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- `var` vs `const`: Compiler enforces const for never-mutated locals
- Entry points: `pub fn main(init: std.process.Init) !void` (not old `pub fn main() !void`)
- `std.mem.trimRight` renamed to `std.mem.trimEnd`
- Runtime env var access: `std.c.getenv(name.ptr)` returns `?[*:0]const u8`
- Signal handlers: use `std.posix.Sigaction` with `callconv(.c)` handler functions

Do NOT run `zig fmt .` at the repo root — use `zig build fix` which scopes to `src/`, `build.zig`, `build/`, and `test/`.

## Key Conventions

- Functions and variables: `camelCase`
- Types and structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Enum variants: `snake_case`
- Doc comments (`///`) on public API only
- Use `linkIfDarwin()` from `build/linking.zig` instead of inline macOS checks
- Database engine thread safety: every public `Engine` method must acquire `db_lock` before reading shared state
- AI pipeline memory: string literals in `ProfileResponse.content` crash on `deinit` — always `allocator.dupe()` heap copies
- Abbey emotion files: `emotions.zig` is canonical (not `emotion.zig`)

### Error Handling Convention

- `@compileError` — Compile-time contract violations only
- `@panic` — Unrecoverable invariant violations; only in CLI entry points and tests, never in library code
- `unreachable` — Provably impossible branches where the compiler can verify exhaustiveness
- Error unions — All runtime failure paths in library code; prefer `error.FeatureDisabled` in stubs

### Testing

Two test suites run under `zig build test`:

1. **Unit tests** (`src/root.zig`) — `refAllDecls` walks the entire module tree
2. **Integration tests** (`test/mod.zig`) — imports `@import("abi")` as external consumer

Most files end with:

```zig
test {
    std.testing.refAllDecls(@This());
}
```

**Known pre-existing failures**: inference engine connector backend tests (2), auth integration tests (1 failure, 3 leaks) — not regressions.

### refAllDecls Convention

Files with known pre-existing sub-module errors (refAllDecls deferred):

- `features/ai/abbey/mod.zig` — abbey_train.zig, config.zig
- `features/cloud/mod.zig` — aws_lambda, azure_functions, gcp_functions
- `features/gpu/mod.zig` — stdgpu, diagnostics, profiling, recovery
- `features/network/mod.zig` — linking/, raft_transport, scheduler, tcp, unified_memory
- `features/web/mod.zig` — middleware/auth, profile_routes, server/
- `foundation/utils.zig` — memory/stack.zig, memory/thread_cache.zig

## CLI Commands

Build with `zig build cli` (or `./build.sh cli`). Binary: `zig-out/bin/abi`.

```bash
abi                    # Smart status (feature count, enabled/disabled tags)
abi version            # Version and build info
abi doctor             # Build config report (all feature flags + GPU backends)
abi features           # List all 60 features from catalog with [+]/[-] status
abi platform           # Platform detection (OS, arch, CPU, GPU backends)
abi connectors         # List 16 LLM provider connectors with env var status
abi search <sub>       # Full-text search (create, index, query, delete, stats)
abi info               # Framework architecture summary
abi chat <message...>  # Route through multi-profile pipeline
abi db <subcommand>    # Vector database (add, query, stats, diagnostics, optimize, backup, restore, serve)
abi serve              # Start ACP HTTP server (default 127.0.0.1:8080)
abi dashboard          # Developer diagnostics shell (requires -Dfeat-tui=true)
abi help               # Full help reference
```

## MCP Server

`zig build mcp` produces `zig-out/bin/abi-mcp`, a JSON-RPC 2.0 stdio server exposing database and ZLS tools. Entry point: `src/mcp_main.zig`. After rebuilding, restart Claude Code to pick up the new binary.

## Workflow

- Read `tasks/lessons.md` at the start of every session
- Update `tasks/todo.md` before implementation for non-trivial tasks
- Run `zig build check-parity` after ANY public API change
- Verification gate before marking done: `./build.sh full-check` (or `zig build full-check` on Linux)
- Conventional Commits required
- **No `rm`**: Use safe alternatives

## Available Agents (in `.claude/agents/`)

- **abi-stub-fixer** — Updates `stub.zig` files when `mod.zig` public APIs change
- **darwin-build-doctor** — Diagnoses Zig linker failures on macOS 25+
- **abi-expert** — General ABI framework guidance
- **stub-parity-reviewer** — Reviews mod.zig/stub.zig pairs for API mismatches
- **feature-scaffolder** — Scaffolds new feature modules
- **build-troubleshooter** — Diagnoses Zig build failures
- **abi-test-writer** — Writes integration tests following conventions
- **abbey-aviva-abi-architect** — Multi-profile AI pipeline expert

Invoke via `Agent` tool with `subagent_type: "<agent-name>"`.

## Available Skills (in `.claude/skills/`)

- **lessons-review** — Reviews `tasks/lessons.md` for recurring pitfalls
- **stub-audit** — Verifies AI sub-feature stubs match their mod.zig public API
- **cross-check** — Runs cross-compilation verification
- **baseline-sync** — Tracks test pass/skip counts and reports drift
- **full-check** — One-command full validation gate
- **pre-commit-check** — Run lint + parity check before committing

Invoke via `Skill` tool with `skill: "<skill-name>"`.
