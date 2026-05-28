# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Build & Validation
- `./build.sh check` – Primary validation gate for build integrity and API parity checks.
- `./build.sh full-check` – Runs `check`, integration tests, benchmarks, and TUI smoke.
- `./build.sh cli` – Builds the main executable (`abi`).
- `./build.sh mcp` – Builds the MCP server binary.
- `zig build lint` – Runs `zig fmt --check` on all source files for formatting compliance.
- `zig build fix` – Automatically formats source files based on project standards.
- `zig build check-parity` – Verifies top-level public declaration-name parity for feature/plugin `mod.zig` and `stub.zig` pairs.
- Run a single test: `zig build test -- --test-filter "<pattern>"`

### Running Tests
- `zig build test-integration` – Executes the integration test suite.
- `zig build benchmarks` – Runs the benchmark suite.
- `zig build test` – Module + connector tests.
- `zig build test-feature-contracts` – Feature module contracts.
- `zig build test-contracts` – Surface/MCP/plugin/docs contracts.

## Architecture Overview

The ABI framework is a modular Zig codebase with a clear separation of concerns across the following layers:

| Layer | Path | Responsibility |
|-------|------|----------------|
| **Public API** | `src/root.zig` | Exposes the `abi` module to consumers. This is the primary entry point. |
| **CLI** | `src/main.zig`, `src/abi_cli/` | Parses command-line arguments and delegates to sub-commands. |
| **MCP Server** | `src/mcp/main.zig` | Implements a JSON-RPC 2.0 server over stdio and optional HTTP/SSE transport. |
| **Feature Selection** | `src/features/mod.zig` | Enables/disables features via Zig build options (`-Dfeat-*`). Uses the *mod/stub* pattern. |
| **AI Sub-system** | `src/features/ai/` | Implements AI profiles (Abbey, Aviva, Abi), routing, and constitution. |
| **Vector Store (WDBX)** | `src/features/wdbx/` | Provides in-memory key-value and vector storage with HNSW index and MVCC-style snapshot chain. |
| **GPU Backend** | `src/features/gpu/` | Reports GPU status, attempts Metal initialization on macOS, falls back to vectorized CPU implementation. |
| **Connectors** | `src/connectors/` | Provides local/live adapters for external services (OpenAI, Anthropic, Discord, Twilio, HTTP, JSON). |
| **Plugin System** | `src/plugins/`, `src/plugin_registry.zig` | Validates plugin manifests and generates metadata registry. |
| **Core Utilities** | `src/core/` + `src/foundation/` | Scheduler, memory, config, registry, time, sync, logger, IO, credentials, OS abstractions. |

### Key Areas to Focus On

- **Mod/Stub Pattern**: Ensure public API stability by checking mod/stub parity frequently. Every feature has real `mod.zig` and disabled `stub.zig`; update both when changing public APIs.
- **Build Flow**: `./build.sh check` includes contract tests plus focused feature-off and feature-aware public contracts for every `-Dfeat-*` stub.
- **Feature Flags**: 
  - Enabled by default: `feat-ai`, `feat-wdbx`, `feat-gpu`, `feat-accelerator`, `feat-shader`, `feat-mlir`, `feat-os-control`, `feat-tui`, `feat-hash`
  - Disabled by default: `feat-mobile`, `feat-metrics`
- **Import Rules**: Within `src/`, use relative `.zig` imports. `@import("abi")` is only allowed from `src/mcp/main.zig` and `src/mcp/handlers.zig`. Always include `.zig` extension on path imports.
- **CLI Contracts**: Implemented commands: `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`. Do not dispatch legacy names like `version`, `doctor`, `features`, etc.
- **MCP Contracts**: Tools: `ai_run`, `ai_complete`, `ai_train`, `wdbx_query`, `scheduler_stats`, `gpu_status`, `wdbx_stats`, `plugin_run`. HTTP defaults to `127.0.0.1:8080`; set `ABI_MCP_HTTP_PORT` to override.
- **Generated Code**: Do not manually edit `src/plugin_registry.zig`; update plugin manifests or generator code and rerun the build.
- **Zig 0.17 Patterns**: 
  - Entry: `pub fn main(init: std.process.Init) !void`
  - Use `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
  - Use `std.mem.trimEnd` (not `trimRight`)
  - Use `std.mem.splitScalar`, `splitAny`, or `splitSequence`
  - Use `foundation.time.unixMs()` for timestamps
  - Avoid silent empty `catch {}` in data, inference, or persistence paths
- **Connector Validation**: 
  - Discord: validates printable non-whitespace credentials, numeric snowflake-like IDs, author IDs, message size.
  - Twilio: validates `AC` + 32-hex account SIDs, 32-hex auth tokens, base URL, timeout, explicit `.live` transport, XML/form escaping, ConversationRelay aliases.

### Verification

Before finishing code changes, run:
```bash
./build.sh check
```

For full validation including integration tests, benchmarks, and TUI smoke:
```bash
./build.sh full-check
```

## Important Files

- `tasks/lessons.md` – Startup checklist and conventions
- `tasks/todo.md` – Current work items and known failures
- `docs/index.md` – Architecture, public API contracts, onboarding, and development guides
- `CHANGELOG.md` – Release-note style modernization highlights