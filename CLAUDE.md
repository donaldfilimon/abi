# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Build & Validation
- `./build.sh check` – Primary validation gate for build integrity and API parity checks.
- `./build.sh full-check` – Runs all checks including integration tests and benchmark suite.
- `./build.sh cli` – Builds the main executable (`abi`).
- `./build.sh mcp` – Builds the MCP server binary.
- `zig build lint` – Runs `zig fmt --check` on all source files for formatting compliance.
- `zig build fix` – Automatically formats source files based on project standards.
- `zig build check-parity` – Verifies public API parity between mod/stub and feature implementations.

### Running Tests
- `zig build test-integration` – Executes the integration test suite (`src/integration_tests.zig`).
- `zig build benchmarks` – Runs the benchmark suite (`src/benchmarks.zig`).

## Architecture Overview

The ABI framework is a modular Zig codebase with a clear separation of concerns across the following layers:

| Layer | Path | Responsibility |
|-------|------|----------------|
| **Public API** | `src/root.zig` | Exposes the `abi` module to consumers. This is the primary entry point. |
| **CLI** | `src/main.zig`, `src/abi_cli/` | Parses command-line arguments and delegates to sub-commands defined in `src/abi_cli/usage.zig`. |
| **MCP Server** | `src/mcp/main.zig` | Implements a JSON-RPC 2.0 server over stdio and optional HTTP/SSE transport. HTTP defaults to `127.0.0.1:8080` and can be moved with `ABI_MCP_HTTP_PORT`. |
| **Feature Selection** | `src/features/mod.zig` | Enables/disables features via Zig build options (`-Dfeat-*`). Uses the *mod/stub* pattern to keep public APIs stable. This layer defines the major components: GPU, AI, Vector Store (WDBX), etc. |
| **AI Sub-system** | `src/features/ai/` | Implements AI profiles (Abbey, Aviva, Abi), routing (`router.zig`), and a six-principle constitution (`constitution.zig`). |
| **Vector Store (WDBX)** | `src/features/wdbx/` | Provides in-memory key-value and vector storage with HNSW index (`hnsw.zig`) and MVCC-style snapshot chain (`chain.zig`). |
| **GPU Backend** | `src/features/gpu/` | Reports GPU status, attempts Metal initialization on macOS, and falls back to a vectorized CPU implementation. Includes components like `metal_shared.zig` and reporting. |
| **Connectors** | `src/connectors/` | Provides local/live adapters for external services (OpenAI, Anthropic, Discord, Twilio, HTTP, JSON). |
| **Plugin System** | `src/plugins/`, `src/plugin_registry.zig` | Validates required plugin manifests (`name`, `version`, `description`, `target_feature`, `entry_point`) and generates a metadata registry via `tools/generate_plugin_registry.zig`. |
| **Scheduler & Memory** | `src/core/scheduler.zig`, `src/core/memory.zig` | Handles task scheduling, counters, and custom memory allocator tracking for the system. Includes core utilities like time, sync, logging, OS abstractions, and credential management. |

### Key Areas to Focus On
- **Mod/Stub Pattern**: Ensure public API stability by checking mod/stub parity frequently.
- **Build Flow**: Understand that `./build.sh full-check` validates the entire system's correctness across layers.
- **Layer Interaction**: Pay close attention to how features enable/disable functionality via `mod.zig`.
- **Core Utilities**: The `foundation` layer handles OS abstractions, IO, logging, and synchronization primitives.
- **Generated Code**: Do not manually edit `src/plugin_registry.zig`; update plugin manifests or generator code and rerun the build.
- **Connector Boundaries**: Discord local/live paths validate credentials, numeric snowflake IDs, and message size before dispatch.