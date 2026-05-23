# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- **Build & validation**
  - `./build.sh check` – Primary validation gate (builds CLI & MCP, runs module/connector tests, format check, mod/stub parity)
  - `./build.sh full-check` – `check` plus integration tests and benchmark suite.
  - `./build.sh cli` – Build the `abi` executable (`zig-out/bin/abi`).
  - `./build.sh mcp` – Build the `abi-mcp` server binary.
  - `zig build lint` – Run `zig fmt --check` on all source files.
  - `zig build fix` – Auto‑format source files.
  - `zig build check-parity` – Verify public API parity between `mod.zig` and `stub.zig`.

## Running tests
- `zig build test-integration` – Execute the integration test suite (`src/integration_tests.zig`).
- `zig build benchmarks` – Run the benchmark suite (`src/benchmarks.zig`).

## Architecture Overview

The ABI framework is a modular Zig codebase with a clear separation of concerns. The most important entry points and layers are:

| Layer | Path | Responsibility |
|-------|------|----------------|
| **Public API** | `src/root.zig` | Exposes the `abi` module to consumers. |
| **CLI** | `src/main.zig` & `src/abi_cli/` | Parses command‑line arguments, delegates to sub‑commands defined in `src/abi_cli/usage.zig`. |
| **MCP Server** | `src/mcp/main.zig` | Implements a JSON‑RPC 2.0 server over stdio and optional HTTP/SSE transport. |
| **Feature Selection** | `src/features/mod.zig` | Enables/disabled features via Zig build options (`-Dfeat‑*`). Uses the *mod/stub* pattern to keep public APIs stable. |
| **AI Sub‑system** | `src/features/ai/` | Implements AI profiles (Abbey, Aviva, Abi), routing (`router.zig`), and a six‑principle constitution (`constitution.zig`). |
| **Vector Store (WDBX)** | `src/features/wdbx/` | In‑memory key‑value and vector storage with HNSW index (`hnsw.zig`) and MVCC‑style snapshot chain (`chain.zig`). |
| **GPU Backend** | `src/features/gpu/mod.zig` | Reports GPU status, attempts Metal initialization on macOS, and falls back to vectorised CPU implementation. |
| **Connectors** | `src/connectors/mod.zig` | Provides local/live adapters for OpenAI, Anthropic, Discord, and Twilio. |
| **Foundations** | `src/foundation/` | Core utilities (time, sync, logging, errors, OS abstractions, credentials). |
| **Plugin System** | `src/plugins/` & `src/plugin_registry.zig` | Manages plugin manifests (`abi-plugin.json`) and generates a registry via `tools/generate_plugin_registry.zig`. |
| **Scheduler & Memory** | `src/core/scheduler.zig`, `src/core/memory.zig` | Task scheduling, counters, and custom memory allocator tracking.
