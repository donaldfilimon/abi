# AGENTS.md

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

> **Codebase Status:** Synced with repository as of 2026-01-23.

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

## Quick Start for AI Agents

```bash
# Essential commands
zig build                              # Build the project
zig build test --summary all           # Run all tests
zig fmt .                              # Format code after edits
zig build run -- --help                # CLI help

# Example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example

# LLM commands
zig build run -- llm chat --model llama-7b   # Interactive chat
zig build run -- llm generate "Once" --max 50 # Text generation
zig build run -- llm list                    # List available models
```

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

### Architecture

The codebase uses a domain-driven modular structure with unified configuration and compile-time feature gating:

```
src/
├── abi.zig              # Public API entry point
├── framework.zig        # Framework orchestration
├── config/              # Domain-specific configuration modules
│   ├── gpu.zig          # GPU configuration (gated by build_options.enable_gpu)
│   ├── ai.zig           # AI configuration (gated by build_options.enable_ai)
│   ├── database.zig     # Database configuration (gated by build_options.enable_database)
│   ├── network.zig      # Network configuration (gated by build_options.enable_network)
│   ├── observability.zig # Observability configuration (gated by build_options.enable_profiling)
│   ├── web.zig          # Web configuration (gated by build_options.enable_web)
│   └── mod.zig          # Unified config shim (backward compatibility)
├── ai/                  # AI module (core/, implementation/, sub-features/)
│   └── gpu_interface.zig # Lightweight GPU acceleration interface (optional)
├── connectors/          # External API connectors (OpenAI, Ollama, etc.)
├── database/            # Vector database (WDBX)
├── gpu/                 # GPU acceleration (Vulkan, CUDA, Metal, etc.)
├── ha/                  # High Availability (Replication, Backup, Failover)
├── network/             # Distributed compute and Raft
├── observability/       # Metrics, tracing, monitoring
├── registry/            # Plugin registry system
│   ├── lifecycle.zig    # Feature lifecycle management (init/deinit, modes)
│   ├── plugins/         # Plugin-specific code and discovery
│   └── mod.zig          # Registry orchestration facade
├── runtime/             # Always-on infrastructure (Task engine)
├── shared/              # Consolidated utilities and platform helpers
│   └── gpu_ai_utils.zig # Shared GPU/AI utilities
├── tasks/               # Task management system (functionality split)
│   ├── persistence.zig  # Save/load logic (JSON, file I/O)
│   ├── querying.zig     # List/filter/sort operations
│   ├── lifecycle.zig    # Add/update/delete/complete/start operations
│   └── mod.zig          # Unified task facade
└── web/                 # Web/HTTP utilities
```

### Key Patterns

1. **Framework Initialization**:
   ```zig
   const abi = @import("abi");

   // Default init
   var fw = try abi.init(allocator);
   defer fw.deinit();

   // Access features
   if (fw.isEnabled(.gpu)) {
       const gpu = try fw.getGpu();
   }
   ```

2. **Configuration**: Use `Config` struct (single source of truth).
   ```zig
   const config = abi.Config{
       .gpu = .{ .backend = .vulkan },
       .ai = .{ .llm = .{} },
   };
   ```

3. **Feature Gating**: Compile-time flags with stub modules.
   ```zig
   const impl = if (build_options.enable_ai) @import("mod.zig") else @import("stub.zig");
   ```

4. **Module Convention**: `mod.zig` (entry), `stub.zig` (disabled placeholder).

5. **Memory**: Use `std.ArrayListUnmanaged`, `defer`/`errdefer`, explicit allocators.

6. **Domain Splits**: Config and tasks split into domain-specific files for modularity (e.g., config/gpu.zig, tasks/persistence.zig).

7. **Stub Parity Automation**: Use build scripts or comptime generators to maintain mod.zig/stub.zig API parity.

8. **GPU/AI Interface**: Decouple with ai/gpu_interface.zig for optional GPU acceleration, gated by build_options.enable_gpu.

## Critical Rules

### DO
- Read relevant files before editing.
- Run `zig fmt .` after code changes.
- Run `zig build test --summary all` to verify changes.
- Run `zig build lint` and `zig build typecheck` (if available) to ensure code quality.
- Use specific error types (never `anyerror`).
- Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()`.
- Automate stub parity with build scripts to avoid manual drift.

### DON'T
- Create new directories for small files; prefer consolidation in parent domain.
- break the `mod.zig`/`stub.zig` parity.
- break public API stability in `abi.zig`.
- introduce code that exposes or logs secrets/keys.
- commit secrets or keys to the repository.

## File Organization

| Domain | Location | Purpose |
|--------|----------|---------|
| **Public API** | `src/abi.zig` | Entry point for framework users |
| **Config** | `src/config/` | Domain-specific config modules (gpu.zig, ai.zig, etc.) |
| **AI** | `src/ai/` | API in `src/ai/`, implementation in `src/ai/implementation/`, gpu_interface.zig for optional GPU |
| **LLM** | `src/ai/llm/` | Local GGUF inference (Q4/Q5/Q8, BPE/SentencePiece, CUDA) |
| **GPU** | `src/gpu/` | Unified API and hardware backends |
| **Database** | `src/database/` | WDBX vector database |
| **Shared** | `src/shared/` | Consolidated utils, logging, platform detection, gpu_ai_utils.zig |
| **Tasks** | `src/tasks/` | Split functionality: persistence.zig, querying.zig, lifecycle.zig, mod.zig facade |
| **Registry** | `src/registry/` | lifecycle.zig for management, plugins/ for discovery, mod.zig facade |
| **HA** | `src/ha/` | High Availability and replication |
| **Connectors** | `src/connectors/` | External AI model providers |

## Common Workflows

### Adding a New Public API Function
1. Add to the real module (`src/<feature>/mod.zig`)
2. Mirror the same signature in `src/<feature>/stub.zig`
3. If the function needs types from core, import from `../../core/mod.zig`
4. Run `zig build test` to verify both paths compile

### Adding a New Feature Module
1. Check feature flag in `src/config/`
2. Create `mod.zig` and `stub.zig` in the new domain directory
3. Wire into `src/framework.zig` and `src/abi.zig`

### Adding a New Example
1. Add entry to `example_targets` array in `build.zig`
2. Create the example file in `examples/`
3. Run `zig build examples` to verify compilation

### Consolidation Rule
If a module grows beyond a single file, keep it in a directory with `mod.zig`. If it contains only a few small helpers, consolidate into the domain's primary file or `src/shared/utils.zig`.

## Stub Parity Automation
- Use build scripts to generate stub.zig files from mod.zig signatures to maintain API parity.
- Run parity checks during CI to prevent drift between enabled and disabled feature paths.

## Testing & Debugging
```bash
zig build test --summary all
zig test src/runtime/engine/engine.zig
```
Use the `{t}` format specifier for errors and enums in Zig 0.16.

## Refactor Phases Overview
The codebase underwent a modular refactor in phases to improve maintainability and feature gating. Key changes include domain splits in config/ and tasks/, registry lifecycle/plugins separation, AI/GPU decoupling with interfaces, and automated stub parity. Refer to the JSON todo list for detailed phase breakdowns.

## Need Help?
- Read `CLAUDE.md` for detailed engineering guidelines.
- Run `zig build run -- --help` for CLI command details.

## New Build Examples
The example binaries use a `run-` prefix and forward all arguments. Typical usage:

```bash
zig build run-hello          # Hello world demo
zig build run-database       # Vector database example
zig build run-agent          # Autonomous agent example
zig build run-llm            # Local LLM inference demo
zig build run-gpu            # GPU compute example
```

To add a new example, insert a `BuildTarget{ .cli = "new", .source = "examples/new.zig" }` into the `example_targets` array in `build.zig` and create the file in `examples/`.

## Important Gotchas
* **Stub‑Real Sync** – Mutating `src/<feature>/mod.zig` requires the companion `src/<feature>/stub.zig` to stay API‑compatible. The CI parity checker on every push will flag mismatches.
* **GPU Backend List** – `-Dgpu-backend` accepts a comma‑separated list. For a single backend, supply only one identifier such as `-Dgpu-backend=cuda`.
* **WebAssembly Builds** – When `zig build wasm` or `-Dtarget=wasm32` is used, the `database`, `network`, and `gpu` features are automatically disabled; explicitly disabling them makes the build deterministic.
* **Public API Import** – All public APIs must be accessed via `@import("abi")`. Direct relative imports break the public contract and will cause link errors when size‑optimised builds are performed.
* **Compile‑time vs Runtime Flags** – Feature toggles prefixed with `-D` affect compilation. Runtime flags such as `--enable-gpu` only modify a running binary's behaviour.

★ *Tip:* Run `zig fmt .` after any edit and immediately verify with `zig build test --summary all`.
