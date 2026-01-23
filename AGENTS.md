# AGENTS.md

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

> **Codebase Status:** Synced with repository as of 2026-01-23.

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
├── config/              # Domain-specific configs (gpu, ai, database, etc.)
├── ai/                  # AI module (core, implementation, gpu_interface)
├── connectors/          # External API connectors
├── database/            # Vector database (WDBX)
├── gpu/                 # GPU acceleration (backends)
├── ha/                  # High Availability
├── network/             # Distributed compute and Raft
├── observability/       # Metrics, tracing, monitoring
├── registry/            # Plugin registry (lifecycle, plugins)
├── runtime/             # Task engine
├── shared/              # Utils and helpers
├── tasks/               # Task management (persistence, querying, lifecycle)
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

## Build, Lint, and Test Commands

- **Building the Project**: Use `zig build` to compile the entire project. For optimized builds, add `-Doptimize=ReleaseFast`.
- **Linting**: Run `zig build lint` to check for code quality issues and style violations.
- **Type Checking**: Run `zig build typecheck` (if available) to verify type correctness without full compilation.
- **Running All Tests**: Use `zig build test --summary all` to execute all tests and see a summary of results.
- **Running a Single Test**: To run tests in a specific file, use `zig test <path/to/test.zig>`. For example, `zig test src/runtime/engine/engine.zig`. If the build system supports filtering, use `zig build test --filter <test_name>` to run a specific test function.

## Code Style Guidelines

### Imports
- Use `@import("std")` for standard library imports.
- Use relative imports like `@import("../mod.zig")` for internal modules within the same domain.
- Avoid wildcard imports; import specific modules or functions.
- Group imports by type: std first, then external, then internal.

### Formatting
- Always run `zig fmt .` after making changes to ensure consistent formatting.
- Follow Zig's standard indentation (4 spaces, no tabs).
- Use line breaks appropriately; keep lines under 100 characters where possible.
- Align struct fields and function parameters for readability.

### Types
- Prefer specific types over generic ones (e.g., `u32` over `usize` when size is known).
- Use enums for options or states instead of magic numbers or strings.
- Avoid `anytype` in public APIs; use generics with constraints if needed.
- Use `comptime` for compile-time constants and checks.

### Naming Conventions
- **Functions and Variables**: Use `snake_case` (e.g., `get_user_name`).
- **Types (Structs, Enums, Unions)**: Use `PascalCase` (e.g., `UserConfig`).
- **Constants**: Use `UPPER_CASE` (e.g., `MAX_SIZE`).
- **Modules/Files**: Use `snake_case` or `kebab-case` for file names (e.g., `gpu_interface.zig`).
- Avoid abbreviations unless widely accepted (e.g., `idx` for index, but prefer `index`).

### Error Handling
- Use specific error types instead of `anyerror`; define custom error sets when needed.
- Use `try` for propagating errors up the call stack.
- Employ `errdefer` for cleanup on error paths.
- Handle errors explicitly; avoid ignoring them with `_`.

### Memory Management
- Use `std.ArrayListUnmanaged` for dynamic arrays to avoid implicit allocations.
- Always use explicit allocators; avoid global allocators unless necessary.
- Use `defer` and `errdefer` for resource cleanup.
- Prefer stack allocation for small, short-lived data; use heap only when needed.

### Other Best Practices
- Write self-documenting code; use comments sparingly for complex logic.
- Follow the domain-driven structure; keep related code together.
- Maintain mod.zig/stub.zig parity for feature-gated modules.
- Use compile-time feature flags for optional dependencies.

## Common Workflows

### Adding a New Public API Function
1. Add to the real module (`src/<feature>/mod.zig`)
2. Mirror the same signature in `src/<feature>/stub.zig`
3. If the function needs types from core, import from `../../core/mod.zig`
4. Run `zig build test` to verify both paths compile

### Consolidation Rule
If a module grows beyond a single file, keep it in a directory with `mod.zig`. If it contains only a few small helpers, consolidate into the domain's primary file or `src/shared/utils.zig`.

## Testing & Debugging
```bash
zig build test --summary all
zig test src/runtime/engine/engine.zig
```
Use the `{t}` format specifier for errors and enums in Zig 0.16.

## Important Gotchas
* **Stub‑Real Sync** – Mutating `src/<feature>/mod.zig` requires the companion `src/<feature>/stub.zig` to stay API‑compatible. The CI parity checker on every push will flag mismatches.
* **GPU Backend List** – `-Dgpu-backend` accepts a comma‑separated list. For a single backend, supply only one identifier such as `-Dgpu-backend=cuda`.
* **Public API Import** – All public APIs must be accessed via `@import("abi")`. Direct relative imports break the public contract and will cause link errors when size‑optimised builds are performed.

★ *Tip:* Run `zig fmt .` after any edit and immediately verify with `zig build test --summary all`.
