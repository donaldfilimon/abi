# AGENTS.md

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
```

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

### Architecture

The codebase uses a domain-driven flat structure with unified configuration:

```
src/
├── abi.zig              # Public API entry point
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration
├── ai/                  # AI module (core/, implementation/, sub-features/)
├── connectors/          # External API connectors (OpenAI, Ollama, etc.)
├── database/            # Vector database (WDBX)
├── gpu/                 # GPU acceleration (Vulkan, CUDA, Metal, etc.)
├── ha/                  # High Availability (Replication, Backup, Failover)
├── network/             # Distributed compute and Raft
├── observability/       # Metrics, tracing, monitoring
├── registry/            # Plugin registry system
├── runtime/             # Always-on infrastructure (Task engine)
├── shared/              # Consolidated utilities and platform helpers
├── tasks.zig            # Consolidated task management system
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
- Use specific error types (never `anyerror`).
- Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()`.

### DON'T
- Create new directories for small files; prefer consolidation in parent domain.
- break the `mod.zig`/`stub.zig` parity.
- break public API stability in `abi.zig`.

## File Organization

| Domain | Location | Purpose |
|--------|----------|---------|
| **Public API** | `src/abi.zig` | Entry point for framework users |
| **AI** | `src/ai/` | API in `src/ai/`, implementation in `src/ai/implementation/` |
| **GPU** | `src/gpu/` | Unified API and hardware backends |
| **Database** | `src/database/` | WDBX vector database |
| **Shared** | `src/shared/` | Consolidated utils, logging, platform detection |
| **Tasks** | `src/tasks.zig` | Integrated task management system |
| **HA** | `src/ha/` | High Availability and replication |
| **Connectors** | `src/connectors/` | External AI model providers |

## Common Tasks

### Adding a New Feature
1. Check feature flag in `src/config.zig`.
2. Create `mod.zig` and `stub.zig` in the new domain directory.
3. Wire into `src/framework.zig` and `src/abi.zig`.

### Consolidation Rule
If a module grows beyond a single file, keep it in a directory with `mod.zig`. If it contains only a few small helpers, consolidate into the domain's primary file or `src/shared/utils.zig`.

## Testing & Debugging
```bash
zig build test --summary all
zig test src/runtime/engine/engine.zig
```
Use the `{t}` format specifier for errors and enums in Zig 0.16.

## Need Help?
- Read `CLAUDE.md` for detailed engineering guidelines.
- Run `zig build run -- --help` for CLI command details.