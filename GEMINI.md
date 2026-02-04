# GEMINI.md

This file provides guidance to Google Gemini when working with code in this repository.

## Zig Version Requirement

**Required:** Zig `0.16.0-dev.2471+e9eadee00` or later (master branch)

The codebase uses Zig 0.16 APIs. Earlier versions will fail to compile.

## Before Making Changes

1. Run `git status` to see uncommitted work
2. Run `git diff --stat` to understand the scope of existing changes
3. Review existing changes before adding new ones to avoid conflicts

## Quick Reference

```bash
# Build and test
zig build                              # Build the project
zig build test --summary all           # Run tests with detailed output
zig fmt .                              # Format code (run after edits)
zig build lint                         # Check formatting (CI uses this)

# Single file testing
zig test src/path/to/file.zig --test-filter "pattern"

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
zig build -Dgpu-backend=vulkan         # GPU backends: auto, cuda, vulkan, metal, etc.
```

## Architecture

Flat domain structure with unified configuration. Each domain has `mod.zig` (entry point) and `stub.zig` (feature-gated placeholder).

```
src/
├── abi.zig              # Public API entry point
├── config/              # Unified configuration system
├── ai/                  # AI module (agents, llm, streaming, training)
├── database/            # Vector database (WDBX with HNSW/IVF-PQ)
├── gpu/                 # GPU acceleration (Vulkan, CUDA, Metal, etc.)
├── network/             # Distributed compute and Raft consensus
├── runtime/             # Task execution, scheduling, concurrency
├── shared/              # Utilities, security, SIMD
└── tests/               # Test suite (chaos, e2e, integration, stress)
```

**Import guidance:**
- Always use `@import("abi")` for public API - never import files directly
- When modifying a module's API, update both `mod.zig` and `stub.zig`

## Zig 0.16 API Patterns (Required)

```zig
// I/O Backend - required for file/network operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

// File system - use std.Io.Dir, NOT std.fs
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));

// Timing - use Timer.start(), NOT Instant.now()
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// ArrayListUnmanaged - use .empty, NOT .init()
var list = std.ArrayListUnmanaged(u8).empty;

// Format specifiers - use {t} for enums/errors
std.debug.print("Error: {t}, State: {t}", .{err, state});

// Reserved keywords - escape with @"" syntax
const err = result.@"error";
```

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` syntax | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| File system operations | Use `std.Io.Dir.cwd()` not deprecated `std.fs.cwd()` |
| Reserved keywords | Escape with `@"error"` syntax |
| Stub/Real module sync | Changes to `mod.zig` must be mirrored in `stub.zig` |
| ArrayListUnmanaged | Use `.empty` not `.init()`; pass allocator to ops |
| Timer API | Use `std.time.Timer.start()` not `std.time.Instant.now()` |

## Code Style

| Rule | Convention |
|------|------------|
| Indentation | 4 spaces, no tabs |
| Line length | Under 100 characters |
| Types | `PascalCase` |
| Functions/Variables | `camelCase` |
| Error handling | `!` return types, specific error enums |
| Cleanup | Prefer `defer`/`errdefer` |

## Post-Edit Checklist

```bash
zig fmt .                        # Format code
zig build test --summary all     # Run all tests
zig build lint                   # Verify formatting passes CI
```

## References

- `CLAUDE.md` - Detailed Claude Code guidance
- `AGENTS.md` - General agent guidelines
- `PLAN.md` - Development roadmap
- `SECURITY.md` - Security practices
