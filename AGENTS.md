# AI Agent Guidance

This file provides guidance to AI agents (GitHub Copilot, Cursor, Windsurf, etc.) when working with code in this repository.

> **Note**: For comprehensive guidance, see [CLAUDE.md](CLAUDE.md) which contains the full documentation.

## LLM Instructions (Shared)

- Keep changes minimal and consistent with existing patterns; avoid breaking public APIs unless requested.
- Preserve feature gating: stub modules must mirror the real API and return `error.*Disabled`.
- Use Zig 0.16 conventions (`std.Io`, `std.ArrayListUnmanaged`, `{t}` formatting, explicit allocators).
- Always clean up resources with `defer`/`errdefer`; use specific error sets (no `anyerror`).
- Run `zig fmt .` after code edits and `zig build test --summary all` when behavior changes.
- Update docs/examples when APIs or behavior change so references stay in sync.

## Quick Reference

### Project Overview

ABI is a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling.

### Essential Commands

```bash
zig build                         # Build the project
zig build test --summary all      # Run all tests
zig build run -- --help           # CLI help
zig fmt .                         # Format code
```

### Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
```

Flags: `-Denable-ai`, `-Denable-gpu`, `-Denable-web`, `-Denable-database`, `-Denable-network`, `-Denable-profiling`

### Key Patterns

1. **Initialization**: Use `abi.init()` / `abi.shutdown()`
2. **Allocators**: Prefer `std.ArrayListUnmanaged` over `std.ArrayList`
3. **Cleanup**: Always use `defer`/`errdefer` for resource cleanup
4. **Errors**: Use specific error sets, not `anyerror`
5. **I/O**: Use `std.Io.Threaded` for file operations (Zig 0.16)

### Zig 0.16 Notes

```zig
// Format specifiers
std.debug.print("{t}\n", .{status});  // {t} for enums/errors

// Timing
var timer = std.time.Timer.start() catch return error.TimerFailed;

// File I/O - no std.fs.cwd() in Zig 0.16
var io_backend = std.Io.Threaded.init(allocator, .{});
defer io_backend.deinit();
```

### Sleep API

Use `std.Io`-based sleep instead of `std.time.sleep()`:

```zig
// Preferred - use shared/utils/time.zig helpers
const time = @import("shared/utils/time.zig");
time.sleepMs(100);  // Sleep 100 milliseconds

// Or directly with Io context
const duration = std.Io.Clock.Duration{
    .clock = .awake,
    .raw = .fromNanoseconds(nanoseconds),
};
std.Io.Clock.Duration.sleep(duration, io) catch {};
```

### Coding Style

- 4 spaces, max 100 chars/line
- Types: `PascalCase`, Functions: `snake_case`, Constants: `UPPER_SNAKE_CASE`
- Struct fields: `allocator` first

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Comprehensive agent guidance |
| [docs/intro.md](docs/intro.md) | Architecture overview |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API documentation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |
| [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) | Zig 0.16 patterns |
| [src/shared/contacts.zig](src/shared/contacts.zig) | Centralized maintainer contacts |
| [docs/ai.md](docs/ai.md) | AI module guide – includes training pipeline and CLI usage |
| [TODO.md](TODO.md) | List of pending implementations and placeholders |
| [ROADMAP.md](ROADMAP.md) | Project roadmap and milestone planning |

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
The roadmap now includes a **Llama‑CPP parity** section (Version 0.6.0 – Q4 2026) detailing the tasks required to achieve feature parity with llama‑cpp.

## Pending Work

Agents should be aware of the items in **[TODO.md](TODO.md)**. When generating code or suggestions, avoid relying on stubbed functionality (e.g., format converters for GGUF/NPZ) until the corresponding TODO is resolved.
See [TODO.md](TODO.md) for the list of pending implementations.
The TODO list now includes a **Llama‑CPP parity task table** describing required modules to achieve llama‑cpp feature parity.
*See [TODO.md](TODO.md) and [ROADMAP.md](ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
