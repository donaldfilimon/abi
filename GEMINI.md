# Google Gemini Guidance

This file provides guidance to Google Gemini when working with code in this repository.

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
zig build benchmarks              # Run benchmarks
```

### Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI features and connectors |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-web` | true | Web utilities and HTTP |
| `-Denable-database` | true | Vector database |
| `-Denable-network` | true | Distributed compute |
| `-Denable-profiling` | true | Profiling and metrics |

### Project Structure

```
src/
├── abi.zig          # Public API entry point
├── compute/         # Compute engine, concurrency, GPU
├── features/        # AI, database, GPU, monitoring, network
├── framework/       # Lifecycle and orchestration
└── shared/          # Cross-cutting utilities
```

### Key Patterns

1. **Initialization**: `abi.init()` / `abi.shutdown()`
2. **Allocators**: Prefer `std.ArrayListUnmanaged`
3. **Cleanup**: Always use `defer`/`errdefer`
4. **Errors**: Use specific error sets, not `anyerror`
5. **Feature Gating**: Stubs return `error.*Disabled` when disabled

### Zig 0.16 Conventions

```zig
// Format specifiers
std.debug.print("{t}\n", .{status});  // {t} for enums/errors
std.debug.print("{B}\n", .{size});    // {B} for byte sizes

// Timing
var timer = std.time.Timer.start() catch return error.TimerFailed;

// File I/O (no std.fs.cwd() in Zig 0.16)
var io_backend = std.Io.Threaded.init(allocator, .{});
defer io_backend.deinit();
const io = io_backend.io();
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
- Struct fields: `allocator` first, then config/state

### Environment Variables

- `ABI_OPENAI_API_KEY` - OpenAI API key
- `ABI_HF_API_TOKEN` - HuggingFace token
- `ABI_OLLAMA_HOST` - Ollama URL (default: `http://127.0.0.1:11434`)

## Documentation

## Contacts

`src/shared/contacts.zig` provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Comprehensive agent guidance |
| [docs/intro.md](docs/intro.md) | Architecture overview |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API documentation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |
| [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) | Zig 0.16 patterns |
| [src/shared/contacts.zig](src/shared/contacts.zig) | Centralized maintainer contacts |
