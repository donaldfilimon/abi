# AI Agent Guidance

This file provides guidance to AI agents (GitHub Copilot, Cursor, Windsurf, etc.) when working with code in this repository.

> **Note**: For comprehensive guidance, see [CLAUDE.md](CLAUDE.md) which contains the full documentation.

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
