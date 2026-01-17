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
zig build test --summary all      # Run all tests with detailed summary
zig build run -- --help           # CLI help
zig fmt .                         # Format all code
zig fmt --check .                 # Check formatting without changes
```

### Testing

```bash
# Run all tests
zig build test

# Run tests with specific features enabled
zig build test -Denable-gpu=true -Denable-network=true

# Test a single file
zig test src/compute/runtime/engine.zig

# Run tests matching a pattern
zig test --test-filter="engine init"

# Run tests with verbose output
zig build test --summary all
```

### Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
```

Flags: `-Denable-ai`, `-Denable-gpu`, `-Denable-web`, `-Denable-database`, `-Denable-network`, `-Denable-profiling`, `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`

## Code Style Guidelines

### Formatting

- 4 spaces, no tabs
- Maximum 100 characters per line
- Run `zig fmt .` before committing

### Naming Conventions

| Construct | Convention | Examples |
|-----------|------------|----------|
| Types | `PascalCase` | `Framework`, `GpuBuffer`, `NetworkConfig` |
| Functions | `snake_case` | `vector_add`, `deinit`, `createFramework` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_BUFFER_SIZE`, `MAX_THREADS` |
| Variables | `snake_case` | `gpu_workload`, `result_buffer` |
| Struct Fields | `snake_case` with `allocator` first | `allocator`, `user_data`, `stream` |

### Imports

- Use explicit imports only (no `usingnamespace`)
- Group imports: std, internal modules, public exports

```zig
const std = @import("std");
const build_options = @import("build_options");

const GpuBuffer = @import("gpu/buffer.zig").Buffer;
const kernel = @import("kernel.zig");
```

### Error Handling

- Use specific error sets, not `anyerror`
- Use `{t}` format specifier for error/enum values
- Document error sets with comments

```zig
const MyError = error{
    OutOfMemory,
    InvalidInput,
    GpuDisabled,
};

std.debug.print("Error: {t}\n", .{err});
```

### Memory Management

- Prefer `std.ArrayListUnmanaged` over `std.ArrayList`
- Always use `defer` and `errdefer` for cleanup
- Pass allocator explicitly to unmanaged containers

```zig
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
defer list.deinit(allocator);
```

### Zig 0.16 Patterns

| Pattern | Usage |
|---------|-------|
| File I/O | Use `std.Io.Threaded.init(allocator, .{})` |
| Sleep | Use `std.Io.Clock.Duration.sleep()` or `time.sleepMs()` |
| Formatting | Use `{t}` for enums/errors, `{B}` for bytes, `{D}` for durations |
| Timer | `std.time.Timer.start() catch return error.TimerFailed` |
| Pointer casting | Use `@ptrCast(@alignCast(user))` |

### Resource Cleanup

Always clean up resources with `defer`/`errdefer`:

```zig
var gpu = try Gpu.init(allocator, .{});
defer gpu.deinit();

const buffer = try gpu.createBuffer(size, .{});
errdefer gpu.destroyBuffer(buffer);
```

### Key Patterns

1. **Initialization**: Use `abi.init()` / `abi.shutdown()`
2. **Allocators**: `allocator` as first struct field
3. **Cleanup**: `defer`/`errdefer` for all resources
4. **I/O**: `std.Io.Threaded` for file operations (Zig 0.16)
5. **Threading**: Use `std.Thread.Mutex` for synchronization

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Comprehensive agent guidance |
| [docs/intro.md](docs/intro.md) | Architecture overview |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API documentation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |
| [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) | Zig 0.16 patterns |
| [src/shared/contacts.zig](src/shared/contacts.zig) | Centralized maintainer contacts |
| [docs/ai.md](docs/ai.md) | AI module guide |
| [TODO.md](TODO.md) | Pending implementations |
| [ROADMAP.md](ROADMAP.md) | Project roadmap |

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files.

## Pending Work

Agents should be aware of the items in **[TODO.md](TODO.md)**. When generating code or suggestions, avoid relying on stubbed functionality (e.g., format converters for GGUF/NPZ) until the corresponding TODO is resolved.
