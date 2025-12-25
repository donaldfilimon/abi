# ABI Framework – Agent Guidelines

## Build/Test Commands

- `zig build` - Build library + CLI
- `zig build test` - Run all tests
- `zig test tests/mod.zig` - Run smoke tests directly
- `zig test src/compute/runtime/engine.zig` - Run single file tests
- `zig test --test-filter="pattern"` - Run matching tests
- `zig build run` - Run CLI
- `zig build run -- --help` - Run CLI help
- `zig fmt .` - Format code | `zig fmt --check .` - Check formatting

## Code Style Guidelines

- **Formatting**: 4 spaces, 100 char lines, `zig fmt` required
- **Naming**: PascalCase for types, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Imports**: Group std imports first, then internal. No `usingnamespace`. Prefer qualified access.
- **Error Handling**: Use `!` return types, specific enums over generic, `errdefer` for cleanup, `try` for propagation
- **Memory**:
  - Use stable allocator (GPA) for long-lived data.
  - Use worker arenas for per-thread scratch.
  - Results from stable allocator, NOT worker arena.
  - Reset arenas, never destroy mid-session.
- **Documentation**: Module-level docs with `//!`, function docs with `///`, examples in ```zig blocks
- **Testing**: `test` blocks at file end, use `testing.allocator`, co-locate `*_test.zig`, use `mod.zig` for re-exports.

## Zig 0.16 Specifics

- Use `cmpxchgStrong` / `cmpxchgWeak` (returns `?T`).
- Use `std.atomic.spinLoopHint()` instead of `spinLoop`.
- Use `std.Thread.spawn(.{}, ...)` (options struct first).

## Development Workflow

1. Format with `zig fmt .`
2. Build and test with `zig build test`
3. Run specific tests as needed
4. Update documentation for public APIs
5. Follow security best practices (no secrets in code, validate inputs)
6. Feature flags: `-Denable-gpu` (true), `-Denable-network` (false), `-Denable-profiling` (false), `-Denable-ai/web/database` (true)
