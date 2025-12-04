# Repository Guidelines

## 1. Project layout

- **lib/** – Core library modules (main source code)
- **tests/** – Test suites (unit/, integration/, benchmarks/)
- **examples/** – Runnable demos and examples
- **tools/** – Development tools and CLI utilities
- **benchmarks/** – Performance benchmarks

Main library entry point: `lib/mod.zig` imported as `abi`

## 2. Build & test commands

| Command | Purpose |
|---------|---------|
| `zig build` | Build library and CLI |
| `zig build test` | Run unit tests |
| `zig build test-integration` | Run integration tests |
| `zig build test-all` | Run all tests |
| `zig test <file>` | Run single test file |
| `zig build examples` | Build all examples |
| `zig build run-<example>` | Run specific example |
| `zig fmt -w .` | Format all code |

## 3. Code style

* **Indent**: 4 spaces, no tabs
* **Types**: `PascalCase` 
* **Functions/vars**: `snake_case`
* **Constants**: `ALL_CAPS`
* **Files**: lowercase, use underscores
* **Imports**: Use `@import("std")` and module aliases
* **Error handling**: Use `!T` return types, propagate with `try`
* **Documentation**: Use `//!` for module docs, `///` for public APIs

## 4. Testing patterns

Test files end with `_test.zig`. Use `testing.allocator` for allocators. Structure tests with `try` for error handling and `defer` for cleanup. Import main library as `const abi = @import("abi");`

## 5. Build configuration

Feature flags controlled via build options: `enable-ai`, `enable-gpu`, `enable-database`, `enable-web`, `enable-monitoring`. Use build.zig for conditional compilation.

