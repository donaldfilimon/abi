# ABI Framework – Agent Guidelines

## Build/Test Commands
```bash
zig build                    # Build library + CLI
zig build test               # Run all tests
zig test <file>              # Run single test file (e.g., zig test tests/unit/test_build.zig)
zig fmt --check .            # Check code formatting
zig build run                # Run CLI
zig build -Denable-gpu=true -Denable-ai=true  # Build with features
zig build check              # Aggregate validation (format + build + analysis)
zig build docs               # Generate documentation
```

## Code Style Guidelines
- **Formatting**: 4 spaces, no tabs; lines under 100 characters
- **Naming**: PascalCase (types/structs), snake_case (functions/variables), UPPER_SNAKE_CASE (constants)
- **Documentation**: `//!` for modules, `///` for public API with examples and error conditions
- **Imports**: Explicit imports only, avoid `usingnamespace`; group std imports first
- **Error Handling**: Use `!` return types, `try`/`catch`, custom error enums; provide rich error context
- **Memory**: `defer` for cleanup, explicit allocators; use arena allocators for temporaries
- **Zig 0.16**: `std.debug.print`, `vtable.*` allocators, async I/O patterns
- **Testing**: Test blocks at file end, use `std.testing`; dependency injection for testability
- **Architecture**: Modular design with core/features/framework layers; compile-time feature selection

## Testing Organization
- **Unit tests**: `tests/unit/` - individual component tests
- **Integration tests**: `tests/integration/` - system interaction tests
- **Cross-platform tests**: `tests/cross-platform/` - platform-specific validation

## Development Workflow
1. Format code: `zig fmt .`
2. Build and test: `zig build test`
3. Run specific tests: `zig test tests/unit/test_build.zig`
4. Generate docs: `zig build docs`

---
Follow these conventions to keep the codebase consistent and buildable.</content>
<parameter name="filePath">AGENTS.md