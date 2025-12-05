# ABI Framework – Agent Guidelines

## Build/Test Commands
```bash
zig build                    # Build library + CLI
zig build test               # Run all tests
zig test <file>              # Run single test file
zig fmt --check .            # Check code formatting
zig build run                # Run CLI
zig build -Denable-gpu=true -Denable-ai=true  # Build with features
```

## Code Style Guidelines
- **Formatting**: 4 spaces, no tabs
- **Naming**: PascalCase (types), snake_case (functions/variables)
- **Documentation**: `//!` for modules, `///` for public API
- **Imports**: Explicit imports, avoid `usingnamespace`
- **Error Handling**: Use `!` return types, `try`/`catch`, custom error enums
- **Memory**: `defer` for cleanup, explicit allocators
- **Zig 0.16**: `std.debug.print`, `vtable.*` allocators, async I/O
- **Testing**: Test blocks at file end, use `std.testing`

---
Follow these conventions to keep the codebase consistent and buildable.
