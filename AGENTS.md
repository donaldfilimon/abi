# ABI Framework – Agent Guidelines

## Build/Test Commands
```bash
zig build                           # Build library + CLI
zig build test                      # Run all tests
zig build test -- <test_name>       # Run specific test (use test filter)
zig test tests/mod.zig              # Run smoke tests directly
zig fmt .                           # Format code (required before commits)
```

## Build Options & Configuration
```bash
zig build -Denable-gpu=true -Denable-ai=true -Denable-web=true -Denable-database=true  # All features
zig build -Denable-gpu=false -Denable-ai=true                                        # AI only
zig build -Doptimize=ReleaseFast                                                      # Max performance
zig build -Doptimize=Debug                                                            # Debug build
```
- Use `FrameworkConfiguration` for unified config (preferred); `FrameworkOptions` deprecated
- Check `config/default.zig` for examples

## Code Style Guidelines
- **Formatting**: 4 spaces, no tabs; lines < 100 chars; `zig fmt` required before commits
- **Naming**: PascalCase types, snake_case functions/variables, UPPER_SNAKE_CASE constants, snake_case files
- **Imports**: Group std first, then local imports; explicit imports only (no `usingnamespace`)
- **Error Handling**: Use `!` return types, `errdefer` for cleanup, specific error enums from `core.errors`
- **Memory**: Use `defer`/`errdefer` for cleanup, GPA for testing, arena allocators for temporaries
- **Documentation**: `//!` module docs, `///` function docs with Parameters/Returns/Errors sections
- **Types**: Use framework types from `core.types`, prefer explicit types over `anytype`
- **Testing**: Add tests to relevant test files, use `std.testing.expect` for assertions
