# ABI Framework – Agent Guidelines

## Build/Test Commands
```bash
zig build                    # Build library + CLI
zig build test              # Run all tests
zig test tests/mod.zig      # Run smoke tests directly
zig fmt .                   # Format code (required before commits)
```

## Feature Build Options
```bash
# Core features
zig build -Denable-gpu=true -Denable-ai=true -Denable-web=true -Denable-database=true
zig build -Denable-gpu=false -Denable-ai=true  # AI only
zig build -Denable-database=true -Denable-web=false  # Database only

# Optimization levels
zig build -Doptimize=Debug         # Default: includes debug info
zig build -Doptimize=ReleaseFast   # Maximum performance, minimal safety
zig build -Doptimize=ReleaseSafe   # Performance with safety checks
zig build -Doptimize=ReleaseSmall  # Minimize binary size
```

## Code Style Guidelines
- **Formatting**: 4 spaces, no tabs; lines < 100 chars; `zig fmt` required
- **Naming**: PascalCase types, snake_case functions/variables, UPPER_SNAKE_CASE constants, snake_case files
- **Imports**: Group std first, then local; explicit imports only (no `usingnamespace`)
- **Error Handling**: Use `!` return types, `errdefer` for cleanup, specific error enums
- **Memory**: Use `defer` for cleanup, GPA for testing, arena allocators for temporaries
- **Documentation**: `//!` module docs, `///` function docs with # Parameters/Returns/Errors sections
