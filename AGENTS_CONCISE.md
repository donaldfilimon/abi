# ABI Framework – Agent Guidelines

## Build/Test Commands
- `zig build` - Build library + CLI
- `zig build test` - Run all tests
- `zig test tests/unit/test_*.zig` - Run specific unit test
- `zig build run` - Run CLI
- `zig fmt .` - Format code
- `zig build docs` - Generate docs

## Code Style Guidelines
- **Formatting**: 4 spaces, 100 char lines, `zig fmt` required
- **Naming**: PascalCase for types, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Imports**: Group std imports first, relative imports for local modules, explicit imports only
- **Error Handling**: Use `!` return types, `errdefer` for cleanup, `try` for propagation
- **Memory**: Use GPA for general allocation, arena allocators for temporaries
- **Documentation**: Module-level docs with `//!`, function docs with `///`, examples in ```zig blocks
- **Testing**: `test` blocks at file end, use `testing.allocator`, comprehensive coverage required

## Development Workflow
1. Format with `zig fmt .`
2. Build and test with `zig build test`
3. Run specific tests as needed
4. Update documentation for public APIs
5. Follow security best practices (no secrets in code, validate inputs)</content>
<parameter name="filePath">AGENTS.md