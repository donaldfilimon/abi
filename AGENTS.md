# Agentic Coding Guidelines for ABI Framework

This document provides comprehensive guidelines for AI coding agents working on the ABI framework repository.

## Build and Test Commands
```bash
# Build entire project
zig build

# Run all tests
zig build test --summary all
```

## Code Style Guidelines

### File Structure
- **Module docs**: Use `//!` at file top for module-level documentation
- **Function docs**: Use `///` for all public functions
- **Test placement**: Tests go at end of file or in separate `*_test.zig` files
- **Import order**: `std` first, then internal modules alphabetically
- **No usingnamespace**: Always use qualified imports (`std.mem`, not `mem`)

### Naming Conventions
- **Types**: `PascalCase` (structs, enums, unions)
- **Functions**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Error sets**: `PascalCase` with `Error` suffix
- **Modules**: `snake_case` with `.zig` extension

### Formatting Rules
- **Indentation**: 4 spaces
- **Line length**: 100 characters maximum
- **Braces**: Same line for structs/functions, next line for control flow
- **Spacing**: Space around operators, no space in function calls

### Import Guidelines
- No `usingnamespace` allowed - always use qualified imports
- No circular imports - avoid modules importing each other
- Prefer `@import` over `@cImport`

### Memory Management Patterns
- **Function parameters**: Always take `std.mem.Allocator` as first parameter for allocation functions
- **Ownership transfer**: Document who owns allocated memory (caller or callee)
- **Cleanup patterns**: Use `errdefer` for cleanup on error paths
- **Arena allocators**: Use for scratch work, reset between operations
- **Object pools**: Consider for frequently allocated objects

### Error Handling Guidelines
- **Specific errors**: Use descriptive error names, avoid `error.Generic` or `error.Failed`
- **Error propagation**: Use `try` for expected errors, handle appropriately
- **Error context**: Add contextual information to errors for debugging
- **No silent failures**: Use logging for non-critical errors

### Testing Guidelines
- **Test allocator**: Always use `std.testing.allocator` for tests
- **Resource cleanup**: Use `defer` for cleanup in tests
- **Error testing**: Use `std.testing.expectError` for error cases

### Performance Considerations
- **SIMD usage**: Use vectorized operations where possible
- **Cache-friendly designs**: Consider cache locality for hot paths
- **Minimal allocations**: Avoid unnecessary allocations in loops
- **Zero-copy**: Prefer views over copies where possible

### Zig 0.16 Specific Guidelines
- **Use Zig 0.16 features** where beneficial
- **Comptime checks**: Add compile-time assertions for type safety
- **Packed structs**: Use for binary serialization when appropriate
- **std.mem.bytesAsValue**: Use for type-safe struct reading
- **Avoid deprecated APIs**: Check Zig 0.16 release notes
- **Lazy evaluation**: Use `comptime` blocks for compile-time optimizations

## Module Organization

### Import Hierarchy
```
src/
├── abi.zig (main API surface)
├── framework/ (orchestration layer)
├── core/ (infrastructure: platform, version, memory utilities)
├── compute/ (runtime engine, GPU, concurrency)
├── features/ (high-level features: AI, database, web, network)
├── shared/ (common utilities: logging, HTTP, JSON, encoding, binary)
└── cli.zig (command-line interface)
```

### Feature Flags
- **`enable-gpu`**: Enable/disable GPU support
- **`enable-ai`**: Enable/disable AI features
- **`enable-web`**: Enable/disable web features
- **`enable-database`**: Enable/disable database features
- **`enable-network`**: Enable/disable distributed compute
- **`enable-profiling`**: Enable/disable profiling

### Module Documentation

Each module should have:
- Module-level `//!` doc explaining purpose and usage
- Public function documentation with `///`
- Examples for non-trivial APIs
- Notes about feature flags or build requirements

### Common Patterns

#### Serialization Pattern
```zig
// Binary cursor for reading
var cursor = try SerializationCursor.init(data);
defer cursor.deinit();
const value = try cursor.readInt(u32);
const slice = try cursor.readSlice();
```

#### Error Handling Pattern
```zig
pub fn processFile(path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try processContent(file.reader());
}
```

#### Memory Allocation Pattern
```zig
pub fn createBuffer(allocator: std.mem.Allocator, size: usize) ![]u8 {
    return try allocator.alloc(u8, size);
}
```

## Conventions

### Type Safety
- Use `comptime` assertions for compile-time checks
- Prefer explicit `@intCast` over implicit conversions
- Add unsigned integer checks to `readInt()` and `appendInt()`

### Documentation

Add `@param` and `@return` documentation for functions:
```zig
/// Process request with given parameters.
/// @param allocator Memory allocator for allocations
/// @param request The request to process
/// @return Processed result
pub fn processRequest(allocator: std.mem.Allocator, request: Request) !Result;
```

### Build Integration

Add feature modules to `build.zig`:
```zig
if (base_options.enable_gpu) {
    const gpu_module = b.createModule(.{ .root_source_file = "src/compute/gpu/mod.zig" });
    exe.addImport("gpu", gpu_module);
}
```

## Testing

### Property-Based Testing
Use property testing framework in `tests/property_tests.zig` for randomized testing:
```zig
try property_tests.checkProperty(allocator, myPropertyFunction, config, "my property description");
```

## Common Pitfalls to Avoid

1. **Circular imports**: Modules shouldn't import each other
2. **Memory leaks**: Always pair allocations with frees or use arena allocators
3. **Silent failures**: Log errors instead of returning silently
4. **Race conditions**: Use proper synchronization primitives
5. **Overly broad errors**: Be specific about what went wrong
6. **Implicit casts**: Prefer explicit `@intCast` over implicit conversions
7. **Uninitialized memory**: Use `undefined` after free, not zero-initialization

## Performance Tips

1. **Vectorized operations**: Use SIMD for bulk data processing
2. **Batch operations**: Group related operations to reduce overhead
3. **Arena allocators**: Use for temporary allocations
4. **Slice over copy**: Pass slices instead of allocating new buffers
5. **Compile-time evaluation**: Use `comptime` for constant folding

## Security Considerations

1. **Input validation**: Validate all external inputs
2. **Bounds checking**: Use Zig's bounds checking
3. **Safe string handling**: Use bounds-checked string operations
4. **Memory safety**: Zero out sensitive data after use
5. **Path safety**: Validate file paths to prevent traversal attacks
