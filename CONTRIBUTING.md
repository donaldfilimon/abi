# Contributing to Abi AI Framework

Thank you for your interest in contributing to the Abi AI Framework! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/abi.git
   cd abi
   ```

2. **Set Up Development Environment**
   - Install Zig 0.14.1 or later
   - Install Bun (recommended): `curl -fsSL https://bun.sh/install | bash`
   - Set up your editor with Zig language support

3. **Build the Project**
   ```bash
   zig build -Doptimize=Debug
   zig build test
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Code Style Guidelines

#### Zig Style
- Use 4 spaces for indentation
- Keep lines under 100 characters
- Use descriptive variable names
- Add doc comments for public functions

```zig
/// Calculate the squared Euclidean distance between two vectors.
/// Vectors must have the same length.
pub fn distanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    // Implementation...
}
```

#### Error Handling
- Use explicit error types
- Document error conditions
- Handle errors at appropriate levels

```zig
pub const DatabaseError = error{
    InvalidDimension,
    BufferTooSmall,
    CorruptedData,
};
```

### 4. Testing

Write tests for all new functionality:

```zig
test "vector distance calculation" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    const distance = distanceSquared(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), distance, 0.001);
}
```

Run tests:
```bash
zig build test
zig build test-agent
zig build test-database
```

### 5. Performance Considerations

- Profile performance-critical code
- Use SIMD operations where beneficial
- Consider memory allocation patterns
- Document performance characteristics

```zig
/// Process text using SIMD operations.
/// Performance: ~3GB/s on modern x86_64 CPUs with AVX2.
pub fn processText(text: []const u8) void {
    // Implementation...
}
```

### 6. Documentation

- Add doc comments to all public APIs
- Update README.md for significant features
- Add examples to the `examples/` directory
- Update module documentation in `docs/`

## Submitting Changes

### 1. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "feat: Add GPU-accelerated matrix multiplication

- Implement GEMM using compute shaders
- Add benchmarks showing 10x speedup
- Support float32 and float64 types"
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template:
   - Describe what changes you made
   - Explain why the changes are needed
   - List any breaking changes
   - Reference related issues

### 4. PR Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## Areas for Contribution

### High Priority

- **GPU Backend Completion**: Implement missing GPU backend functionality
- **WebAssembly Support**: Add WASM target support
- **Documentation**: Improve API documentation and examples
- **Performance**: Optimize critical paths

### Feature Ideas

- **Advanced Neural Networks**: Implement CNN, RNN architectures
- **Distributed Computing**: Add network-based vector database
- **Language Bindings**: Create bindings for other languages
- **Visualization**: Add data visualization capabilities

### Bug Fixes

Check the [issue tracker](https://github.com/yourusername/abi/issues) for:
- Bugs labeled "good first issue"
- Performance improvements
- Platform-specific issues

## Testing Guidelines

### Unit Tests
- Test individual functions and modules
- Cover edge cases and error conditions
- Use property-based testing where appropriate

### Integration Tests
- Test module interactions
- Verify end-to-end workflows
- Test platform-specific behavior

### Benchmark Tests
- Add benchmarks for performance-critical code
- Compare against baseline performance
- Document performance characteristics

## Release Process

1. **Version Bumping**: Update version in `build.zig.zon`
2. **Changelog**: Update CHANGELOG.md
3. **Documentation**: Ensure docs are up-to-date
4. **Testing**: Run full test suite on all platforms
5. **Tag Release**: Create Git tag with version

## Questions?

- Open an issue for bugs or feature requests
- Join our Discord for real-time discussion
- Check existing issues and PRs before starting work

Thank you for contributing to Abi AI Framework! 