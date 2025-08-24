# Testing Guide for Abi AI Framework

## Overview

This guide covers the testing infrastructure for the Abi AI Framework, including how to run tests, interpret results, and contribute new tests.

## Quick Start

### Running All Tests
```bash
# Using the build system (recommended)
zig build test

# Using the custom test runner
zig run scripts/test_all.zig
```

### Running Individual Test Files
```bash
# Core functionality
zig test src/core/errors.zig
zig test src/core/mod.zig

# SIMD operations
zig test tests/test_simd_vector.zig

# Database operations
zig test tests/test_database.zig

# Basic tests
zig test tests/dummy_test.zig
```

## Test Structure

### Test Files Organization
```
tests/
├── dummy_test.zig          # Basic functionality tests
├── test_simd_vector.zig    # SIMD operations and vector math
├── test_database.zig       # Vector database operations
└── test_weather.zig        # Weather service integration

src/
├── core/
│   ├── errors.zig          # Error handling tests
│   └── mod.zig             # Core utilities tests
├── ai/
│   └── mod.zig             # AI agent tests
├── cell/
│   └── main.zig            # Cell language tests
└── ...
```

### Test Categories

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test interactions between modules
3. **Performance Tests**: Benchmark operations and measure performance
4. **Compatibility Tests**: Ensure compatibility across Zig versions

## Test Results

### Interpreting Output
```
All X tests passed.        ✅ Success
test result: FAILED        ❌ Failure
error: compilation failed  ❌ Build error
```

### Performance Metrics
Tests include performance benchmarks:
- SIMD speedup ratios
- Vector operation timings
- Database query performance

## CI/CD Integration

### GitHub Actions
Tests run automatically on:
- **Push** to main/develop branches
- **Pull requests** targeting main/develop
- **Multiple platforms**: Ubuntu, Windows, macOS
- **Multiple Zig versions**: 0.12.0, 0.13.0

### Status Badges
```
[![CI](https://github.com/your-org/abi/workflows/CI/badge.svg)](https://github.com/your-org/abi/actions)
```

## Writing Tests

### Test Structure
```zig
const std = @import("std");
const testing = std.testing;

test "feature description" {
    // Arrange
    const allocator = testing.allocator;

    // Act
    const result = someFunction(input);

    // Assert
    try testing.expectEqual(expected, result);
}
```

### Testing Patterns

#### 1. Error Testing
```zig
test "error handling" {
    const result = functionThatCanFail();
    try testing.expectError(expectedError, result);
}
```

#### 2. Memory Testing
```zig
test "memory allocation" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(u8, 100);
    defer allocator.free(data);

    // Test operations
}
```

#### 3. Performance Testing
```zig
test "performance benchmark" {
    const allocator = testing.allocator;

    var timer = try std.time.Timer.start();
    // Run operation
    const time = timer.read();

    std.debug.print("Operation took: {}ns\n", .{time});
}
```

## Troubleshooting

### Common Issues

#### 1. Compilation Errors
- **@typeInfo compatibility**: Framework uses Zig 0.16.0-dev with updated @typeInfo structure
- **ArrayList API changes**: Different Zig versions have different ArrayList APIs
- **Format string issues**: Some Zig versions require explicit format specifiers

#### 2. Test Failures
- **Memory leaks**: Use `defer` and proper cleanup in tests
- **Race conditions**: Ensure tests don't interfere with each other
- **Platform differences**: Some features may behave differently across platforms

#### 3. Performance Issues
- **Debug builds**: Tests run slower in debug mode
- **Memory allocation**: Use appropriate allocators for different test scenarios
- **Benchmark variability**: Performance tests may vary between runs

### Debugging Failed Tests
```bash
# Run with verbose output
zig test --verbose

# Run specific test
zig test --test-filter "test name"

# Debug memory issues
zig test --test-filter "memory" --verbose
```

## Contributing Tests

### Guidelines
1. **Test Coverage**: Aim for high coverage of public APIs
2. **Descriptive Names**: Use clear, descriptive test names
3. **Independent Tests**: Tests should not depend on each other
4. **Cleanup**: Always clean up resources (memory, files, etc.)
5. **Documentation**: Document complex test scenarios

### Adding New Tests
1. Create test file in appropriate directory
2. Follow naming convention: `test_*.zig` or `*_test.zig`
3. Add to CI/CD configuration if needed
4. Update documentation

## Test Coverage Goals

### Current Status
- **Core modules**: ✅ Well tested
- **SIMD operations**: ✅ Comprehensive coverage
- **Database operations**: ✅ Basic functionality covered
- **AI modules**: ⚠️ Limited coverage
- **Networking**: ⚠️ Not yet tested
- **Integration**: ⚠️ Limited coverage

### Future Improvements
- [ ] Add AI model testing
- [ ] Expand networking test coverage
- [ ] Add integration tests
- [ ] Implement property-based testing
- [ ] Add fuzz testing

## Performance Testing

### Benchmark Categories
1. **SIMD Operations**: Vector math, dot products, matrix operations
2. **Database Operations**: Insertion, search, indexing
3. **Memory Management**: Allocation patterns, cache efficiency
4. **AI Operations**: Model inference, training loops

### Running Benchmarks
```bash
# Run all benchmarks
zig test --test-filter "benchmark"

# Run specific benchmark
zig test --test-filter "SIMD performance"
```

## Platform-Specific Testing

### Cross-Platform Considerations
- **SIMD availability**: Different CPU architectures
- **Memory alignment**: Platform-specific alignment requirements
- **File system**: Path handling across platforms
- **Networking**: Socket behavior differences

### Platform Testing
Tests run on:
- **Linux** (Ubuntu 20.04/22.04)
- **Windows** (Windows 10/11)
- **macOS** (macOS 11/12/13)

## Best Practices

### Test Design
1. **Arrange-Act-Assert**: Follow the AAA pattern
2. **Single Responsibility**: Each test should verify one thing
3. **Descriptive Names**: Use descriptive test names
4. **Edge Cases**: Test boundary conditions
5. **Error Paths**: Test error conditions

### Code Quality
1. **No Side Effects**: Tests should not affect each other
2. **Deterministic**: Tests should produce consistent results
3. **Fast Execution**: Tests should run quickly
4. **Clear Failure Messages**: Provide clear error messages

### Maintenance
1. **Regular Updates**: Keep tests up to date with code changes
2. **Refactoring**: Refactor tests when code changes
3. **Documentation**: Document complex test scenarios
4. **Review Process**: Review test changes with code changes

## Support

### Getting Help
- Check existing tests for patterns
- Review test documentation
- Ask in discussions/issues

### Reporting Issues
- Include test output
- Provide system information
- Describe expected vs actual behavior
