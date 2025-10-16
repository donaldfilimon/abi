# Developer Onboarding Guide - ABI Framework 0.2.0

## 🚀 Welcome to ABI Framework Development!

This guide will help you get started with developing on the ABI Framework, whether you're contributing to the core framework, building features, or creating applications.

## 📋 Prerequisites

### Required Tools
- **Zig 0.16.0-dev** or later
- **Git** for version control
- **Make** or **Ninja** for build system
- **curl** for downloading dependencies

### Optional Tools
- **Zed Editor** with Zig language support
- **VS Code** with Zig extension
- **Docker** for containerized development

## 🏗️ Project Structure Overview

```
abi/
├── src/                    # Main source code
│   ├── core/              # Core infrastructure (I/O, errors, diagnostics)
│   ├── features/          # Feature modules (AI, GPU, Database, Web, etc.)
│   ├── framework/         # Runtime orchestration
│   ├── shared/            # Cross-cutting utilities
│   └── mod.zig            # Public API entry point
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── examples/              # Usage examples
├── docs/                  # Documentation
└── build.zig              # Build configuration
```

## 🛠️ Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
```

### 2. Install Zig

**Option A: Download from ziglang.org**
```bash
# Download Zig 0.16.0-dev
curl -L https://ziglang.org/download/0.16.0-dev/zig-linux-x86_64-0.16.0-dev.tar.xz -o zig.tar.xz
tar -xf zig.tar.xz
export PATH=$PWD/zig-linux-x86_64-0.16.0-dev:$PATH
```

**Option B: Use package manager**
```bash
# Ubuntu/Debian
sudo apt install zig

# macOS
brew install zig

# Windows
winget install zig
```

### 3. Verify Installation

```bash
zig version
# Should show 0.16.0-dev or later
```

### 4. Build the Framework

```bash
# Build everything
zig build

# Build with specific features
zig build -Denable-ai=true -Denable-gpu=true

# Build and run tests
zig build test

# Build examples
zig build examples
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
zig build test

# Run unit tests only
zig build test-unit

# Run integration tests only
zig build test-integration

# Run with verbose output
zig build test --verbose
```

### Writing Tests

**Unit Test Example:**
```zig
// tests/unit/my_feature_test.zig
const std = @import("std");
const abi = @import("../../src/mod.zig");

test "my feature works correctly" {
    var framework = try abi.init(std.testing.allocator, .{});
    defer abi.shutdown(framework);
    
    // Test your feature
    const result = try myFeature();
    try std.testing.expectEqual(@as(u32, 42), result);
}
```

**Integration Test Example:**
```zig
// tests/integration/feature_integration_test.zig
const std = @import("std");
const abi = @import("../../src/mod.zig");

test "features work together" {
    var framework = try abi.init(std.testing.allocator, .{
        .enable_ai = true,
        .enable_database = true,
    });
    defer abi.shutdown(framework);
    
    // Test feature interaction
}
```

## 🏗️ Building Features

### 1. Create a New Feature Module

```bash
mkdir -p src/features/my_feature
touch src/features/my_feature/mod.zig
touch src/features/my_feature/my_feature.zig
```

### 2. Implement the Feature

**src/features/my_feature/mod.zig:**
```zig
//! My Feature Module
//!
//! Description of what this feature does

pub const my_feature = @import("my_feature.zig");

// Re-export public APIs
pub const MyFeature = my_feature.MyFeature;
pub const MyFeatureError = my_feature.MyFeatureError;
```

**src/features/my_feature/my_feature.zig:**
```zig
//! My Feature Implementation

const std = @import("std");
const core = @import("../../core/mod_new.zig");

pub const MyFeatureError = error{
    InvalidInput,
    OperationFailed,
};

pub const MyFeature = struct {
    allocator: std.mem.Allocator,
    writer: core.Writer,

    pub fn init(allocator: std.mem.Allocator, writer: core.Writer) MyFeature {
        return .{
            .allocator = allocator,
            .writer = writer,
        };
    }

    pub fn process(self: *MyFeature, input: []const u8) !void {
        try self.writer.print("Processing: {s}\n", .{input});
        // Implementation here
    }
};
```

### 3. Register the Feature

**src/features/mod.zig:**
```zig
// Add your feature
pub const my_feature = @import("my_feature/mod.zig");
```

**src/mod.zig:**
```zig
// Add to the public API
pub const my_feature = features.my_feature;
```

## 🔧 Core Infrastructure Usage

### I/O Abstraction

```zig
const abi = @import("abi");
const core = abi.core;

// Use stdout writer
const writer = core.Writer.stdout();
try writer.print("Hello, world!\n", .{});

// Use test writer in tests
const test_writer = core.TestWriter.init(allocator);
defer test_writer.deinit();
try processData(test_writer.writer(), data);
```

### Error Handling

```zig
const core = abi.core;

const result = operation() catch |err| {
    const ctx = core.ErrorContext.init(err, "Operation failed")
        .withLocation(core.here())
        .withContext("User ID: {s}", .{user_id});
    
    std.log.err("{}", .{ctx});
    return err;
};
```

### Diagnostics

```zig
const core = abi.core;

var diagnostics = core.DiagnosticCollector.init(allocator);
defer diagnostics.deinit();

try diagnostics.add(.{
    .severity = .warning,
    .message = "This is a warning",
    .location = core.here(),
});

// Process diagnostics
for (diagnostics.items) |diag| {
    try writer.print("{}", .{diag});
}
```

## 📚 Documentation

### Building Documentation

```bash
# Generate API documentation
zig build docs

# Documentation will be in docs/generated/
```

### Writing Documentation

```zig
//! Module-level documentation
//!
//! This module provides...

/// Function documentation
/// 
/// Detailed description of what this function does.
/// 
/// # Parameters
/// - `allocator`: Memory allocator to use
/// - `input`: Input data to process
/// 
/// # Returns
/// - `Result`: Processing result
/// 
/// # Errors
/// - `InvalidInput`: When input is malformed
/// - `OutOfMemory`: When allocation fails
pub fn processData(allocator: std.mem.Allocator, input: []const u8) !Result {
    // Implementation
}
```

## 🚀 Contributing

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/abi.git
cd abi
git remote add upstream https://github.com/donaldfilimon/abi.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/my-awesome-feature
```

### 3. Make Changes

- Write your code following the style guide
- Add tests for new functionality
- Update documentation as needed
- Run tests to ensure everything works

### 4. Commit Changes

```bash
# Follow conventional commits
git add .
git commit -m "feat: add awesome new feature"
```

### 5. Push and Create PR

```bash
git push origin feature/my-awesome-feature
# Create pull request on GitHub
```

## 📏 Code Style Guide

### Naming Conventions

- **Functions**: `snake_case`
- **Types**: `CamelCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Variables**: `snake_case`

### Formatting

```bash
# Format code
zig fmt src/

# Check formatting
zig fmt --check src/
```

### Error Handling

```zig
// Use tagged unions for errors
const MyError = error{
    InvalidInput,
    OutOfMemory,
};

// Provide context with errors
const result = operation() catch |err| {
    const ctx = core.ErrorContext.init(err, "Operation failed")
        .withLocation(core.here())
        .withContext("Additional info");
    return err;
};
```

### Memory Management

```zig
// Always use explicit allocators
pub fn createThing(allocator: std.mem.Allocator) !*Thing {
    const thing = try allocator.create(Thing);
    errdefer allocator.destroy(thing);
    
    // Initialize thing
    thing.* = .{};
    
    return thing;
}

// Clean up resources
defer thing.deinit();
```

## 🔍 Debugging

### Debug Builds

```bash
# Build with debug info
zig build -Doptimize=Debug

# Run with debugger
gdb ./zig-out/bin/abi
```

### Logging

```zig
// Use structured logging
std.log.info("Processing {d} items", .{count});
std.log.debug("Debug info: {s}", .{debug_data});
std.log.err("Error occurred: {}", .{error});
```

### Profiling

```bash
# Build with profiling
zig build -Doptimize=ReleaseSafe -Dprofiling=true

# Run with profiler
perf record ./zig-out/bin/abi
perf report
```

## 🐛 Common Issues

### Import Errors

**Problem**: `@import("abi")` not found
**Solution**: Make sure you're importing from the correct path

```zig
// Correct
const abi = @import("src/mod.zig");

// Or if using as dependency
const abi = @import("abi");
```

### Build Errors

**Problem**: Feature not found
**Solution**: Enable the feature in build

```bash
zig build -Denable-my-feature=true
```

### Test Failures

**Problem**: Tests failing with memory errors
**Solution**: Use proper allocators in tests

```zig
test "my test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Use allocator in test
}
```

## 📞 Getting Help

### Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **API Reference**: [docs/generated/](docs/generated/)
- **Migration Guide**: [MIGRATION_GUIDE_0.2.0.md](MIGRATION_GUIDE_0.2.0.md)

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Real-time chat and support

### Code Review

- All changes require code review
- Tests must pass before merging
- Documentation must be updated
- Follow the style guide

## 🎯 Next Steps

1. **Explore Examples**: Check out the examples in `examples/`
2. **Read Documentation**: Browse the comprehensive docs
3. **Run Tests**: Make sure everything works
4. **Start Contributing**: Pick an issue or create a feature
5. **Join Community**: Connect with other developers

---

**Happy coding! 🚀**

The ABI Framework is designed to be developer-friendly and powerful. If you have questions or need help, don't hesitate to reach out to the community!