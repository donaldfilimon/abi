# Migration Guide: Upgrading from ABI Framework 0.1.x to 0.2.0

## 🚀 Overview

ABI Framework 0.2.0 introduces significant architectural improvements and breaking changes. This guide will help you migrate your existing code to take advantage of the new features.

## ⚠️ Breaking Changes

### 1. Core Module Structure

**Before (0.1.x):**
```zig
const abi = @import("abi");
const core = abi.core; // Old core structure
```

**After (0.2.0):**
```zig
const abi = @import("abi");
const core = abi.core; // New core structure with I/O abstraction
```

### 2. Framework Initialization

**Before (0.1.x):**
```zig
var framework = try abi.createDefaultFramework(allocator);
defer framework.deinit();
```

**After (0.2.0):**
```zig
var framework = try abi.init(allocator, .{});
defer abi.shutdown(framework);
```

### 3. I/O Operations

**Before (0.1.x):**
```zig
std.debug.print("Hello, world!\n", .{});
```

**After (0.2.0):**
```zig
const writer = abi.core.Writer.stdout();
try writer.print("Hello, world!\n", .{});
```

### 4. Error Handling

**Before (0.1.x):**
```zig
const result = operation() catch |err| {
    std.log.err("Operation failed: {}", .{err});
    return err;
};
```

**After (0.2.0):**
```zig
const result = operation() catch |err| {
    const ctx = abi.core.ErrorContext.init(err, "Operation failed")
        .withLocation(abi.core.here())
        .withContext("Additional context here");
    
    std.log.err("{}", .{ctx});
    return err;
};
```

## 🔧 Migration Steps

### Step 1: Update Dependencies

Update your `build.zig.zon`:
```zig
.{
    .name = "your-project",
    .version = "0.2.0",
    .minimum_zig_version = "0.16.0", // Updated from 0.15.1
    .dependencies = .{
        .abi = .{
            .url = "https://github.com/donaldfilimon/abi/archive/main.tar.gz",
            .hash = "your-hash-here",
        },
    },
}
```

### Step 2: Update Imports

**Before:**
```zig
const abi = @import("lib/mod.zig");
```

**After:**
```zig
const abi = @import("src/mod.zig");
```

### Step 3: Update Framework Initialization

**Before:**
```zig
var framework = try abi.createDefaultFramework(allocator);
defer framework.deinit();

try framework.start();
// ... use framework
framework.stop();
```

**After:**
```zig
var framework = try abi.init(allocator, .{
    .enable_ai = true,
    .enable_gpu = true,
    .enable_database = true,
});
defer abi.shutdown(framework);

// Framework is automatically started
// ... use framework
// Framework is automatically stopped on shutdown
```

### Step 4: Update I/O Operations

**Before:**
```zig
std.debug.print("Processing {d} items\n", .{count});
std.log.info("Status: {s}", .{status});
```

**After:**
```zig
const writer = abi.core.Writer.stdout();
try writer.print("Processing {d} items\n", .{count});

// For logging, you can still use std.log
std.log.info("Status: {s}", .{status});

// Or use the I/O abstraction for consistency
try writer.print("Status: {s}\n", .{status});
```

### Step 5: Update Error Handling

**Before:**
```zig
const result = riskyOperation() catch |err| {
    std.log.err("Operation failed: {}", .{err});
    return err;
};
```

**After:**
```zig
const result = riskyOperation() catch |err| {
    const ctx = abi.core.ErrorContext.init(err, "Risky operation failed")
        .withLocation(abi.core.here())
        .withContext("This operation processes user data");
    
    std.log.err("{}", .{ctx});
    return err;
};
```

### Step 6: Update Test Code

**Before:**
```zig
test "my test" {
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    // ... test code
}
```

**After:**
```zig
test "my test" {
    var framework = try abi.init(std.testing.allocator, .{});
    defer abi.shutdown(framework);
    
    // Use test writer for output
    const test_writer = abi.core.TestWriter.init(std.testing.allocator);
    defer test_writer.deinit();
    
    // ... test code using test_writer.writer()
}
```

## 🆕 New Features to Adopt

### 1. I/O Abstraction Layer

The new I/O system makes your code more testable:

```zig
// Instead of direct stdout
std.debug.print("Result: {d}\n", .{result});

// Use the abstraction
const writer = abi.core.Writer.stdout();
try writer.print("Result: {d}\n", .{result});

// In tests, use TestWriter
const test_writer = abi.core.TestWriter.init(allocator);
defer test_writer.deinit();
try processData(test_writer.writer(), data);
```

### 2. Rich Error Context

Get better error messages with context:

```zig
const result = operation() catch |err| {
    const ctx = abi.core.ErrorContext.init(err, "User authentication failed")
        .withLocation(abi.core.here())
        .withContext("User ID: {s}", .{user_id})
        .withContext("Attempt: {d}/3", .{attempt_count});
    
    std.log.err("Authentication error: {}", .{ctx});
    return err;
};
```

### 3. Modular Build System

Use feature flags to reduce binary size:

```bash
# Build with only AI features
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=false

# Build with specific GPU backend
zig build -Denable-gpu=true -Dgpu-cuda=true
```

### 4. Improved Testing

The new test organization provides better structure:

```zig
// In tests/unit/my_feature_test.zig
const std = @import("std");
const abi = @import("../../src/mod.zig");

test "feature works correctly" {
    var framework = try abi.init(std.testing.allocator, .{});
    defer abi.shutdown(framework);
    
    // Test your feature
}
```

## 🔍 Common Migration Issues

### Issue 1: Missing Core Functions

**Problem:** `abi.core.allocators` not found
**Solution:** Use standard Zig allocators or the new core utilities

**Before:**
```zig
var tracked = abi.core.allocators.AllocatorFactory.createTracked(allocator, limit);
```

**After:**
```zig
// Use standard Zig allocators
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const tracked_allocator = gpa.allocator();
```

### Issue 2: Framework API Changes

**Problem:** `framework.start()` and `framework.stop()` not found
**Solution:** The new framework automatically manages lifecycle

**Before:**
```zig
try framework.start();
// ... use framework
framework.stop();
```

**After:**
```zig
// Framework starts automatically after init()
// ... use framework
// Framework stops automatically on shutdown()
```

### Issue 3: Import Path Changes

**Problem:** `lib/mod.zig` not found
**Solution:** Update to use `src/mod.zig`

**Before:**
```zig
const abi = @import("lib/mod.zig");
```

**After:**
```zig
const abi = @import("src/mod.zig");
```

## 📚 Additional Resources

- [New Architecture Guide](docs/PROJECT_STRUCTURE.md)
- [Module Organization](docs/MODULE_ORGANIZATION.md)
- [API Reference](docs/generated/)
- [Examples](examples/)

## 🆘 Getting Help

If you encounter issues during migration:

1. Check the [CHANGELOG.md](CHANGELOG.md) for detailed changes
2. Review the [examples](examples/) for usage patterns
3. Open an issue on GitHub with your specific error
4. Join the community discussions

## ✅ Migration Checklist

- [ ] Updated `build.zig.zon` with new version and Zig requirement
- [ ] Changed imports from `lib/mod.zig` to `src/mod.zig`
- [ ] Updated framework initialization to use `abi.init()` and `abi.shutdown()`
- [ ] Replaced direct I/O with `abi.core.Writer` abstraction
- [ ] Enhanced error handling with `abi.core.ErrorContext`
- [ ] Updated test code to use new patterns
- [ ] Verified all features work with new architecture
- [ ] Updated documentation and comments

---

**Happy migrating! 🚀**

The new ABI Framework 0.2.0 provides a more robust, testable, and maintainable foundation for your projects. While migration requires some changes, the benefits of the new architecture will make your code more reliable and easier to maintain.