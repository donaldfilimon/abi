# ABI Framework Migration Guide

## Overview

This guide helps you migrate from the old ABI Framework structure to the new v0.2.0 architecture after the mega refactor.

## What Changed

### Directory Structure

**Before (v0.1.x):**
```
abi/
├── src/                    # Mixed source files
│   ├── core/              # Core utilities
│   ├── features/          # Feature modules
│   ├── tools/             # Tools mixed with source
│   ├── examples/          # Examples mixed with source
│   └── tests/             # Tests mixed with source
```

**After (v0.2.0):**
```
abi/
├── lib/                    # Primary library source
│   ├── core/              # Core utilities (enhanced)
│   ├── features/          # Feature modules
│   ├── framework/         # Framework infrastructure
│   └── shared/            # Shared utilities
├── tools/                 # Development tools and CLI
├── examples/             # Standalone examples
├── tests/                # Comprehensive test suite
└── benchmarks/           # Performance tests
```

### Key Changes

1. **Single Source of Truth**: All library code is now in `lib/`
2. **Clean Separation**: Tools, examples, and tests are in separate directories
3. **Enhanced Core**: New I/O abstraction and diagnostics system
4. **Modern Patterns**: Eliminated `usingnamespace`, explicit exports
5. **Build System**: Updated to use new structure

## Migration Steps

### 1. Update Imports

**Before:**
```zig
const abi = @import("abi");
const core = @import("abi").core;
const features = @import("abi").features;
```

**After:**
```zig
const abi = @import("abi");
const core = abi.core;
const features = abi.features;
```

### 2. Update Build Commands

**Before:**
```bash
zig build
zig build test
zig build examples
```

**After:**
```bash
zig build                    # Build library
zig build -Dcli=true        # Build CLI tool
zig build -Dexamples=true   # Build examples
zig build test              # Run tests
```

### 3. Update I/O Usage

**Before:**
```zig
std.debug.print("Processing {d} items\n", .{count});
```

**After:**
```zig
// Inject writer for testability
try writer.print("Processing {d} items\n", .{count});

// Or use the new I/O abstraction
const io = @import("abi").core.io;
const writer = io.OutputContext.init(std.io.getStdOut().writer());
try writer.print("Processing {d} items\n", .{count});
```

### 4. Update Error Handling

**Before:**
```zig
return error.OperationFailed;
```

**After:**
```zig
const diagnostics = @import("abi").core.diagnostics;
const ctx = diagnostics.ErrorContext.init(error.OperationFailed, "Failed to process data")
    .withLocation(diagnostics.here())
    .withContext("Additional context here");
return ctx;
```

### 5. Update Module Exports

**Before:**
```zig
pub const wdbx = struct {
    pub usingnamespace features.database.unified;
};
```

**After:**
```zig
pub const wdbx = struct {
    // Explicit exports instead of usingnamespace
    pub const createDatabase = features.database.unified.createDatabase;
    pub const connectDatabase = features.database.unified.connectDatabase;
    // ... other explicit exports
};
```

## Breaking Changes

### None! 🎉

The public API remains exactly the same. All breaking changes are internal:

- **Internal Structure**: Only affects internal organization
- **Build System**: Updated but backward compatible
- **Module Exports**: Same public interface
- **API Surface**: No changes to public functions

## New Features

### 1. I/O Abstraction Layer

```zig
const io = @import("abi").core.io;

// Create a writer for testing
const test_writer = io.TestWriter.init(allocator);
defer test_writer.deinit();

// Use in your code
try writer.print("Hello, {s}!\n", .{"World"});

// Capture output for testing
const output = test_writer.getOutput();
try std.testing.expectEqualStrings("Hello, World!\n", output);
```

### 2. Rich Diagnostics System

```zig
const diagnostics = @import("abi").core.diagnostics;

// Create diagnostic with context
const diag = diagnostics.Diagnostic.init(.err, "Operation failed")
    .withLocation(diagnostics.here())
    .withContext("Additional context");

// Collect diagnostics
var collector = diagnostics.DiagnosticCollector.init(allocator);
defer collector.deinit();

try collector.add(diag);
try collector.emit(writer);
```

### 3. Enhanced Collections

```zig
const collections = @import("abi").core.collections;

// Use utility functions for initialization
const list = collections.utils.createArrayList(i32, allocator);
defer list.deinit();

const map = collections.utils.createStringHashMap(i32, allocator);
defer map.deinit();
```

## Testing Your Migration

### 1. Build Test

```bash
# Test that everything builds
zig build

# Test CLI builds
zig build -Dcli=true

# Test examples build
zig build -Dexamples=true
```

### 2. Run Tests

```bash
# Run all tests
zig build test

# Run specific test
zig test lib/core/collections.zig
```

### 3. Check Imports

```bash
# Verify no old imports remain
grep -r "src/" --include="*.zig" .
# Should return no results
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're importing from `abi` not `src/`
2. **Build Errors**: Ensure you're using the new build commands
3. **Test Failures**: Update test imports to use new structure

### Getting Help

- Check the [README.md](README.md) for updated usage
- Review [MEGA_REFACTOR_COMPLETE.md](MEGA_REFACTOR_COMPLETE.md) for details
- Open an issue if you encounter problems

## Benefits of Migration

### For Developers
- **Cleaner Codebase**: Easy to navigate and understand
- **Better Testing**: I/O abstraction enables dependency injection
- **Modern Patterns**: Zig 0.16 best practices
- **Rich Debugging**: Comprehensive diagnostics system

### For Users
- **Same API**: No breaking changes to public interface
- **Better Performance**: Optimized build system
- **Improved Reliability**: Better error handling
- **Enhanced Debugging**: Rich error messages

### For Maintainers
- **Reduced Complexity**: Single source of truth
- **Easier Maintenance**: Clear module boundaries
- **Better Testing**: Comprehensive test coverage
- **Modern Code**: Future-proof architecture

## Conclusion

The migration to v0.2.0 is straightforward with no breaking changes to your code. The new architecture provides better organization, modern patterns, and enhanced capabilities while maintaining full backward compatibility.

---

**Happy coding with the new ABI Framework! 🚀**