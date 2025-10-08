# Migration Guide: v0.1.0a → v0.2.0

## Overview

This guide helps you migrate from Abi Framework v0.1.0a to v0.2.0, which includes significant architectural improvements and new features.

## Breaking Changes Summary

1. ✅ **Build System**: New feature flags and build options
2. ✅ **Error Handling**: Unified error sets replace ad-hoc errors
3. ✅ **I/O Operations**: Writer injection replaces direct stdout/stderr
4. ⚠️ **Import Paths**: Some internal paths have changed
5. ⚠️ **Function Signatures**: Many functions accept new parameters

## Step-by-Step Migration

### 1. Update Build Configuration

#### Before (v0.1.0a)
```bash
zig build
zig build test
```

#### After (v0.2.0)
```bash
# Build with feature selection
zig build -Denable-ai=true -Denable-gpu=true

# Different test suites
zig build test              # Unit tests
zig build test-integration  # Integration tests
zig build test-all          # All tests
```

#### Build.zig Updates

If you're using Abi as a dependency:

```zig
// Before
const abi = b.dependency("abi", .{
    .target = target,
    .optimize = optimize,
});

// After - with feature flags
const abi = b.dependency("abi", .{
    .target = target,
    .optimize = optimize,
    .@"enable-ai" = true,
    .@"enable-gpu" = false,  // Disable GPU if not needed
});
```

### 2. Update Error Handling

#### Before (v0.1.0a)
```zig
pub fn loadModel(path: []const u8) !Model {
    const file = std.fs.cwd().openFile(path, .{}) catch {
        return error.ModelLoadFailed;
    };
    defer file.close();
    // ...
}
```

#### After (v0.2.0)
```zig
const abi = @import("abi");
const core = abi.core;

pub fn loadModel(path: []const u8) !Model {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        const ctx = core.ErrorContext.init(err, "Failed to open model file")
            .withLocation(core.here())
            .withContext(path);
        
        // Log detailed error
        std.log.err("{}", .{ctx});
        
        return core.errors.AIError.ModelLoadFailed;
    };
    defer file.close();
    // ...
}
```

**Key Changes:**
- Use unified error sets from `core.errors`
- Add error context with `ErrorContext`
- Use `here()` for source location tracking
- Provide user-friendly error messages

### 3. Update I/O Operations

#### Before (v0.1.0a)
```zig
pub fn processData(data: []const u8) !void {
    std.debug.print("Processing {d} bytes\n", .{data.len});
    // ... processing logic
    std.debug.print("Complete\n", .{});
}
```

#### After (v0.2.0)
```zig
const abi = @import("abi");
const core = abi.core;

pub fn processData(
    writer: core.Writer,
    data: []const u8,
) !void {
    try writer.print("Processing {d} bytes\n", .{data.len});
    // ... processing logic
    try writer.print("Complete\n", .{});
}

// Usage
const writer = core.Writer.stdout();
try processData(writer, data);
```

**Key Changes:**
- Inject `Writer` parameter
- Use `writer.print()` instead of `std.debug.print()`
- Makes code testable and composable

#### Testing the New I/O

```zig
test "processData outputs correct message" {
    var test_writer = core.TestWriter.init(testing.allocator);
    defer test_writer.deinit();
    
    try processData(test_writer.writer(), "test");
    
    try testing.expectEqualStrings(
        "Processing 4 bytes\nComplete\n",
        test_writer.getWritten(),
    );
}
```

### 4. Update Module Imports

#### Before (v0.1.0a)
```zig
const abi = @import("abi");
const utils = @import("shared/utils/mod.zig");
const logging = @import("shared/logging/mod.zig");
```

#### After (v0.2.0)
```zig
const abi = @import("abi");
const core = abi.core;      // Core infrastructure
const utils = core.utils;    // Utilities now in core
const io = core.io;          // I/O abstractions

// Feature modules (unchanged)
const ai = abi.ai;
const database = abi.database;
const gpu = abi.gpu;
```

### 5. Update Framework Initialization

#### Before (v0.1.0a)
```zig
var framework = try abi.init(allocator, .{});
defer abi.shutdown(&framework);
```

#### After (v0.2.0) - No Change!
```zig
// Same API, but now with better error handling and diagnostics
var framework = try abi.init(allocator, .{}) catch |err| {
    const ctx = core.ErrorContext.init(err, "Framework init failed");
    std.log.err("{}", .{ctx});
    return err;
};
defer abi.shutdown(&framework);
```

### 6. Update Diagnostic Collection

#### Before (v0.1.0a)
```zig
pub fn validate(config: Config) !void {
    if (config.name.len == 0) {
        std.log.err("Invalid config: name is empty", .{});
        return error.InvalidConfiguration;
    }
    // ...
}
```

#### After (v0.2.0)
```zig
const abi = @import("abi");
const core = abi.core;

pub fn validate(
    config: Config,
    diagnostics: *core.DiagnosticCollector,
) !void {
    if (config.name.len == 0) {
        try diagnostics.add(
            core.Diagnostic.init(.err, "Configuration name cannot be empty")
                .withLocation(core.here())
                .withContext("Provide a valid name in config.name")
        );
        return core.errors.FrameworkError.InvalidConfiguration;
    }
    // ...
}

// Usage
var diagnostics = core.DiagnosticCollector.init(allocator);
defer diagnostics.deinit();

validate(config, &diagnostics) catch |err| {
    try diagnostics.emit(core.Writer.stderr());
    return err;
};
```

### 7. Update Custom Features

If you've created custom features, update the interface:

#### Before (v0.1.0a)
```zig
pub const MyFeature = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) !MyFeature {
        return .{ .allocator = allocator };
    }
    
    pub fn process(self: *MyFeature, input: []const u8) ![]const u8 {
        std.debug.print("Processing: {s}\n", .{input});
        return try self.allocator.dupe(u8, input);
    }
};
```

#### After (v0.2.0)
```zig
const abi = @import("abi");
const core = abi.core;

pub const MyFeature = struct {
    allocator: Allocator,
    output: core.OutputContext,
    
    pub fn init(
        allocator: Allocator,
        output: core.OutputContext,
    ) !MyFeature {
        return .{
            .allocator = allocator,
            .output = output,
        };
    }
    
    pub fn process(
        self: *MyFeature,
        input: []const u8,
    ) ![]const u8 {
        try self.output.stdout.print("Processing: {s}\n", .{input});
        return try self.allocator.dupe(u8, input);
    }
};
```

### 8. Update Tests

#### Test Structure Changes

Move tests to appropriate directories:

```bash
# Before
src/features/ai/tests/agent_test.zig

# After
tests/unit/features/ai/agent_test.zig  # Unit tests
tests/integration/ai_pipeline_test.zig # Integration tests
```

#### Test Utilities

```zig
// Before
test "feature works" {
    const allocator = testing.allocator;
    var feature = try Feature.init(allocator);
    defer feature.deinit();
    
    const result = try feature.process("test");
    try testing.expect(result.len > 0);
}

// After - with I/O testing
test "feature works" {
    const allocator = testing.allocator;
    
    var test_writer = core.TestWriter.init(allocator);
    defer test_writer.deinit();
    
    const output = core.OutputContext{
        .stdout = test_writer.writer(),
        .stderr = core.Writer.null(),
    };
    
    var feature = try Feature.init(allocator, output);
    defer feature.deinit();
    
    const result = try feature.process("test");
    try testing.expect(result.len > 0);
    
    // Verify output
    const written = test_writer.getWritten();
    try testing.expect(std.mem.indexOf(u8, written, "Processing") != null);
}
```

## Common Patterns

### Pattern 1: Error Handling with Context

```zig
fn operation(allocator: Allocator) !Result {
    const data = loadData(allocator) catch |err| {
        return core.ErrorContext.init(err, "Data load failed")
            .withLocation(core.here())
            .withContext("Check file permissions");
    };
    return processData(data);
}
```

### Pattern 2: Testable I/O

```zig
pub fn MyComponent = struct {
    output: core.OutputContext,
    
    pub fn run(self: *MyComponent) !void {
        try self.output.stdout.print("Running...\n", .{});
    }
};

test "component outputs correctly" {
    var test_writer = core.TestWriter.init(testing.allocator);
    defer test_writer.deinit();
    
    var component = MyComponent{
        .output = .{
            .stdout = test_writer.writer(),
            .stderr = core.Writer.null(),
        },
    };
    
    try component.run();
    try testing.expectEqualStrings("Running...\n", test_writer.getWritten());
}
```

### Pattern 3: Diagnostic Collection

```zig
pub fn validateAll(
    items: []Item,
    diagnostics: *core.DiagnosticCollector,
) !void {
    for (items, 0..) |item, i| {
        if (!item.isValid()) {
            try diagnostics.add(
                core.Diagnostic.init(.warning, "Invalid item")
                    .withContext(std.fmt.allocPrint(
                        diagnostics.diagnostics.allocator,
                        "Item {d} failed validation",
                        .{i},
                    ) catch unreachable)
            );
        }
    }
    
    if (diagnostics.hasErrors()) {
        return error.ValidationFailed;
    }
}
```

## Compatibility Layer

For gradual migration, we provide compatibility shims:

### Using Compatibility Mode

```zig
const abi = @import("abi");
const compat = @import("abi").compat;

// Old-style function that uses stdout directly
pub fn legacyFunction() void {
    compat.print("This works like std.debug.print\n", .{});
}

// Gradually migrate to new style
pub fn newFunction(writer: abi.core.Writer) !void {
    try writer.print("Using injected writer\n", .{});
}
```

## Checklist

Use this checklist to track your migration:

- [ ] Update build configuration with feature flags
- [ ] Replace ad-hoc errors with unified error sets
- [ ] Add `Writer` parameters to output functions
- [ ] Update module imports to use `core` namespace
- [ ] Add error context and diagnostics
- [ ] Move tests to new structure
- [ ] Update custom features with `OutputContext`
- [ ] Test with new `TestWriter` utilities
- [ ] Update documentation and examples
- [ ] Remove compatibility shims (after full migration)

## Getting Help

If you encounter issues during migration:

1. **Check Examples**: See [examples/](../examples/) for updated code
2. **Read Docs**: Consult [ARCHITECTURE.md](ARCHITECTURE.md) and [REDESIGN_SUMMARY.md](REDESIGN_SUMMARY.md)
3. **Open Issue**: Report problems on [GitHub Issues](https://github.com/donaldfilimon/abi/issues)
4. **Ask Community**: Use [GitHub Discussions](https://github.com/donaldfilimon/abi/discussions)

## Timeline

Recommended migration timeline:

1. **Week 1**: Update build configuration and dependencies
2. **Week 2**: Migrate error handling
3. **Week 3**: Update I/O operations
4. **Week 4**: Reorganize tests
5. **Week 5**: Remove compatibility shims, final testing

## Conclusion

The v0.2.0 redesign brings significant improvements in:

- **Testability**: Through dependency injection
- **Error Handling**: With rich context and diagnostics
- **Modularity**: Via feature flags and better organization
- **Maintainability**: Through clearer architecture

While migration requires some effort, the result is more robust, testable, and maintainable code.

---

*For more information, see the [REDESIGN_SUMMARY.md](REDESIGN_SUMMARY.md)*
