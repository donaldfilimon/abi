# Getting Started with Abi Framework

## Welcome! 👋

This guide will help you get up and running with the Abi Framework in under 15 minutes.

## What You'll Build

By the end of this guide, you'll have:
- ✅ A working Abi installation
- ✅ Your first AI agent application
- ✅ Understanding of core concepts
- ✅ Knowledge of how to test your code

## Prerequisites

### Required
- **Zig 0.16.0-dev** or later ([Download](https://ziglang.org/download/))
- Basic understanding of Zig syntax
- A text editor or IDE

### Recommended
- **Git** for version control
- **VS Code** with Zig extension (optional)

## Step 1: Installation

### Clone the Repository

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
```

### Verify Installation

```bash
# Check Zig version
zig version

# Build the framework
zig build

# Run tests to verify everything works
zig build test
```

You should see:
```
All tests passed!
```

## Step 2: Your First Application

### Create a New File

Create `hello_abi.zig` in your project:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Setup allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the framework
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Print framework version
    std.debug.print("Abi Framework v{s}\n", .{abi.version()});
    std.debug.print("Framework initialized successfully!\n", .{});
}
```

### Build and Run

```bash
zig build-exe hello_abi.zig --deps abi --mod abi::src/mod.zig

./hello_abi
```

**Output:**
```
Abi Framework v0.2.0
Framework initialized successfully!
```

## Step 3: Build an AI Agent

### Simple Agent Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Create an AI agent
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(allocator, .{
        .name = "HelloAgent",
    });
    defer agent.deinit();

    // Process a query
    const query = "What is the meaning of life?";
    std.debug.print("Query: {s}\n", .{query});

    const response = try agent.process(query, allocator);
    defer allocator.free(@constCast(response));

    std.debug.print("Response: {s}\n", .{response});
}
```

### Understanding the Code

1. **Memory Management**: We use `GeneralPurposeAllocator` for memory allocation
2. **Framework Init**: `abi.init()` sets up the framework
3. **Agent Creation**: `Agent.init()` creates an AI agent
4. **Processing**: `agent.process()` handles queries
5. **Cleanup**: `defer` ensures proper resource cleanup

## Step 4: Working with the Database

### Vector Database Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Create a vector (embedding)
    const vector_size = 128;
    var embedding = try allocator.alloc(f32, vector_size);
    defer allocator.free(embedding);

    // Initialize with some values
    for (embedding, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.01;
    }

    std.debug.print("Created vector of size {d}\n", .{embedding.len});
    std.debug.print("First 5 values: ", .{});
    for (embedding[0..5]) |v| {
        std.debug.print("{d:.3} ", .{v});
    }
    std.debug.print("\n", .{});
}
```

## Step 5: Using the CLI

The Abi CLI provides quick access to framework features:

### Feature Management

```bash
# List available features
./zig-out/bin/abi features list

# Check feature status
./zig-out/bin/abi features status
```

### Agent Operations

```bash
# Run an agent
./zig-out/bin/abi agent run --name "MyAgent"

# List available agents
./zig-out/bin/abi agent list
```

### Database Operations

```bash
# Create a database
./zig-out/bin/abi db create --name my_vectors

# Query the database
./zig-out/bin/abi db query --vector "[0.1, 0.2, ...]"
```

## Step 6: Writing Tests

### Testing Your Code

Create `hello_test.zig`:

```zig
const std = @import("std");
const abi = @import("abi");
const testing = std.testing;

test "framework initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    try testing.expect(framework.state == .initialized);
}

test "agent creation and processing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(allocator, .{ .name = "TestAgent" });
    defer agent.deinit();

    const response = try agent.process("test", allocator);
    defer allocator.free(@constCast(response));

    try testing.expect(response.len > 0);
}
```

### Run Tests

```bash
zig test hello_test.zig --deps abi --mod abi::src/mod.zig
```

## Step 7: Using Advanced Features

### Error Handling with Context

```zig
const std = @import("std");
const abi = @import("abi");
const core = abi.core;

pub fn loadConfig(path: []const u8) !Config {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        const ctx = core.ErrorContext.init(err, "Failed to load config")
            .withLocation(core.here())
            .withContext(path);
        
        std.log.err("{}", .{ctx});
        return core.errors.FrameworkError.InvalidConfiguration;
    };
    defer file.close();

    // ... parse config
}
```

### Testable I/O

```zig
const std = @import("std");
const abi = @import("abi");
const core = abi.core;

pub fn greet(writer: core.Writer, name: []const u8) !void {
    try writer.print("Hello, {s}!\n", .{name});
}

test "greeting outputs correct message" {
    var test_writer = core.TestWriter.init(std.testing.allocator);
    defer test_writer.deinit();

    try greet(test_writer.writer(), "World");

    try std.testing.expectEqualStrings(
        "Hello, World!\n",
        test_writer.getWritten(),
    );
}
```

### Diagnostics Collection

```zig
const std = @import("std");
const abi = @import("abi");
const core = abi.core;

pub fn validate(config: Config) !void {
    var diagnostics = core.DiagnosticCollector.init(
        std.heap.page_allocator
    );
    defer diagnostics.deinit();

    if (config.name.len == 0) {
        try diagnostics.add(
            core.Diagnostic.init(.err, "Name cannot be empty")
                .withLocation(core.here())
        );
    }

    if (diagnostics.hasErrors()) {
        try diagnostics.emit(core.Writer.stderr());
        return error.ValidationFailed;
    }
}
```

## Step 8: Build Configuration

### Feature Flags

Control which features are compiled:

```bash
# Build with specific features
zig build -Denable-ai=true -Denable-gpu=false

# Build with GPU backend
zig build -Denable-gpu=true -Dgpu-vulkan=true

# Build optimized
zig build -Doptimize=ReleaseFast
```

### In build.zig

```zig
const abi = b.dependency("abi", .{
    .target = target,
    .optimize = optimize,
    .@"enable-ai" = true,
    .@"enable-gpu" = false,
    .@"enable-database" = true,
});
```

## Common Patterns

### Pattern 1: Resource Management

```zig
// Always use defer for cleanup
var resource = try Resource.init(allocator);
defer resource.deinit();

// For arrays
var items = try allocator.alloc(Item, count);
defer allocator.free(items);
```

### Pattern 2: Error Handling

```zig
// Provide context on errors
doSomething() catch |err| {
    const ctx = core.ErrorContext.init(err, "Operation failed")
        .withLocation(core.here());
    std.log.err("{}", .{ctx});
    return err;
};
```

### Pattern 3: Testable Code

```zig
// Inject dependencies
pub fn process(
    allocator: Allocator,
    writer: core.Writer,
    data: []const u8,
) !void {
    try writer.print("Processing...\n", .{});
    // ... logic
}

// Easy to test
test "process works" {
    var test_writer = core.TestWriter.init(testing.allocator);
    defer test_writer.deinit();
    
    try process(testing.allocator, test_writer.writer(), "test");
    
    try testing.expectEqualStrings(
        "Processing...\n",
        test_writer.getWritten(),
    );
}
```

## Next Steps

Now that you have the basics, explore:

1. **[Architecture Guide](../ARCHITECTURE.md)** - Understand the framework design
2. **[Examples](../../examples/)** - See practical applications
3. **[API Reference](../api/)** - Dive into the API
4. **[Migration Guide](../MIGRATION_GUIDE.md)** - Upgrade from v0.1.0a

## Troubleshooting

### Build Errors

**Problem**: Zig version mismatch
```
error: Zig version mismatch
```

**Solution**: Ensure you're using Zig 0.16.0-dev or later
```bash
zig version
```

### Memory Leaks

**Problem**: Memory leaks detected
```
[gpa] (err): memory leak
```

**Solution**: Ensure all allocations have corresponding `defer free()`
```zig
var data = try allocator.alloc(u8, 100);
defer allocator.free(data);  // Don't forget this!
```

### Feature Not Available

**Problem**: Feature not found at runtime
```
error: FeatureNotAvailable
```

**Solution**: Enable the feature at build time
```bash
zig build -Denable-ai=true
```

## Getting Help

- 📖 **Documentation**: [docs/](../)
- 🐛 **Issues**: [GitHub Issues](https://github.com/donaldfilimon/abi/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/donaldfilimon/abi/discussions)
- 📝 **Examples**: [examples/](../../examples/)

## Checklist

Track your progress:

- [ ] Installed Zig 0.16+
- [ ] Cloned and built Abi
- [ ] Created "Hello, Abi" app
- [ ] Built an AI agent
- [ ] Worked with database features
- [ ] Used the CLI
- [ ] Written tests
- [ ] Explored advanced features
- [ ] Configured build with feature flags

**Congratulations!** 🎉 You're now ready to build with Abi!

---

*Happy coding with Abi Framework!*
