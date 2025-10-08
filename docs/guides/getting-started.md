# Getting Started with ABI Framework

This guide will help you get started with the ABI Framework quickly.

## Prerequisites

- **Zig** `0.16.0-dev.463+f624191f9` or later
- A C++ compiler for Zig's build dependencies

## Installation

### Clone the Repository

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
```

### Build the Framework

```bash
zig build
```

This will create the CLI executable at `zig-out/bin/abi`.

### Run Tests

```bash
zig build test
```

## Quick Start

### Using the CLI

The ABI Framework includes a comprehensive CLI for managing the framework:

```bash
# Show help
./zig-out/bin/abi help

# Show version
./zig-out/bin/abi version

# List available features
./zig-out/bin/abi features list

# Enable specific features
./zig-out/bin/abi features enable gpu monitoring

# Show framework status
./zig-out/bin/abi framework status
```

### Using the Library

Here's a simple example of using the ABI Framework in your own code:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Create framework with default configuration
    var framework = try abi.createDefaultFramework(gpa.allocator());
    defer framework.deinit();

    // Check which features are enabled
    std.log.info("AI feature enabled: {}", .{framework.isFeatureEnabled(.ai)});
    std.log.info("GPU feature enabled: {}", .{framework.isFeatureEnabled(.gpu)});

    // Start the framework
    try framework.start();
    defer framework.stop();

    // Get runtime statistics
    const stats = framework.getStats();
    std.log.info("Framework running with {} components", .{stats.total_components});
}
```

## Examples

The framework includes several examples to help you get started:

### Basic Usage

```bash
zig run examples/basic-usage.zig
```

This example demonstrates:
- Framework initialization
- Feature management
- Component registration
- Memory management
- Runtime operations

### Advanced Features

```bash
zig run examples/advanced-features.zig
```

This example shows:
- Custom configuration
- Advanced component lifecycle
- Error handling patterns
- Memory tracking
- Performance monitoring

## Next Steps

1. **Explore the Features**: Try enabling different features and see what's available
2. **Read the Examples**: Study the example code to understand common patterns
3. **Check the API Reference**: Browse the generated documentation for detailed API information
4. **Join the Community**: Check out the contributing guidelines and join discussions

## Common Tasks

### Enable GPU Acceleration

```bash
./zig-out/bin/abi features enable gpu
```

### Configure Memory Limits

```zig
var framework = try abi.createFramework(allocator, .{
    .memory_limit_mb = 512,
    .enable_profiling = true,
});
```

### Register Custom Components

```zig
const my_component = abi.framework.Component{
    .name = "my_component",
    .version = "1.0.0",
    .init_fn = myInit,
    .deinit_fn = myDeinit,
    .update_fn = myUpdate,
};
try framework.registerComponent(my_component);
```

## Getting Help

- Check the [API Reference](../api/) for detailed documentation
- Look at the [examples](../examples/) for working code
- Read the [development guide](development.md) for contributing
- Open an issue on GitHub for bugs or feature requests