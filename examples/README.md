# ABI Framework Examples

This directory contains example programs demonstrating various features of the ABI framework.

## Examples

### hello.zig

Basic framework initialization and version check.

**Run:**

```bash
zig run examples/hello.zig
```

### database.zig

Vector database operations including insert, search, and statistics.

**Run:**

```bash
zig run examples/database.zig
```

### agent.zig

AI agent usage with conversation processing.

**Run:**

```bash
zig run examples/agent.zig -Denable-ai=true
```

### compute.zig

Compute engine task execution and result handling.

**Run:**

```bash
zig run examples/compute.zig
```

### gpu.zig

GPU acceleration and SIMD operations.

**Run:**

```bash
zig run examples/gpu.zig -Denable-gpu=true
```

### network.zig

Network cluster setup and node management.

**Run:**

```bash
zig run examples/network.zig -Denable-network=true
```

## Building Examples

All examples are integrated into the main build system:

```bash
# Build all examples
zig build examples

# Run a specific example
zig build run-hello
zig build run-database
zig build run-compute
zig build run-gpu
zig build run-network
```

## Running Benchmarks

The comprehensive benchmark suite tests all framework features:

```bash
# Run all benchmarks
zig build benchmarks
```

## Learning Path

1. **Start with `hello.zig`** - Learn basic framework initialization
2. **Try `database.zig`** - Understand vector storage and search
3. **Explore `compute.zig`** - Learn about task execution
4. **Check `agent.zig`** - See AI integration
5. **Review `gpu.zig`** - Understand GPU acceleration
6. **Study `network.zig`** - Learn distributed computing

## Common Patterns

All examples follow these patterns:

1. **Allocator Setup:**

   ```zig
   var gpa = std.heap.GeneralPurposeAllocator(.{}){};
   defer _ = gpa.deinit();
   const allocator = gpa.allocator();
   ```

2. **Framework Initialization:**

   ```zig
   var framework = try abi.init(allocator, abi.FrameworkOptions{});
   defer abi.shutdown(&framework);
   ```

3. **Error Handling:**

   ```zig
   pub fn main() !void {
       try someOperation();
       return;
   }
   ```

4. **Cleanup with defer:**
   ```zig
   const data = try allocateData();
   defer allocator.free(data);
   ```

## Need Help?

See the [Documentation Index](docs/intro.md) for comprehensive guides, or check API_REFERENCE.md for detailed API information.
