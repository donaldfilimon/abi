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

AI agent usage with conversational chat interface. Demonstrates the `Agent.chat()` method for processing user input with history tracking.

**Features:**
- Agent initialization with configuration
- Using `chat()` method for conversational interface
- Proper memory management with defer

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

### discord.zig

Discord bot integration with bot info, guild listing, and gateway information.

**Prerequisites:**
- Set `DISCORD_BOT_TOKEN` environment variable with your bot token

**Run:**

```bash
zig build run-discord
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
zig build run-discord
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
7. **Check `discord.zig`** - Discord bot integration

## Common Patterns

All examples follow these Zig 0.16 best practices:

1. **Modern Main Signature (Zig 0.16):**

   ```zig
   pub fn main(init: std.process.Init) !void {
       const allocator = init.gpa;
       // ... your code
   }
   ```

2. **Framework Initialization:**

   ```zig
   var framework = try abi.init(allocator, abi.FrameworkOptions{});
   defer abi.shutdown(&framework);
   ```

3. **Error Handling:**

   ```zig
   pub fn main(init: std.process.Init) !void {
       try someOperation();
       return;
   }
   ```

4. **Cleanup with defer:**

   ```zig
   const data = try allocateData();
   defer allocator.free(data);
   ```

5. **Format Specifiers (Zig 0.16):**

   ```zig
   std.debug.print("Status: {t}\n", .{status});  // {t} for enums
   std.debug.print("Count: {d}\n", .{count});    // {d} for integers
   ```

## Need Help?

See the [Documentation Index](docs/intro.md) for comprehensive guides, or check API_REFERENCE.md for detailed API information.

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

