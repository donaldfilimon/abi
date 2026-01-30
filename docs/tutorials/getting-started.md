# Tutorial: Getting Started with ABI
> **Codebase Status:** Synced with repository as of 2026-01-30.

> **Duration:** 20 minutes | **Level:** Beginner | **Video:** [Watch](videos/01-getting-started.md)

## What You'll Learn

- Install and configure the ABI framework
- Initialize and shut down the framework
- Understand feature gating and build options
- Run your first ABI program

## Prerequisites

- Zig 0.16.x or later installed
- Basic familiarity with the command line
- Text editor of your choice

---

## Step 1: Clone and Build

First, clone the ABI repository and build the project.

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
```

**Expected output:**
```
Build completed successfully.
```

### Build Options

ABI uses feature gating to let you compile only what you need:

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI/LLM capabilities |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed computing |

**Example: Build with only AI and database:**
```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true -Denable-network=false
```

---

## Step 2: Your First Program

Create a file `hello_abi.zig` in the project root:

**Code:** `docs/tutorials/code/getting-started/01-hello-abi.zig`

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Get an allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize ABI
    std.debug.print("Initializing ABI framework...\n", .{});
    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    // Print version
    const version = abi.version();
    std.debug.print("ABI Version: {d}.{d}.{d}\n", .{
        version.major,
        version.minor,
        version.patch,
    });

    std.debug.print("ABI framework initialized successfully!\n", .{});
}
```

**Run:**
```bash
zig run hello_abi.zig
```

**Expected output:**
```
Initializing ABI framework...
ABI Version: 1.0.0
ABI framework initialized successfully!
```

### Key Patterns

| Pattern | Purpose |
|---------|---------|
| `abi.initDefault(allocator)` | Initialize all enabled features |
| `defer framework.deinit()` | Clean up resources on scope exit |
| `abi.version()` | Get semantic version info |

---

## Step 3: Feature Detection

ABI lets you check which features are enabled at runtime:

**Code:** `docs/tutorials/code/getting-started/02-feature-detection.zig`

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    std.debug.print("\n=== ABI Feature Status ===\n\n", .{});

    // Check each feature
    const features = [_]struct { name: []const u8, enabled: bool }{
        .{ .name = "AI/LLM", .enabled = framework.isEnabled(.ai) },
        .{ .name = "GPU", .enabled = framework.isEnabled(.gpu) },
        .{ .name = "Database", .enabled = framework.isEnabled(.database) },
        .{ .name = "Network", .enabled = framework.isEnabled(.network) },
        .{ .name = "Web", .enabled = framework.isEnabled(.web) },
        .{ .name = "Observability", .enabled = framework.isEnabled(.observability) },
    };

    for (features) |f| {
        const status = if (f.enabled) "[ENABLED]" else "[DISABLED]";
        std.debug.print("  {s:<12} {s}\n", .{ f.name, status });
    }

    std.debug.print("\n", .{});
}
```

**Run:**
```bash
zig run feature_check.zig
```

**Expected output:**
```
=== ABI Feature Status ===

  AI/LLM       [ENABLED]
  GPU          [ENABLED]
  Database     [ENABLED]
  Network      [ENABLED]
  Web          [ENABLED]
  Observability [ENABLED]
```

---

## Step 4: Error Handling

ABI uses Zig's error handling system. When a feature is disabled, operations return `error.*Disabled`:

**Code:** `docs/tutorials/code/getting-started/03-error-handling.zig`

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    // Try to use the database feature
    if (abi.database.openOrCreate(allocator, "test_db")) |*db| {
        defer abi.database.close(db);
        std.debug.print("Database opened successfully\n", .{});
    } else |err| {
        switch (err) {
            error.DatabaseDisabled => {
                std.debug.print("Database feature is disabled.\n", .{});
                std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
            },
            else => {
                std.debug.print("Database error: {t}\n", .{err});
            },
        }
    }
}
```

### Common Error Types

| Error | Cause | Solution |
|-------|-------|----------|
| `error.AiDisabled` | AI feature not compiled | `-Denable-ai=true` |
| `error.GpuDisabled` | GPU feature not compiled | `-Denable-gpu=true` |
| `error.DatabaseDisabled` | Database not compiled | `-Denable-database=true` |
| `error.NetworkDisabled` | Network not compiled | `-Denable-network=true` |

---

## Step 5: Using the CLI

ABI includes a powerful CLI for common operations:

```bash
# Show help
zig build run -- --help

# Check system info
zig build run -- system-info

# List GPU backends
zig build run -- gpu backends

# Explore codebase
zig build run -- explore "fn init" --level thorough
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `system-info` | Display system capabilities |
| `gpu backends` | List available GPU backends |
| `gpu devices` | Show GPU devices |
| `db stats` | Database statistics |
| `explore` | Search codebase |
| `tui` | Interactive terminal UI |

---

## Step 6: Configuration via Environment

ABI reads configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |

**Example: Setting up Ollama integration:**

```bash
export ABI_OLLAMA_HOST="http://localhost:11434"
export ABI_OLLAMA_MODEL="gpt-oss"

zig build run -- agent chat --prompt "Hello, world!"
```

---

## Practice Exercises

### Exercise 1: Custom Initialization

Create a program that:
1. Initializes ABI
2. Checks if GPU is available
3. Prints GPU device info if available
4. Falls back gracefully if not

**Hints:**
- Use `framework.isEnabled(.gpu)`
- Use `abi.gpu.listDevices()` if GPU is enabled

### Exercise 2: Build Variants

Build ABI three times with different configurations:
1. Full features (all enabled)
2. Minimal (AI only)
3. Embedded (no network, no GPU)

Compare the resulting binary sizes.

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `error: FileNotFound` | Wrong import path | Use `@import("abi")` |
| `error: OutOfMemory` | Allocator issue | Check allocator lifecycle |
| Build fails | Missing dependencies | Run `zig build` first |
| Feature disabled | Compile flag missing | Add `-Denable-X=true` |

---

## Next Steps

- [Tutorial 2: Vector Database](vector-database.md) - Store and search vectors
- [API Reference](../API_REFERENCE.md) - Complete API docs
- [AI Guide](../ai.md) - Connect to LLMs
- [GPU Guide](../gpu.md) - Accelerate compute

---

**Video Walkthrough:** [Watch the 20-minute guided tutorial](videos/01-getting-started.md)
