# ABI Framework Documentation

Welcome to the official documentation for the **ABI Framework**.

## Quick Links

| Resource | Description |
|----------|-------------|
| [Introduction](intro.md) | Architecture overview and philosophy |
| [API Reference](../API_REFERENCE.md) | Public API documentation |
| [Quickstart](../QUICKSTART.md) | Get started quickly |
| [Migration Guide](migration/zig-0.16-migration.md) | Zig 0.16 migration notes |

## Feature Documentation

| Feature | Description |
|---------|-------------|
| [AI & Agents](ai.md) | AI features, LLM, embeddings, connectors |
| [Compute Engine](compute.md) | Work-stealing scheduler, concurrency |
| [Database](database.md) | WDBX vector database, HNSW indexing |
| [GPU Acceleration](gpu.md) | CUDA, Vulkan, Metal, WebGPU backends |
| [Framework](framework.md) | Lifecycle management, configuration |
| [Monitoring](monitoring.md) | Logging, metrics, tracing |
| [Network](network.md) | Distributed compute, node discovery |
| [Explore](explore.md) | Codebase exploration and search |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/abi.git
cd abi

# Build
zig build

# Run tests
zig build test
```

## Getting Started

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

For a comprehensive guide, see the [Introduction](intro.md).
