# ABI Framework API Documentation

Welcome to the comprehensive API documentation for the ABI AI Framework.

## Modules

### Core Modules

- [**Database API**](database.md) - Vector database operations
- [**AI/ML API**](ai.md) - AI agents and neural networks
- [**SIMD API**](simd.md) - SIMD-accelerated operations

### Infrastructure

- [**HTTP Client**](http_client.md) - Enhanced HTTP client
- [**Plugin System**](plugins.md) - Extensibility framework
- [**WDBX Utilities**](wdbx.md) - Database management tools

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var framework = try abi.init(std.heap.page_allocator, .{});
    defer framework.deinit();
    // Your code here
}
```

## Performance Guarantees

- **Throughput**: 2,777+ ops/sec
- **Latency**: <1ms average
- **Success Rate**: 99.98%
- **Memory**: Zero leaks

## License

Apache License 2.0 - see LICENSE file for details.
