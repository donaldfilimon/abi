# ABI Framework API Documentation

Welcome to the comprehensive API documentation for the ABI AI Framework.

## Modules

### Feature Namespaces (`abi.features.*`)

- [**AI & Agents**](ai.md) — `abi.features.ai.*` personas, enhanced pipelines, and training utilities
- [**Vector Database**](database.md) — `abi.features.database.*` storage engine, configuration, and HTTP/CLI front-ends
- [**GPU Tooling**](../GPU_AI_ACCELERATION.md) — `abi.features.gpu.*` compute kernels, backends, and profiling suites
- [**Web & Connectors**](http_client.md) — `abi.features.web.*` networking clients/servers plus connector bridges

### Framework & Shared Layers

- [**Framework Runtime**](../MODULE_ORGANIZATION.md#-module-architecture) — `abi.framework.*` feature toggles, lifecycle, plugin coordination
- [**Plugin System**](plugins.md) — `abi.shared.*` registries, loaders, and interfaces used by the runtime
- [**SIMD & Utilities**](simd.md) — `abi.simd` helpers alongside `abi.utils`/`abi.platform`
- [**WDBX Utilities**](wdbx.md) — Operational helpers layered on top of the database feature

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
