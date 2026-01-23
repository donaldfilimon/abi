---
title: "network"
tags: []
---
# Network & Distributed Compute
> **Codebase Status:** Synced with repository as of 2026-01-22.

<p align="center">
  <img src="https://img.shields.io/badge/Module-Network-blue?style=for-the-badge&logo=cloudflare&logoColor=white" alt="Network Module"/>
  <img src="https://img.shields.io/badge/Status-Ready-success?style=for-the-badge" alt="Ready"/>
  <img src="https://img.shields.io/badge/Feature_Flag-enable--network-yellow?style=for-the-badge" alt="Feature Flag"/>
</p>

<p align="center">
  <a href="#node-discovery">Discovery</a> •
  <a href="#remote-execution">Remote Exec</a> •
  <a href="#circuit-breaker">Circuit Breaker</a> •
  <a href="#cli-commands">CLI</a>
</p>

---

> **Build Flag**: Requires `-Denable-network=true`

The **Network** module (`abi.network`) enables ABI nodes to discover each other and distribute computational tasks across a cluster.

## Feature Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Node Discovery** | P2P node registry | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Task Serialization** | Binary wire format | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Circuit Breaker** | Failure detection/recovery | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Service Discovery** | Auto network discovery | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Raft Consensus** | Distributed consensus | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Node Discovery

Nodes broadcast their presence on the local network (or configured subnets).

```zig
const abi = @import("abi");

var registry = try abi.network.NodeRegistry.init(allocator, .{
    .port = 8081,
    .discovery_enabled = true,
    .broadcast_interval_ms = 5000,
});
defer registry.deinit();

try registry.startDiscovery();

// List discovered nodes
const nodes = try registry.getActiveNodes(allocator);
defer allocator.free(nodes);

for (nodes) |node| {
    std.debug.print("Node: {s} - CPUs: {d}, GPU: {t}\n", .{
        node.id,
        node.cpu_cores,
        node.has_gpu,
    });
}
```

## Remote Execution

Instead of running a task on the local `Engine`, you can submit it to the `Cluster`.

```zig
const abi = @import("abi");

// Serialize task
const payload = try abi.network.serializeTask(allocator, myTaskData);
defer allocator.free(payload);

// Send to best available node
const result = try cluster.submitToNode(target_node_id, payload);
defer allocator.free(result);

// Or let the cluster choose the best node
const auto_result = try cluster.submitAuto(payload);
defer allocator.free(auto_result);
```

## Task Serialization

The network module provides binary serialization for efficient task transfer:

```zig
const abi = @import("abi");

// Serialize a work item
const serialized = try abi.network.serializeTask(
    allocator,
    &work_item,
    "matrix_multiply",
    user_data_bytes,
);
defer allocator.free(serialized);

// Deserialize on remote node
const task = try abi.network.deserializeTask(allocator, serialized);
defer {
    allocator.free(task.payload_type);
    allocator.free(task.user_data);
}
```

## Error Handling

Network operations can fail due to connectivity issues:

```zig
const result = cluster.submitToNode(node_id, payload) catch |err| {
    switch (err) {
        error.NodeUnreachable => {
            // Node is down, try another
            return cluster.submitAuto(payload);
        },
        error.Timeout => {
            // Request timed out
            std.debug.print("Request timed out\n", .{});
            return err;
        },
        error.NetworkDisabled => {
            // Feature not enabled at build time
            std.debug.print("Network feature disabled\n", .{});
            return err;
        },
        else => return err,
    }
};
```

## Circuit Breaker

The circuit breaker prevents cascading failures:

```zig
var breaker = try abi.network.CircuitBreaker.init(allocator, .{
    .failure_threshold = 5,
    .reset_timeout_ms = 30000,
    .half_open_max_calls = 3,
});
defer breaker.deinit();

// Wrap network calls
const result = try breaker.call(fn () !Response {
    return cluster.submitToNode(node_id, payload);
});
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `port` | 8081 | Network registry port |
| `discovery_enabled` | true | Enable automatic discovery |
| `broadcast_interval_ms` | 5000 | Discovery broadcast interval |
| `connection_timeout_ms` | 10000 | Connection timeout |
| `request_timeout_ms` | 30000 | Request timeout |
| `max_retries` | 3 | Maximum retry attempts |

## CLI Commands

```bash
# Network registry operations
zig build run -- network list                   # List discovered nodes
zig build run -- network register --host HOST  # Register a node
zig build run -- network status                 # Show network status
```

---

## See Also

<table>
<tr>
<td>

### Related Guides
- [Compute Engine](compute.md) — Local task execution
- [Monitoring](monitoring.md) — Network metrics and alerting
- [Framework](framework.md) — Configuration options

</td>
<td>

### Resources
- [Troubleshooting](troubleshooting.md) — Connection issues
- [API Reference](../API_REFERENCE.md) — Network API details
- [Examples](../examples/) — Network code samples

</td>
</tr>
</table>

---

<p align="center">
  <a href="database.md">← Database Guide</a> •
  <a href="docs-index.md">Documentation Index</a> •
  <a href="monitoring.md">Monitoring Guide →</a>
</p>

