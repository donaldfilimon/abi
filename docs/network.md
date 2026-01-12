# Network & Distributed Compute

> [!NOTE]
> **Status**: Experimental. Requires `-Denable-network=true`.

The **Network** module (`abi.network`) enables ABI nodes to discover each other and distribute computational tasks across a cluster.

## Architecture

- **Registry**: A peer-to-peer registry that tracks active nodes and their capabilities (CPU cores, GPU availability).
- **Task Serialization**: A binary serialization format to transmit workloads over the wire.
- **Circuit Breaker**: Automatic failure detection and recovery for remote nodes.
- **Service Discovery**: Automatic node discovery on local networks or configured subnets.

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
    std.debug.print("Node: {s} - CPUs: {d}, GPU: {}\n", .{
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

- [Compute Engine](compute.md) - Local task execution
- [Monitoring](monitoring.md) - Network metrics and alerting
- [Framework](framework.md) - Configuration options
- [Troubleshooting](troubleshooting.md) - Connection issues
