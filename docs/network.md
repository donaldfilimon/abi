# Network & Distributed Compute

> [!NOTE]
> **Status**: Experimental. Requires `-Denable-network=true`.

The **Network** module (`abi.network`) enables ABI nodes to discover each other and distrubute computational tasks across a cluster.

## Architecture

- **Registry**: A peer-to-peer registry that tracks active nodes and their capabilities (CPU cores, GPU availability).
- **Task Serialization**: A binary serialization format to transmit workloads over the wire.

## Node Discovery

Nodes broadcast their presence on the local network (or configured subnets).

```zig
var registry = try abi.network.NodeRegistry.init(allocator, port);
try registry.startDiscovery();
```

## Remote Execution

Instead of running a task on the local `Engine`, you can submit it to the `Cluster`.

```zig
// Serialize task
const payload = try abi.network.serializeTask(allocator, myTaskData);

// Send to best available node
try cluster.submitToNode(target_node_id, payload);
```
