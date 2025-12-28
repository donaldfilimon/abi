# Compute Engine

The **Compute Engine** (`abi.compute`) is the heart of ABI's performance capabilities. It uses a work-stealing scheduler to efficiently execute concurrent tasks.

## Concepts

### Engine

The `Engine` manages the thread pool and task queues.

```zig
var engine = try abi.compute.createDefaultEngine(allocator);
defer engine.deinit();
```

### Workloads

A `Workload` is a unit of execution. It can be a simple closure or a complex struct implementing the Workload VTable.

```zig
fn myTask(_: std.mem.Allocator) !u32 {
    return 42;
}

// Run a task
const result = try abi.compute.runTask(&engine, u32, myTask, 1000);
```

## Timeout Semantics

When retrieving results or waiting for tasks, proper timeout handling is crucial. As of **v0.2.1**, the behavior is:

- **`timeout_ms=0`**: Non-blocking check. Immediately returns `EngineError.Timeout` if the result is not ready.
- **`timeout_ms>0`**: Blocks the calling thread for up to `timeout_ms` milliseconds. Returns `EngineError.Timeout` if time expires.
- **`timeout_ms=null`**: Waits indefinitely (blocking) until the task completes.

> [!WARNING]
> **Breaking Change**: behavior for `0` changed in v0.2.1. Previously it checked once and returned `ResultNotFound`. Now it returns `Timeout` explicitly.

## Advanced Usage

### CPU Affinity

On supported platforms (Linux, Windows), the engine can pind worker threads to specific cores using `abi.compute.AffinityMask`.

### NUMA Awareness

The engine detects NUMA topology to optimize memory allocation and thread placement, reducing cross-node traffic.
