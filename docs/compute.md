# Compute Engine

> **Developer Guide**: See [AGENTS.md](../AGENTS.md) for compute coding patterns and [CLAUDE.md](../CLAUDE.md) for engine internals.
> **GPU Offloading**: See [GPU Guide](gpu.md) for GPU-accelerated compute.

The **Compute Engine** (`abi.compute`) is heart of ABI's performance capabilities. It uses a work-stealing scheduler to efficiently execute concurrent tasks.

## Concepts

### Engine

The `Engine` manages the thread pool and task queues.

```zig
var engine = try abi.compute.createDefaultEngine(allocator);
defer engine.deinit();
```

### Workloads

A `Workload` (also called a task) is a unit of execution. It can be a simple closure or a complex struct implementing the Workload VTable.

```zig
fn myTask(_: std.mem.Allocator) !u32 {
    return 42;
}

// Run a task (submit and wait for result)
const result = try abi.compute.runTask(&engine, u32, myTask, 1000);

// Alternative: use runWorkload (alias for runTask)
const result2 = try abi.compute.runWorkload(&engine, u32, myTask, 1000);
```

### Submitting and Retrieving Results

You can also submit tasks and retrieve results separately:

```zig
// Submit task for execution
const task_id = try abi.compute.submitTask(&engine, u32, myTask);

// Wait for result with timeout
const result = try abi.compute.waitForResult(&engine, u32, task_id, 1000);
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

On supported platforms (Linux, Windows), the engine can pin worker threads to specific cores using `abi.compute.AffinityMask`.

### NUMA Awareness

The engine detects NUMA topology to optimize memory allocation and thread placement, reducing cross-node traffic.

```zig
var engine = try abi.compute.createEngine(allocator, .{
    .numa_enabled = true,
    .cpu_affinity_enabled = true,
});
```

---

## Concurrency Primitives

The compute module provides lock-free data structures:

| Primitive | Description |
|-----------|-------------|
| `WorkStealingQueue` | LIFO for owner, FIFO for thieves |
| `LockFreeQueue` | Atomic CAS-based queue |
| `LockFreeStack` | Atomic CAS-based stack |
| `PriorityQueue` | Lock-free priority queue |
| `ShardedMap` | Reduces contention via sharding |

---

## CLI Commands

```bash
# Show system and compute info
zig build run -- system-info

# Run SIMD performance demo
zig build run -- simd

# Run benchmarks
zig build benchmarks
```

---

## See Also

- [GPU Acceleration](gpu.md) - GPU workload offloading
- [Network](network.md) - Distributed task execution
- [Monitoring](monitoring.md) - Engine metrics and profiling
- [Troubleshooting](troubleshooting.md) - Timeout and performance issues
*See [../TODO.md](../TODO.md) and [../ROADMAP.md](../ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
[Main Workspace](../MAIN_WORKSPACE.md)
