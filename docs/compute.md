---
title: "compute"
tags: []
---
# Compute Engine
> **Codebase Status:** Synced with repository as of 2026-01-22.

<p align="center">
  <img src="https://img.shields.io/badge/Module-Compute-red?style=for-the-badge&logo=speedtest&logoColor=white" alt="Compute Module"/>
  <img src="https://img.shields.io/badge/Status-Ready-success?style=for-the-badge" alt="Ready"/>
  <img src="https://img.shields.io/badge/Scheduler-Work_Stealing-blue?style=for-the-badge" alt="Work Stealing"/>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#advanced-features">Advanced</a> •
  <a href="#api-reference">API Reference</a>
</p>

---

> **Note**: GPU functionality is a separate top-level module - see [GPU Guide](gpu.md).

Work-stealing scheduler for efficient concurrent task execution.

## Feature Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Work-Stealing** | LIFO owner, FIFO thieves | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **NUMA-Aware** | NUMA topology support | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Lock-Free** | CAS-based data structures | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Futures** | Async result handling | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Cancellation** | Cooperative cancellation | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Task Groups** | Hierarchical tasks | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Quick Start

```zig
const abi = @import("abi");

// Create engine
var engine = try abi.runtime.createDefaultEngine(allocator);
defer engine.deinit();

// Run a task
fn myTask(_: std.mem.Allocator) !u32 {
    return 42;
}

const result = try abi.runtime.runTask(&engine, u32, myTask, 1000);

// Or submit/wait separately
const task_id = try abi.runtime.submitTask(&engine, u32, myTask);
const result2 = try abi.runtime.waitForResult(&engine, u32, task_id, 1000);
```

## Architecture

```text
src/runtime/           # Engine, scheduler, futures, cancellation
  ├── engine/          # Task engine implementation
  ├── scheduling/      # Future, CancellationToken, TaskGroup
  ├── concurrency/     # Lock-free primitives, priority queue
  └── memory/          # Memory utilities
src/gpu/               # GPU acceleration (separate module)
```

## Timeout Semantics

- `timeout_ms=0`: Non-blocking, returns `EngineError.Timeout` immediately if not ready
- `timeout_ms>0`: Blocks up to `timeout_ms` milliseconds
- `timeout_ms=null`: Waits indefinitely

> **Breaking Change (v0.2.1)**: `timeout_ms=0` now returns `Timeout` instead of checking once.

## Advanced Features

### NUMA & CPU Affinity

```zig
var engine = try abi.runtime.createEngine(allocator, .{
    .numa_enabled = true,
    .cpu_affinity_enabled = true,
});
```

### Concurrency Primitives

- `WorkStealingQueue` - LIFO owner, FIFO thieves
- `LockFreeQueue/Stack` - Atomic CAS-based
- `PriorityQueue` - Lock-free priority scheduling
- `ShardedMap` - Contention-reducing sharding

### Futures & Cancellation

```zig
const Future = abi.runtime.Future;
const CancellationToken = abi.runtime.CancellationToken;
const TaskGroup = abi.runtime.TaskGroup;
```

---

## API Reference

**Source:** `src/runtime/mod.zig`

### Engine Functions

| Function | Description |
|----------|-------------|
| `createEngine(allocator, config)` | Create compute engine with custom config |
| `submitTask(engine, ResultType, task)` | Submit task for execution, returns task ID |
| `waitForResult(engine, ResultType, id, timeout_ms)` | Wait for task result by ID |
| `runTask(engine, ResultType, task, timeout_ms)` | Submit and wait for result in one call |
| `runWorkload(engine, ResultType, work, timeout_ms)` | Alias for `runTask()` |

### Core Types

- `DistributedComputeEngine` - Main engine instance
- `EngineConfig` - Engine configuration
- `EngineError` - Engine error types
- `TaskId` - Task identifier

### Futures & Promises

- `Future(ResultType)` - Async result with `.then()`, `.catch()`
- `Promise(ResultType)` - Future producer
- `all(...)` - Wait for all futures
- `race(...)` - Wait for first future
- `delay(ms)` - Delayed future

### Cancellation

- `CancellationToken` - Cooperative cancellation
- `CancellationSource` - Token source
- `LinkedCancellation` - Linked cancellation tokens

### Task Groups

- `TaskGroup` - Hierarchical task organization
- `TaskGroupConfig` - Group configuration
- `parallelForEach()` - Parallel iteration

### Workloads

- `WorkloadVTable` - Workload interface
- `GPUWorkloadVTable` - GPU workload interface
- `WorkItem` - Work item type
- `matMul`, `dense`, `relu` - Built-in workload operations

---

## See Also

<table>
<tr>
<td>

### Related Guides
- [GPU Guide](gpu.md) — GPU acceleration
- [Network](network.md) — Distributed compute
- [Monitoring](monitoring.md) — Metrics and profiling

</td>
<td>

### Resources
- [Framework](framework.md) — Framework configuration
- [API Reference](../API_REFERENCE.md) — Compute API details
- [Examples](../examples/) — Code samples

</td>
</tr>
</table>

---

<p align="center">
  <a href="monitoring.md">← Monitoring Guide</a> •
  <a href="docs-index.md">Documentation Index</a> •
  <a href="gpu.md">GPU Guide →</a>
</p>

