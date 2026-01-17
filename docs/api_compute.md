# Compute API Reference

**Source:** `src/compute/runtime/mod.zig`

## Engine Functions

### `createEngine(allocator, config) !DistributedComputeEngine`

Create compute engine with custom config.

### `submitTask(engine, ResultType, task) !TaskId`

Submit task for execution, returns task ID.

### `waitForResult(engine, ResultType, id, timeout_ms) !ResultType`

Wait for task result by ID. See timeout semantics in [compute.md](compute.md).

### `runTask(engine, ResultType, task, timeout_ms) !ResultType`

Submit and wait for result in one call.

### `runWorkload(engine, ResultType, work, timeout_ms) !ResultType`

Alias for `runTask()`.

## Core Types

- `DistributedComputeEngine` - Main engine instance
- `EngineConfig` - Engine configuration
- `EngineError` - Engine error types
- `TaskId` - Task identifier

## Futures & Promises

- `Future(ResultType)` - Async result with `.then()`, `.catch()`
- `Promise(ResultType)` - Future producer
- `all(...)` - Wait for all futures
- `race(...)` - Wait for first future
- `delay(ms)` - Delayed future

## Cancellation

- `CancellationToken` - Cooperative cancellation
- `CancellationSource` - Token source
- `LinkedCancellation` - Linked cancellation tokens

## Task Groups

- `TaskGroup` - Hierarchical task organization
- `TaskGroupConfig` - Group configuration
- `parallelForEach()` - Parallel iteration

## Workloads

- `WorkloadVTable` - Workload interface
- `GPUWorkloadVTable` - GPU workload interface
- `WorkItem` - Work item type
- `matMul`, `dense`, `relu` - Built-in workload operations
