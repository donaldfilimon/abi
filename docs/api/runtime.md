---
title: runtime API
purpose: Generated API reference for runtime
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# runtime

> Runtime Module - Always-on Core Infrastructure

This module provides the foundational runtime infrastructure that is always
available regardless of which features are enabled. It includes:

- Task scheduling and execution engine
- Concurrency primitives (futures, task groups, cancellation)
- Memory management utilities

## Module Organization

```
runtime/
├── mod.zig          # This file - unified entry point
├── engine/          # Task execution engine
├── scheduling/      # Futures, cancellation, task groups
├── concurrency/     # Lock-free data structures
└── memory/          # Memory pools and allocators
```

## Usage

```zig
const runtime = @import("runtime");

// Create runtime context
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Use task groups for parallel work
var group = try ctx.createTaskGroup(.{});
defer group.deinit();
```

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-engine"></a>`pub const Engine`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L51)

Core task execution engine.

### <a id="pub-const-distributedcomputeengine"></a>`pub const DistributedComputeEngine`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L53)

Compute engine with distributed execution capabilities.

### <a id="pub-const-engineconfig"></a>`pub const EngineConfig`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L55)

Configuration for the task engine.

### <a id="pub-const-engineerror"></a>`pub const EngineError`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L57)

Error set for compute engine operations.

### <a id="pub-const-taskid"></a>`pub const TaskId`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L59)

Unique identifier for a task.

### <a id="pub-const-executioncontext"></a>`pub const ExecutionContext`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L63)

Context providing resources for workload execution.

### <a id="pub-const-workloadhints"></a>`pub const WorkloadHints`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L65)

Optimization hints for the workload scheduler.

### <a id="pub-const-workloadvtable"></a>`pub const WorkloadVTable`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L67)

Interface for custom compute workloads.

### <a id="pub-const-gpuworkloadvtable"></a>`pub const GPUWorkloadVTable`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L69)

Interface for GPU-accelerated workloads.

### <a id="pub-const-resulthandle"></a>`pub const ResultHandle`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L71)

Handle to an asynchronous execution result.

### <a id="pub-const-resultvtable"></a>`pub const ResultVTable`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L73)

Virtual table for result polling and retrieval.

### <a id="pub-const-workitem"></a>`pub const WorkItem`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L75)

A single unit of work to be executed.

### <a id="pub-const-runworkitem"></a>`pub const runWorkItem`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L77)

Helper to execute a work item in a specific context.

### <a id="pub-const-benchmarkresult"></a>`pub const BenchmarkResult`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L81)

Results from a benchmark run.

### <a id="pub-const-runbenchmarks"></a>`pub const runBenchmarks`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L83)

Execute the internal benchmark suite.

### <a id="pub-const-future"></a>`pub const Future`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L91)

A handle to a future value that may not be available yet.

### <a id="pub-const-futurestate"></a>`pub const FutureState`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L93)

Current state of a future (pending, completed, etc.).

### <a id="pub-const-futureresult"></a>`pub const FutureResult`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L95)

Result of a future completion.

### <a id="pub-const-promise"></a>`pub const Promise`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L97)

The producer side of a future value.

### <a id="pub-const-all"></a>`pub const all`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L99)

Wait for all futures in a list to complete.

### <a id="pub-const-race"></a>`pub const race`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L101)

Wait for the first future in a list to complete.

### <a id="pub-const-delay"></a>`pub const delay`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L103)

Create a future that completes after a delay.

### <a id="pub-const-cancellationtoken"></a>`pub const CancellationToken`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L107)

Token used to signal and check for operation cancellation.

### <a id="pub-const-cancellationsource"></a>`pub const CancellationSource`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L109)

Source that can trigger cancellation for associated tokens.

### <a id="pub-const-cancellationstate"></a>`pub const CancellationState`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L111)

State of a cancellation request.

### <a id="pub-const-cancellationreason"></a>`pub const CancellationReason`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L113)

Reason for an operation being cancelled.

### <a id="pub-const-linkedcancellation"></a>`pub const LinkedCancellation`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L115)

Cancellation token that propagates from a parent source.

### <a id="pub-const-scopedcancellation"></a>`pub const ScopedCancellation`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L117)

Token that automatically cancels when it goes out of scope.

### <a id="pub-const-taskgroup"></a>`pub const TaskGroup`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L121)

Group of related tasks that can be managed together.

### <a id="pub-const-taskgroupconfig"></a>`pub const TaskGroupConfig`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L123)

Configuration for task group behavior.

### <a id="pub-const-taskgroupbuilder"></a>`pub const TaskGroupBuilder`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L125)

Builder for constructing task groups.

### <a id="pub-const-scopedtaskgroup"></a>`pub const ScopedTaskGroup`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L127)

Task group that automatically waits/cancels on scope exit.

### <a id="pub-const-taskcontext"></a>`pub const TaskContext`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L129)

Execution context for a specific task.

### <a id="pub-const-taskfn"></a>`pub const TaskFn`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L131)

Function signature for tasks in a group.

### <a id="pub-const-taskstate"></a>`pub const TaskState`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L133)

Lifecycle state of a task in a group.

### <a id="pub-const-taskresult"></a>`pub const TaskResult`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L135)

Outcome of a task execution.

### <a id="pub-const-taskinfo"></a>`pub const TaskInfo`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L137)

Metadata about a task in a group.

### <a id="pub-const-groupstats"></a>`pub const GroupStats`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L139)

Performance metrics for a task group.

### <a id="pub-const-parallelforeach"></a>`pub const parallelForEach`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L141)

Execute a function in parallel for each element in a slice.

### <a id="pub-const-asyncruntime"></a>`pub const AsyncRuntime`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L145)

High-level asynchronous runtime for task orchestration.

### <a id="pub-const-asyncruntimeoptions"></a>`pub const AsyncRuntimeOptions`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L147)

Options for configuring the async runtime.

### <a id="pub-const-taskhandle"></a>`pub const TaskHandle`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L149)

Handle to a task running in the async runtime.

### <a id="pub-const-asynctaskgroup"></a>`pub const AsyncTaskGroup`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L151)

Group for managing tasks within the async runtime.

### <a id="pub-const-asyncerror"></a>`pub const AsyncError`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L153)

Error set for async runtime operations.

### <a id="pub-const-threadpool"></a>`pub const ThreadPool`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L157)

Work-stealing thread pool for high-performance batch processing.

### <a id="pub-const-threadpooltask"></a>`pub const ThreadPoolTask`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L159)

A single task to be executed by the thread pool.

### <a id="pub-const-parallelfor"></a>`pub const parallelFor`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L161)

Execute a loop in parallel using the thread pool.

### <a id="pub-const-dagpipeline"></a>`pub const DagPipeline`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L165)

Directed Acyclic Graph (DAG) pipeline for complex data flows.

### <a id="pub-const-dagpipelineresult"></a>`pub const DagPipelineResult`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L167)

Collective result from a pipeline execution.

### <a id="pub-const-dagstage"></a>`pub const DagStage`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L169)

A single stage within a compute pipeline.

### <a id="pub-const-dagstagestatus"></a>`pub const DagStageStatus`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L171)

Current execution status of a pipeline stage.

### <a id="pub-const-createinferencepipeline"></a>`pub const createInferencePipeline`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L173)

Create a standard inference pipeline for LLM processing.

### <a id="pub-const-workstealingqueue"></a>`pub const WorkStealingQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L180)

Queue optimized for work-stealing schedulers.

### <a id="pub-const-workqueue"></a>`pub const WorkQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L182)

Generic thread-safe work queue.

### <a id="pub-const-lockfreequeue"></a>`pub const LockFreeQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L184)

Multi-producer, multi-consumer lock-free queue.

### <a id="pub-const-lockfreestack"></a>`pub const LockFreeStack`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L186)

Thread-safe lock-free stack.

### <a id="pub-const-shardedmap"></a>`pub const ShardedMap`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L188)

Map partitioned into shards to reduce contention.

### <a id="pub-const-priorityqueue"></a>`pub const PriorityQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L190)

Thread-safe priority queue.

### <a id="pub-const-backoff"></a>`pub const Backoff`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L192)

Exponential backoff helper for spin-loops.

### <a id="pub-const-chaselevdeque"></a>`pub const ChaseLevDeque`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L196)

Double-ended queue for work-stealing schedulers.

### <a id="pub-const-workstealingscheduler"></a>`pub const WorkStealingScheduler`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L198)

Scheduler that uses work-stealing for load balancing.

### <a id="pub-const-epochreclamation"></a>`pub const EpochReclamation`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L200)

Epoch-based memory reclamation for lock-free structures.

### <a id="pub-const-lockfreestackebr"></a>`pub const LockFreeStackEBR`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L202)

Lock-free stack using epoch-based reclamation.

### <a id="pub-const-mpmcqueue"></a>`pub const MpmcQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L204)

High-performance multi-producer, multi-consumer queue.

### <a id="pub-const-blockingmpmcqueue"></a>`pub const BlockingMpmcQueue`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L206)

MPMC queue that blocks when empty or full.

### <a id="pub-const-channel"></a>`pub const Channel`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L210)

Multi-producer, multi-consumer communication channel.

### <a id="pub-const-bytechannel"></a>`pub const ByteChannel`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L212)

Channel optimized for raw byte transfers.

### <a id="pub-const-messagechannel"></a>`pub const MessageChannel`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L214)

Channel for typed message passing.

### <a id="pub-const-channelmessage"></a>`pub const ChannelMessage`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L216)

Standard message structure for channels.

### <a id="pub-const-resultcache"></a>`pub const ResultCache`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L220)

Cache for storing and reusing compute results.

### <a id="pub-const-cacheconfig"></a>`pub const CacheConfig`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L222)

Configuration for the result cache.

### <a id="pub-const-cachestats"></a>`pub const CacheStats`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L224)

Metrics for cache hits, misses, and evictions.

### <a id="pub-const-memoize"></a>`pub const Memoize`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L226)

Memoization helper for function results.

### <a id="pub-const-numastealpolicy"></a>`pub const NumaStealPolicy`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L230)

Work-stealing policy that respects NUMA boundaries.

### <a id="pub-const-roundrobinstealpolicy"></a>`pub const RoundRobinStealPolicy`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L232)

Simple round-robin work-stealing policy.

### <a id="pub-const-stealpolicyconfig"></a>`pub const StealPolicyConfig`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L234)

Configuration for stealing behavior.

### <a id="pub-const-stealstats"></a>`pub const StealStats`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L236)

Metrics for work-stealing performance.

### <a id="pub-const-memorypool"></a>`pub const MemoryPool`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L243)

Thread-safe object pool for reducing allocation pressure.

### <a id="pub-const-arenaallocator"></a>`pub const ArenaAllocator`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L245)

Arena allocator for fast bulk allocations.

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/services/runtime/mod.zig#L254)

Runtime context - the always-available infrastructure.
This is created automatically by the Framework and provides
access to scheduling, concurrency, and memory primitives.

### <a id="pub-fn-init-allocator-std-mem-allocator-error-context"></a>`pub fn init(allocator: std.mem.Allocator) Error!*Context`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L266)

Initialize the runtime context.

### <a id="pub-fn-deinit-self-context-void"></a>`pub fn deinit(self: *Context) void`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L276)

Shutdown the runtime context.

### <a id="pub-fn-getengine-self-context-error-engine"></a>`pub fn getEngine(self: *Context) Error!*Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L286)

Get or create the compute engine.

### <a id="pub-fn-createtaskgroup-self-context-config-taskgroupconfig-taskgroup"></a>`pub fn createTaskGroup(self: *Context, config: TaskGroupConfig) !TaskGroup`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L299)

Create a new task group.

### <a id="pub-fn-createfuture-self-context-comptime-t-type-future-t"></a>`pub fn createFuture(self: *Context, comptime T: type) !Future(T)`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L304)

Create a new future.

### <a id="pub-fn-createcancellationsource-self-context-cancellationsource"></a>`pub fn createCancellationSource(self: *Context) !CancellationSource`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L309)

Create a cancellation source.

### <a id="pub-fn-createengine-allocator-std-mem-allocator-config-engineconfig-engine"></a>`pub fn createEngine(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L319)

Create an engine with configuration (2-arg version for compatibility).

### <a id="pub-fn-createdefaultengine-allocator-std-mem-allocator-engine"></a>`pub fn createDefaultEngine(allocator: std.mem.Allocator) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L324)

Create an engine with default configuration.

### <a id="pub-fn-createenginewithconfig-allocator-std-mem-allocator-config-engineconfig-engine"></a>`pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup> | [source](../../src/services/runtime/mod.zig#L329)

Create an engine with custom configuration.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
