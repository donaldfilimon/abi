# Runtime Module

The runtime module provides always-on core infrastructure for task scheduling, execution, concurrency primitives, and memory management. These services are available regardless of which features are enabled.

## What It Provides

- **Task Execution Engine**: Work-stealing scheduler with configurable policies
- **Task Scheduling**: Futures, promises, task groups, and async task management
- **Lock-Free Concurrency**: High-performance lock-free data structures for concurrent code
- **Memory Management**: Memory pools and arena allocators for efficient resource handling

## Key Components

### Engine (`engine/`)

The task execution engine provides work-stealing scheduling for parallel task execution:

- `Engine`: Main compute engine with work-stealing scheduler
- `DistributedComputeEngine`: For distributed task execution
- `ResultCache`: Fast-path task completion caching with memoization
- `StealPolicy`: NUMA-aware and round-robin work-stealing policies

```zig
var engine = try runtime.createEngine(allocator, config);
defer engine.deinit();
```

### Scheduling (`scheduling/`)

Asynchronous primitives for coordinating concurrent work:

- **Futures/Promises**: `Future(T)`, `Promise`, `all()`, `race()`
- **Task Groups**: `TaskGroup` for parallel iteration with `parallelForEach()`
- **Cancellation**: `CancellationToken`, `CancellationSource`, `ScopedCancellation`
- **Async Runtime**: `AsyncRuntime` for managing async task execution

```zig
var group = try ctx.createTaskGroup(.{ .max_threads = 4 });
defer group.deinit();

try group.parallelForEach(items, processItem);
```

### Concurrency (`concurrency/`)

Lock-free data structures for high-performance concurrent applications:

| Primitive | Purpose |
|-----------|---------|
| `ChaseLevDeque` | Work-stealing deque for task scheduling |
| `EpochReclamation` | Safe memory reclamation for lock-free structures |
| `WorkStealingScheduler` | Scheduling algorithm with work stealing |
| `LockFreeStackEBR` | Stack with epoch-based reclamation |
| `MpmcQueue` | Multi-producer multi-consumer bounded queue |
| `BlockingMpmcQueue` | MPMC queue with blocking push/pop |
| `PriorityQueue` | Lock-free priority queue |
| `WorkStealingQueue` | Work queue with stealing support |

```zig
const concurrency = @import("abi").runtime.concurrency;

var queue = try concurrency.MpmcQueue(u64).init(allocator, 1024);
defer queue.deinit();

try queue.push(42);
if (queue.pop()) |value| {
    // Process value
}
```

### Memory (`memory/`)

Efficient memory management utilities:

- `MemoryPool`: Configurable memory allocation with defragmentation
- `ArenaAllocator`: Fast arena-based allocation for scoped allocations

```zig
var pool = try memory.MemoryPool.init(allocator, .{
    .strategy = .best_fit,
    .auto_defrag = true,
});
defer pool.deinit();

const buffer = try pool.allocate(32768);
defer pool.free(buffer);
```

## Usage Example

```zig
const abi = @import("abi");
const runtime = abi.runtime;
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize runtime context
    var ctx = try runtime.Context.init(allocator);
    defer ctx.deinit();

    // Create a task group for parallel work
    var group = try ctx.createTaskGroup(.{ .max_threads = 4 });
    defer group.deinit();

    // Define work items
    const items = [_]u32{ 1, 2, 3, 4, 5 };

    // Execute in parallel
    try group.parallelForEach(&items, struct {
        fn process(item: u32) !void {
            std.debug.print("Processing: {}\n", .{item});
        }
    }.process);

    // Use lock-free queue for concurrent communication
    var queue = try runtime.MpmcQueue(u32).init(allocator, 100);
    defer queue.deinit();

    try queue.push(42);
    if (queue.pop()) |value| {
        std.debug.print("Received: {}\n", .{value});
    }
}
```

## Architecture

The runtime module is organized into four subdomains:

```
runtime/
├── mod.zig           # Entry point and Context definition
├── engine/           # Task execution engine
├── scheduling/       # Futures, promises, task groups, cancellation
├── concurrency/      # Lock-free data structures and primitives
└── memory/           # Memory pools and allocators
```

All types are re-exported from `mod.zig` for convenient access via `runtime.*`.

## Runtime Context

The `Context` type provides lazy initialization and access to runtime components:

```zig
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Lazily create engine on first access
const engine = try ctx.getEngine();

// Create task groups
var group = try ctx.createTaskGroup(config);

// Create futures
var future = try ctx.createFuture(u32);
```

## Integration

The runtime module is always available and requires no feature flags. Access it through the public API:

```zig
const abi = @import("abi");
const runtime = abi.runtime;

// Access any runtime component
const deque = runtime.ChaseLevDeque(Task).init(allocator);
const queue = try runtime.MpmcQueue(Message).init(allocator, 1000);
```

## Performance Considerations

- **Work Stealing**: Minimizes lock contention and load imbalance
- **Lock-Free Structures**: No mutexes; safe for high-concurrency scenarios
- **Memory Pools**: Reduces allocation fragmentation
- **Result Caching**: Fast-path for previously computed results

For GPU workload execution, use the `workload` types with custom vtables for backend-specific implementations.
