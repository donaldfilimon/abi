# runtime-concurrency API Reference

> Lock-free concurrent primitives for high-performance task execution

**Source:** [`src/runtime/concurrency/mod.zig`](../../src/runtime/concurrency/mod.zig)

---

## Overview

The concurrency module provides lock-free and thread-safe data structures optimized for work-stealing schedulers and high-throughput concurrent applications.

### Key Features

- **Lock-free operations**: Most common paths require no locks
- **ABA-safe**: Epoch-based reclamation prevents ABA problems
- **NUMA-aware**: Topology-aware work stealing for optimal memory access
- **Cache-friendly**: Designed to minimize false sharing and cache misses

---

## Core Types

### ChaseLevDeque

Lock-free work-stealing deque based on Chase-Lev algorithm (SPAA 2005).

**Properties:**
- Owner push/pop: O(1) amortized, no contention
- Thief steal: O(1), may contend with other thieves
- Dynamic resizing (grows as needed)

```zig
const runtime = @import("abi").runtime.concurrency;

// Initialize
var deque = try runtime.ChaseLevDeque(Task).init(allocator);
defer deque.deinit();

// Owner thread operations
try deque.push(task);           // Push to bottom (LIFO)
const my_task = deque.pop();     // Pop from bottom

// Thief thread operations
const stolen = deque.steal();    // Steal from top (FIFO)
```

**Methods:**
| Method | Description |
|--------|-------------|
| `init(allocator)` | Create with default capacity (16) |
| `initWithCapacity(allocator, cap)` | Create with specific capacity (power of 2) |
| `push(value)` | Push to bottom (owner only) |
| `pop()` | Pop from bottom (owner only) |
| `steal()` | Steal from top (any thread) |
| `isEmpty()` | Check if empty |
| `deinit()` | Free resources |

---

### EpochReclamation

Safe memory reclamation for lock-free data structures using epoch-based reclamation (EBR).

**How it works:**
1. Threads "pin" themselves to an epoch when accessing shared data
2. Memory is placed in a retirement list when "freed"
3. Memory is only truly freed when all threads have advanced past the retirement epoch

```zig
const runtime = @import("abi").runtime.concurrency;

var ebr = runtime.EpochReclamation.init(allocator);
defer ebr.deinit();

// Pin before accessing shared data
ebr.pin();
defer ebr.unpin();

// When done with a node, retire instead of freeing
ebr.retire(node);
```

**Methods:**
| Method | Description |
|--------|-------------|
| `init(allocator)` | Initialize EBR manager |
| `pin()` | Pin current thread to global epoch |
| `unpin()` | Unpin thread, allowing epoch advancement |
| `retire(ptr)` | Schedule pointer for deferred reclamation |
| `retireWithFn(ptr, fn)` | Retire with custom deinitialization |
| `deinit()` | Free all resources |

---

### MpmcQueue

Lock-free multi-producer multi-consumer bounded queue based on Dmitry Vyukov's algorithm.

**Properties:**
- Lock-free for all operations
- Bounded memory usage (fixed capacity)
- FIFO ordering
- No ABA problem due to monotonically increasing sequences

```zig
const runtime = @import("abi").runtime.concurrency;

// Initialize with capacity (must be power of 2)
var queue = try runtime.MpmcQueue(Message).init(allocator, 1024);
defer queue.deinit();

// Any thread can push
try queue.push(message);

// Any thread can pop
if (queue.pop()) |msg| {
    // process message
}
```

**Methods:**
| Method | Description |
|--------|-------------|
| `init(allocator, capacity)` | Create with capacity (power of 2) |
| `push(value)` | Push value, returns `error.QueueFull` if full |
| `tryPush(value)` | Try push, returns bool |
| `pop()` | Pop value, returns null if empty |
| `isEmpty()` | Check if empty |
| `len()` | Get current length |
| `deinit()` | Free resources |

---

### BlockingMpmcQueue

MPMC queue with blocking operations for producer/consumer patterns.

```zig
const runtime = @import("abi").runtime.concurrency;

var queue = try runtime.BlockingMpmcQueue(Task).init(allocator, 256);
defer queue.deinit();

// Blocking push (waits if full)
try queue.pushBlocking(task);

// Blocking pop (waits if empty)
const task = try queue.popBlocking();

// Timed operations
const task = queue.popTimeout(1_000_000_000); // 1 second timeout
```

---

### LockFreeStackEBR

ABA-safe lock-free stack using epoch-based reclamation.

```zig
const runtime = @import("abi").runtime.concurrency;

var stack = runtime.LockFreeStackEBR(Node).init(allocator);
defer stack.deinit();

// Push (any thread)
try stack.push(node);

// Pop (any thread)
if (stack.pop()) |node| {
    // use node
}
```

---

### PriorityQueue

Lock-free priority queue for task scheduling with multiple priority levels.

```zig
const runtime = @import("abi").runtime.concurrency;

var pq = try runtime.PriorityQueue(Task).init(allocator, .{
    .levels = 8,
    .capacity_per_level = 256,
});
defer pq.deinit();

// Push with priority
try pq.push(task, .high);

// Pop highest priority available
if (pq.pop()) |task| {
    // process task
}
```

**Priority Levels:**
| Level | Value | Use Case |
|-------|-------|----------|
| `critical` | 0 | System-critical tasks |
| `high` | 1 | User-facing latency-sensitive |
| `normal` | 2 | Default priority |
| `low` | 3 | Background work |
| `idle` | 4 | Only when system idle |

---

## Utility Types

### WorkStealingQueue

Simple mutex-based work-stealing queue for comparison/fallback.

```zig
var queue = runtime.WorkStealingQueue(Task).init(allocator);
defer queue.deinit();

try queue.push(task);     // Owner pushes to back
const t = queue.pop();     // Owner pops from back (LIFO)
const s = queue.steal();   // Thief steals from front (FIFO)
```

### WorkQueue

Simple FIFO queue with mutex for low-contention scenarios.

```zig
var queue = runtime.WorkQueue(Task).init(allocator);
defer queue.deinit();

try queue.enqueue(task);
const t = queue.dequeue();
```

### Backoff

Exponential backoff utility for spin-wait loops.

```zig
var backoff = runtime.Backoff{};

while (!condition) {
    backoff.spin();  // Progressive backoff
}

backoff.reset();  // Reset for reuse
```

### ShardedMap

Partitioned map to reduce contention on concurrent access.

```zig
var map = try runtime.ShardedMap(K, V).init(allocator, 16); // 16 shards
defer map.deinit();

try map.put(key, value);
const v = map.get(key);
```

---

## Engine Types

These types are in `src/runtime/engine/` and complement the concurrency primitives.

### ResultCache

High-performance cache for task results with LRU eviction and TTL support.

```zig
const engine = @import("abi").runtime.engine;

var cache = try engine.ResultCache(TaskKey, TaskResult).init(allocator, .{
    .max_entries = 4096,
    .ttl_ms = 60000,      // 1 minute TTL
    .shard_count = 16,
    .enable_stats = true,
});
defer cache.deinit();

// Try cache first
if (cache.get(key)) |result| {
    return result;
}

// Compute and cache
const result = compute(key);
cache.put(key, result);

// Get statistics
const stats = cache.getStats();
std.debug.print("Hit rate: {d:.1}%\n", .{stats.hitRate()});
```

**CacheConfig Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `max_entries` | 4096 | Maximum cached entries |
| `shard_count` | 16 | Write distribution shards |
| `ttl_ms` | 0 | Time-to-live (0 = no expiry) |
| `enable_stats` | true | Collect hit/miss statistics |
| `eviction_batch` | 16 | Entries to evict when full |

---

### NumaStealPolicy

NUMA-aware victim selection for work-stealing schedulers.

```zig
const engine = @import("abi").runtime.engine;

var policy = try engine.NumaStealPolicy.init(allocator, topology, worker_count, .{
    .local_steal_probability = 80,  // 80% prefer local
    .max_steal_attempts = 8,
    .adaptive = true,
});
defer policy.deinit();

// Select victim for stealing
if (policy.selectVictim(worker_id, &rng)) |victim| {
    if (workers[victim].deque.steal()) |task| {
        // Got work from victim
    }
}

// Get statistics
const stats = policy.getStats();
std.debug.print("Local steal rate: {d:.1}%\n", .{stats.localRate()});
```

**StealPolicyConfig Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `local_steal_probability` | 80 | % preference for same NUMA node |
| `max_steal_attempts` | 8 | Attempts before giving up |
| `prefer_loaded_victims` | true | Target workers with more work |
| `adaptive` | true | Adjust based on load |
| `backoff_factor` | 2 | Backoff multiplier on failures |

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| ChaseLevDeque push | ~15ns | Single atomic store |
| ChaseLevDeque pop | ~20ns | Owner-only, no contention |
| ChaseLevDeque steal | ~30ns | May contend with other thieves |
| MpmcQueue push | ~25ns | Single CAS typical |
| MpmcQueue pop | ~25ns | Single CAS typical |
| ResultCache hit | ~20ns | Atomic load + hash |
| ResultCache miss | ~50ns | Hash + atomic load + null check |
| ResultCache put | ~100ns | Hash + lock + insert |
| EBR pin/unpin | ~10ns | Atomic stores |

---

## Thread Safety

| Type | Push | Pop | Steal | Notes |
|------|------|-----|-------|-------|
| ChaseLevDeque | Owner only | Owner only | Any thread | Classic work-stealing |
| MpmcQueue | Any thread | Any thread | N/A | Full MPMC |
| LockFreeStackEBR | Any thread | Any thread | N/A | ABA-safe |
| PriorityQueue | Any thread | Any thread | N/A | Lock-free |
| WorkStealingQueue | Mutex | Mutex | Mutex | Simple fallback |

---

## Best Practices

1. **Prefer ChaseLevDeque** for work-stealing schedulers
2. **Use EpochReclamation** when nodes may be accessed after logical deletion
3. **Size MpmcQueue** as power of 2 for optimal performance
4. **Enable ResultCache stats** during development, disable in production
5. **Configure NumaStealPolicy** based on actual NUMA topology
6. **Use Backoff** in spin loops to prevent CPU starvation

---

## See Also

- [runtime.md](runtime.md) - Runtime module overview
- [runtime-engine.md](runtime-engine.md) - Task engine documentation
- [runtime-scheduling.md](runtime-scheduling.md) - Scheduling primitives

---

*Generated from source documentation*
