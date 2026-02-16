---
title: Cache
description: In-memory caching with LRU/LFU/FIFO/Random eviction, TTL support, and thread-safe concurrent access
section: Data
order: 2
---

# Cache

The cache module (`src/features/cache/mod.zig`) provides high-performance
in-memory caching with four eviction policies, TTL-based expiration,
thread-safe concurrent access via RwLock, and slab-based memory allocation
to minimize per-entry heap overhead.

## Features

- **Four eviction policies**: LRU, LFU, FIFO, and Random
- **TTL support**: Per-entry time-to-live with lazy expiration on `get()` plus size-triggered sweep
- **Thread-safe**: RwLock for read-heavy concurrency (multiple concurrent readers, single writer)
- **Slab allocation**: Entries stored in a slab array to avoid per-entry heap allocations
- **Ownership model**: Cache owns all keys and values (copies on `put()`, caller borrows on `get()`)
- **Atomic statistics**: Lock-free reads for hit/miss/eviction counters
- **Zero overhead when disabled**: Comptime feature gating eliminates the module from the binary

## Build Configuration

```bash
# Enable (default)
zig build -Denable-cache=true

# Disable
zig build -Denable-cache=false
```

**Namespace**: `abi.cache`

## Quick Start

### Framework Integration

```zig
const abi = @import("abi");

// Initialize with cache enabled
var fw = try abi.Framework.init(allocator, .{
    .cache = .{
        .max_entries = 10000,
        .eviction_policy = .lru,
    },
});
defer fw.deinit();
```

### Standalone Usage

```zig
const cache = abi.cache;

// Initialize the global cache singleton
try cache.init(allocator, .{
    .max_entries = 1000,
    .eviction_policy = .lru,
});
defer cache.deinit();

// Store a value
try cache.put("user:42", "{\"name\": \"Alice\"}");

// Store with TTL (milliseconds)
try cache.putWithTtl("session:abc", "token_data", 30_000); // 30 seconds

// Retrieve a value
if (try cache.get("user:42")) |value| {
    // value is a borrowed slice -- valid until the entry is evicted or deleted
    std.debug.print("Found: {s}\n", .{value});
}

// Check existence without retrieving
if (cache.contains("user:42")) {
    // Key exists and is not expired
}

// Delete a specific key
const deleted = try cache.delete("user:42");

// Clear all entries
cache.clear();

// Get current entry count
const count = cache.size();

// Get detailed statistics
const st = cache.stats();
std.debug.print("Hits: {}, Misses: {}, Evictions: {}\n", .{
    st.hits, st.misses, st.evictions,
});
```

## API Reference

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(allocator, CacheConfig) !void` | Initialize the global cache singleton |
| `deinit` | `() void` | Tear down the cache and free all memory |
| `isEnabled` | `() bool` | Returns `true` when cache is compiled in |
| `isInitialized` | `() bool` | Returns `true` after successful `init()` |
| `get` | `(key) !?[]const u8` | Retrieve a value by key (lazy TTL check) |
| `put` | `(key, value) !void` | Store a key-value pair (no expiry) |
| `putWithTtl` | `(key, value, ttl_ms) !void` | Store with time-to-live in milliseconds |
| `delete` | `(key) !bool` | Delete a key; returns whether it existed |
| `contains` | `(key) bool` | Check if a non-expired key exists |
| `clear` | `() void` | Remove all entries |
| `size` | `() u32` | Current number of live entries |
| `stats` | `() CacheStats` | Aggregate hit/miss/eviction statistics |

### Types

| Type | Description |
|------|-------------|
| `CacheConfig` | Configuration struct (max entries, eviction policy) |
| `EvictionPolicy` | Enum: `.lru`, `.lfu`, `.fifo`, `.random` |
| `CacheEntry` | Public view of a cached entry (key, value, TTL, created_at) |
| `CacheStats` | Statistics: hits, misses, entries, memory_used, evictions, expired |
| `CacheError` | Error set: `FeatureDisabled`, `CacheFull`, `KeyNotFound`, `InvalidTTL`, `OutOfMemory` |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_entries` | `u32` | -- | Maximum number of cached entries |
| `eviction_policy` | `EvictionPolicy` | `.lru` | Eviction strategy when cache is full |

## Eviction Policies

### LRU (Least Recently Used)

Evicts the entry that was accessed least recently. Implemented with a
doubly-linked list where each `get()` moves the entry to the head. Eviction
removes from the tail.

Best for workloads with temporal locality -- recently accessed items are
likely to be accessed again.

### LFU (Least Frequently Used)

Evicts the entry with the lowest access frequency. Each `get()` increments
the entry's frequency counter. The eviction scan finds the entry with the
minimum frequency.

Best for workloads where some keys are consistently popular regardless of
recency.

### FIFO (First In, First Out)

Evicts the oldest entry regardless of access pattern. Uses the same
doubly-linked list as LRU but does not promote entries on `get()`.

Best for streaming workloads where older data naturally loses relevance.

### Random

Evicts a randomly selected entry using a splitmix64 PRNG. Provides O(1)
eviction with no metadata overhead.

Best when access patterns are unpredictable or eviction fairness is
acceptable.

## Architecture

The cache uses a layered internal design:

1. **KeyMap** (`StringHashMapUnmanaged(NodeIndex)`) -- O(1) key-to-slab lookup
2. **EntrySlab** -- Contiguous array of `InternalEntry` structs with a free list for recycling slots
3. **Linked list** -- Doubly-linked list threaded through slab entries for LRU/FIFO ordering
4. **RwLock** -- Read-write lock allowing concurrent readers with exclusive writers
5. **Atomic counters** -- Hit/miss/eviction stats readable without acquiring the lock

TTL expiration is lazy: expired entries are detected and removed on `get()`.
When the cache reaches capacity and needs to evict, a sweep also purges
expired entries before selecting a victim.

## Disabling at Build Time

```bash
zig build -Denable-cache=false
```

When disabled, `abi.cache` resolves to `src/features/cache/stub.zig`, which
returns `error.FeatureDisabled` for all mutating operations and safe defaults
for read-only calls (`isEnabled()` returns `false`, `size()` returns 0, etc.).

## Examples

See [`examples/cache.zig`](https://github.com/donaldfilimon/abi/blob/main/examples/cache.zig)
for a complete working example.

## Related

- [Database](database.html) -- Vector database with query caching
- [Storage](storage.html) -- Persistent object storage
- [Search](search.html) -- Full-text search (pair with cache for query results)
- [Architecture](architecture.html) -- Comptime feature gating pattern
