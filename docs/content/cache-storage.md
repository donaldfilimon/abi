---
title: Cache & Storage
description: In-memory caching with LRU/LFU eviction and unified object storage
section: Modules
order: 10
---

# Cache & Storage

ABI provides two complementary data modules: **Cache** for high-speed in-memory
key-value lookups with automatic eviction, and **Storage** for unified object
persistence with backend abstraction.

---

## Cache (`abi.cache`)

An in-memory key-value cache with four eviction policies, TTL-based expiry,
slab-allocated entries, and thread-safe concurrent access.

**Build flag:** `-Denable-cache=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Key lookup** | SwissMap-backed `StringHashMap` for O(1) key resolution |
| **Slab allocator** | Pre-allocated entry array avoids per-entry heap allocation |
| **Eviction** | LRU (doubly-linked list), LFU (frequency buckets), FIFO (queue), Random (RNG) |
| **TTL** | Lazy expiration on `get()` plus size-triggered sweep |
| **Concurrency** | `RwLock` for read-heavy workloads (multiple readers, single writer) |
| **Ownership** | Cache copies keys and values on `put()`; caller borrows on `get()` |
| **Stats** | Atomic counters for hits, misses, evictions, and expired entries |

### Eviction Policies

| Policy | Strategy | Best For |
|--------|----------|----------|
| `lru` | Evicts least-recently-used entry | General purpose, temporal locality |
| `lfu` | Evicts least-frequently-used entry | Hot-key workloads |
| `fifo` | Evicts oldest entry | Simple queue behavior |
| `random` | Evicts a random entry | Uniform access patterns |

### API

```zig
const abi = @import("abi");
const cache = abi.cache;

// Initialize with LRU eviction and 1000-entry capacity
try cache.init(allocator, .{
    .max_entries = 1000,
    .eviction_policy = .lru,
});
defer cache.deinit();

// Store a value (cache copies both key and value)
try cache.put("user:42", "{\"name\": \"Alice\"}");

// Store with TTL (expires after 30 seconds)
try cache.putWithTtl("session:abc", "token-data", 30_000);

// Retrieve (returns null if expired or missing)
if (try cache.get("user:42")) |value| {
    // value is borrowed from the cache — do not free
    std.log.info("found: {s}", .{value});
}

// Check existence without fetching
const exists = cache.contains("user:42");

// Delete a specific key
const deleted = try cache.delete("session:abc");

// Inspect stats
const s = cache.stats();
std.log.info("hits={d} misses={d} evictions={d}", .{
    s.hits, s.misses, s.evictions,
});

// Clear all entries
cache.clear();
```

### Types

| Type | Description |
|------|-------------|
| `CacheConfig` | Configuration: `max_entries`, `eviction_policy`, `default_ttl_ms` |
| `EvictionPolicy` | Enum: `.lru`, `.lfu`, `.fifo`, `.random` |
| `CacheEntry` | Key-value pair with TTL metadata |
| `CacheStats` | Aggregate stats: hits, misses, entries, memory_used, evictions, expired |
| `CacheError` | Error set: `FeatureDisabled`, `CacheFull`, `KeyNotFound`, `InvalidTTL`, `OutOfMemory` |

---

## Storage (`abi.storage`)

Unified file and object storage with a vtable-based backend abstraction. Ships
with an in-memory backend; local filesystem and cloud backends (S3, GCS) are
planned.

**Build flag:** `-Denable-storage=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Backend vtable** | Function pointers for `put`, `get`, `delete`, `list`, `exists`, `deinit` |
| **Memory backend** | `StringHashMap`-based in-memory store (default) |
| **Key validation** | Path traversal rejection (`../`, absolute paths, null bytes) |
| **Metadata** | Per-object `ObjectMetadata` with content type and up to 4 custom key-value pairs |
| **Concurrency** | `RwLock` protects backend state |

### API

```zig
const abi = @import("abi");
const storage = abi.storage;

// Initialize with memory backend
try storage.init(allocator, .{
    .backend = .memory,
    .max_object_size = 10 * 1024 * 1024, // 10 MB
});
defer storage.deinit();

// Store an object
try storage.putObject("docs/readme.txt", "Hello, world!");

// Store with metadata
try storage.putObjectWithMetadata(
    "images/logo.png",
    png_bytes,
    .{ .content_type = "image/png" },
);

// Retrieve (caller owns returned slice)
const data = try storage.getObject(allocator, "docs/readme.txt");
defer allocator.free(data);

// Check existence
const exists = try storage.objectExists("docs/readme.txt");

// List objects with optional prefix filter
const keys = try storage.listObjects(allocator, "docs/");
defer {
    for (keys) |k| allocator.free(k);
    allocator.free(keys);
}

// Delete
const deleted = try storage.deleteObject("docs/readme.txt");

// Aggregate stats
const s = storage.stats();
std.log.info("objects={d} bytes={d}", .{ s.total_objects, s.total_bytes });
```

### Types

| Type | Description |
|------|-------------|
| `StorageConfig` | Configuration: `backend`, `max_object_size`, `bucket` |
| `StorageBackend` | Enum: `.memory`, `.local`, `.s3`, `.gcs` |
| `StorageObject` | Object descriptor: key, size, content_type, last_modified |
| `ObjectMetadata` | Content type + up to 4 custom key-value pairs |
| `StorageStats` | Aggregate: total_objects, total_bytes, backend |
| `StorageError` | Error set: `ObjectNotFound`, `InvalidKey`, `StorageFull`, `PermissionDenied`, etc. |

### Security

All storage keys are validated before use:

- Empty keys and keys longer than 4096 bytes are rejected.
- Absolute paths (starting with `/`) are rejected.
- Path traversal sequences (`../`) are rejected.
- Null bytes are rejected.

These checks prevent directory traversal attacks when the storage backend maps
keys to filesystem paths.

---

## Disabling at Build Time

Both modules follow the standard comptime feature gating pattern. When disabled,
all public functions return `error.FeatureDisabled` with zero binary overhead:

```bash
# Disable cache only
zig build -Denable-cache=false

# Disable storage only
zig build -Denable-storage=false
```
