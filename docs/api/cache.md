---
title: cache API
purpose: Generated API reference for cache
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# cache

> Cache Module

In-memory caching with LRU/LFU/FIFO/Random eviction, TTL support,
and thread-safe concurrent access via RwLock.

Architecture:
- SwissMap-backed storage for O(1) key lookup
- 4 eviction strategies: LRU (doubly-linked list), LFU (frequency buckets),
FIFO (queue), Random (RNG selection)
- Lazy TTL expiration on get() + size-triggered sweep
- RwLock for read-heavy concurrency (multiple readers, single writer)
- Cache owns all keys/values (copies on put, caller borrows on get)

**Source:** [`src/features/cache/mod.zig`](../../src/features/cache/mod.zig)

**Build flag:** `-Dfeat_cache=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-cacheconfig-cacheerror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: CacheConfig) CacheError!void`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L370)

Initialize the global cache singleton with the given configuration.
No-op if already initialized.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L376)

Tear down the cache, freeing all entries and internal structures.

### <a id="pub-fn-isenabled-bool"></a>`pub fn isEnabled() bool`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L384)

Returns `true` — the cache module is always enabled at compile time.

### <a id="pub-fn-isinitialized-bool"></a>`pub fn isInitialized() bool`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L389)

Returns whether the cache has been initialized via `init`.

### <a id="pub-fn-get-key-const-u8-cacheerror-const-u8"></a>`pub fn get(key: []const u8) CacheError!?[]const u8`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L399)

Look up a cached value by key. Returns the value slice (borrowed from
the cache) or `null` if the key is absent or expired. Thread-safe.

For FIFO/Random eviction (no promotion needed), uses a shared read lock
to allow concurrent readers. For LRU/LFU, uses an exclusive lock since
promotion mutates the list/frequency.

### <a id="pub-fn-put-key-const-u8-value-const-u8-cacheerror-void"></a>`pub fn put(key: []const u8, value: []const u8) CacheError!void`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L452)

Insert a key-value pair using the default TTL from config.

### <a id="pub-fn-putwithttl-key-const-u8-value-const-u8-ttl-ms-u64-cacheerror-void"></a>`pub fn putWithTtl(key: []const u8, value: []const u8, ttl_ms: u64) CacheError!void`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L458)

Insert a key-value pair with a specific TTL in milliseconds.
Evicts entries as needed if the cache is at capacity.

### <a id="pub-fn-delete-key-const-u8-cacheerror-bool"></a>`pub fn delete(key: []const u8) CacheError!bool`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L536)

Remove an entry by key. Returns `true` if the key was present.

### <a id="pub-fn-contains-key-const-u8-bool"></a>`pub fn contains(key: []const u8) bool`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L549)

Test whether a key exists and has not expired (read-only, no eviction promotion).

### <a id="pub-fn-clear-void"></a>`pub fn clear() void`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L558)

Remove all entries from the cache.

### <a id="pub-fn-size-u32"></a>`pub fn size() u32`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L576)

Return the current number of live entries.

### <a id="pub-fn-stats-cachestats"></a>`pub fn stats() CacheStats`

<sup>**fn**</sup> | [source](../../src/features/cache/mod.zig#L584)

Snapshot hit/miss counters, entry count, memory usage, and eviction stats.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
