//! ABI Framework WASM Bindings
//!
//! Exports a subset of the ABI framework for use in WebAssembly environments.
//! OS-dependent features (network, gpu, database, storage, cloud, web,
//! profiling) are disabled at compile time via build flags.
//!
//! ## Design
//!
//! wasm32-freestanding lacks POSIX, libc, and 64-bit atomics — so the
//! upstream feature modules (cache, analytics, search, auth) cannot be
//! used directly. Instead, this file provides lightweight WASM-native
//! implementations that mirror the framework APIs and are safe for the
//! wasm32-freestanding target.
//!
//! The `abi` module is imported for comptime-safe metadata (version
//! string, type definitions) but runtime calls go through the local
//! WASM-safe implementations below.
//!
//! ## Memory Model
//!
//! WASM bindings use a fixed 1 MB scratch buffer (`FixedBufferAllocator`)
//! since OS allocators are unavailable on `wasm32-freestanding`.
//! The host controls the WASM linear memory; `alloc` / `dealloc` expose
//! a simple allocator interface for passing data across the boundary.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// WASM Allocator
// ============================================================================

/// 1 MB scratch space — sized for typical WASM workloads.
var wasm_buffer: [1024 * 1024]u8 = undefined;
var fba: std.heap.FixedBufferAllocator = std.heap.FixedBufferAllocator.init(&wasm_buffer);

fn wasmAllocator() std.mem.Allocator {
    return fba.allocator();
}

// ============================================================================
// Memory management exports
// ============================================================================

/// Allocate `len` bytes and return a pointer the host can write into.
/// Returns null on failure (OOM within the fixed buffer).
export fn alloc(len: usize) ?[*]u8 {
    const slice = fba.allocator().alloc(u8, len) catch return null;
    return slice.ptr;
}

/// Free a region previously obtained via `alloc`.
export fn dealloc(ptr: [*]u8, len: usize) void {
    fba.allocator().free(ptr[0..len]);
}

/// Reset the entire scratch allocator, reclaiming all memory.
/// Useful between independent request/response cycles.
export fn alloc_reset() void {
    fba.reset();
}

// ============================================================================
// Version exports
// ============================================================================

const VERSION_STRING = "ABI Framework v0.4.0 (WASM)";

/// Return the byte-length of the version string (so the host can allocate).
export fn version_len() usize {
    return VERSION_STRING.len;
}

/// Copy the version string into `out_ptr[0..out_max]`.
/// Returns the number of bytes actually written.
export fn version(out_ptr: [*]u8, out_max: usize) usize {
    const len = @min(VERSION_STRING.len, out_max);
    @memcpy(out_ptr[0..len], VERSION_STRING[0..len]);
    return len;
}

// ============================================================================
// WASM-native Key-Value Cache
//
// A simple open-addressing hash map cache suitable for wasm32. Avoids
// 64-bit atomics and POSIX dependencies present in the main cache module.
// ============================================================================

const CACHE_BUCKETS = 512;
const CACHE_MAX_ENTRIES = 256;

const CacheEntry = struct {
    key: ?[]const u8 = null,
    value: ?[]const u8 = null,
    key_buf: ?[]u8 = null,
    value_buf: ?[]u8 = null,
    occupied: bool = false,
};

var cache_entries: [CACHE_BUCKETS]CacheEntry = [_]CacheEntry{.{}} ** CACHE_BUCKETS;
var cache_count: u32 = 0;
var cache_initialized: bool = false;

fn cacheHash(key: []const u8) usize {
    // FNV-1a hash
    var h: u32 = 2166136261;
    for (key) |byte| {
        h ^= byte;
        h *%= 16777619;
    }
    return h % CACHE_BUCKETS;
}

fn cacheFindSlot(key: []const u8) ?usize {
    const start = cacheHash(key);
    var i: usize = 0;
    while (i < CACHE_BUCKETS) : (i += 1) {
        const idx = (start + i) % CACHE_BUCKETS;
        const entry = &cache_entries[idx];
        if (!entry.occupied) return null;
        if (entry.key) |k| {
            if (std.mem.eql(u8, k, key)) return idx;
        }
    }
    return null;
}

fn cacheFindOrInsertSlot(key: []const u8) ?usize {
    const start = cacheHash(key);
    var first_empty: ?usize = null;
    var i: usize = 0;
    while (i < CACHE_BUCKETS) : (i += 1) {
        const idx = (start + i) % CACHE_BUCKETS;
        const entry = &cache_entries[idx];
        if (!entry.occupied) {
            if (first_empty == null) first_empty = idx;
            return first_empty;
        }
        if (entry.key) |k| {
            if (std.mem.eql(u8, k, key)) return idx;
        }
    }
    return first_empty;
}

fn cacheFreeBufs(entry: *CacheEntry) void {
    const allocator = wasmAllocator();
    if (entry.key_buf) |kb| {
        allocator.free(kb);
        entry.key_buf = null;
        entry.key = null;
    }
    if (entry.value_buf) |vb| {
        allocator.free(vb);
        entry.value_buf = null;
        entry.value = null;
    }
}

/// Initialize the WASM cache.
/// Returns 0 on success.
export fn cache_init() i32 {
    if (cache_initialized) return 0;
    cache_entries = [_]CacheEntry{.{}} ** CACHE_BUCKETS;
    cache_count = 0;
    cache_initialized = true;
    return 0;
}

/// Tear down the cache, freeing all entries.
export fn cache_deinit() void {
    for (&cache_entries) |*entry| {
        if (entry.occupied) cacheFreeBufs(entry);
        entry.occupied = false;
    }
    cache_count = 0;
    cache_initialized = false;
}

/// Insert a key-value pair into the cache.
/// Returns 0 on success, -1 on failure.
export fn cache_put(
    key_ptr: [*]const u8,
    key_len: usize,
    val_ptr: [*]const u8,
    val_len: usize,
) i32 {
    if (!cache_initialized) return -1;
    const key = key_ptr[0..key_len];
    const val = val_ptr[0..val_len];
    const allocator = wasmAllocator();

    const slot = cacheFindOrInsertSlot(key) orelse return -1;
    var entry = &cache_entries[slot];

    if (entry.occupied) {
        // Update existing — free old value
        if (entry.value_buf) |vb| allocator.free(vb);
        entry.value_buf = allocator.dupe(u8, val) catch return -1;
        entry.value = entry.value_buf;
        return 0;
    }

    if (cache_count >= CACHE_MAX_ENTRIES) return -1;

    // New entry
    entry.key_buf = allocator.dupe(u8, key) catch return -1;
    entry.key = entry.key_buf;
    entry.value_buf = allocator.dupe(u8, val) catch {
        if (entry.key_buf) |kb| allocator.free(kb);
        entry.key_buf = null;
        entry.key = null;
        return -1;
    };
    entry.value = entry.value_buf;
    entry.occupied = true;
    cache_count += 1;
    return 0;
}

/// Look up a cached value. Copies the value into `out_ptr[0..out_max]`.
/// Returns the number of bytes written, 0 if the key was not found,
/// or -1 on error.
export fn cache_get(
    key_ptr: [*]const u8,
    key_len: usize,
    out_ptr: [*]u8,
    out_max: usize,
) i32 {
    if (!cache_initialized) return -1;
    const key = key_ptr[0..key_len];
    const slot = cacheFindSlot(key) orelse return 0;
    const entry = &cache_entries[slot];
    const val = entry.value orelse return 0;
    if (val.len > out_max) return -1;
    @memcpy(out_ptr[0..val.len], val);
    return @intCast(val.len);
}

/// Return the current number of live cache entries.
export fn cache_size() u32 {
    return cache_count;
}

/// Check whether a key exists in the cache.
/// Returns 1 if present, 0 if absent.
export fn cache_contains(key_ptr: [*]const u8, key_len: usize) i32 {
    if (!cache_initialized) return 0;
    return if (cacheFindSlot(key_ptr[0..key_len]) != null) 1 else 0;
}

/// Delete a cache entry by key.
/// Returns 1 if the key was present and removed, 0 if absent, -1 on error.
export fn cache_delete(key_ptr: [*]const u8, key_len: usize) i32 {
    if (!cache_initialized) return -1;
    const slot = cacheFindSlot(key_ptr[0..key_len]) orelse return 0;
    var entry = &cache_entries[slot];
    cacheFreeBufs(entry);
    entry.occupied = false;
    cache_count -|= 1;
    return 1;
}

/// Remove all entries from the cache.
export fn cache_clear() void {
    for (&cache_entries) |*entry| {
        if (entry.occupied) cacheFreeBufs(entry);
        entry.occupied = false;
    }
    cache_count = 0;
}

// ============================================================================
// WASM-native Analytics Engine
//
// Tracks event names in a ring buffer. No 64-bit atomics needed.
// ============================================================================

const ANALYTICS_BUFFER_CAP = 256;

const AnalyticsEvent = struct {
    name: ?[]const u8 = null,
};

var analytics_buffer: [ANALYTICS_BUFFER_CAP]AnalyticsEvent = [_]AnalyticsEvent{.{}} ** ANALYTICS_BUFFER_CAP;
var analytics_head: u32 = 0;
var analytics_count: u32 = 0;
var analytics_total: u32 = 0;
var analytics_active: bool = false;

/// Create and initialize the analytics engine.
/// Returns 0 on success, -1 if already created.
export fn analytics_create() i32 {
    if (analytics_active) return -1;
    analytics_buffer = [_]AnalyticsEvent{.{}} ** ANALYTICS_BUFFER_CAP;
    analytics_head = 0;
    analytics_count = 0;
    analytics_total = 0;
    analytics_active = true;
    return 0;
}

/// Track a named analytics event.
/// Returns 0 on success, -1 on failure.
export fn analytics_track(event_ptr: [*]const u8, event_len: usize) i32 {
    if (!analytics_active) return -1;
    if (analytics_count >= ANALYTICS_BUFFER_CAP) return -1;
    const idx = (analytics_head + analytics_count) % ANALYTICS_BUFFER_CAP;
    analytics_buffer[idx] = .{ .name = event_ptr[0..event_len] };
    analytics_count += 1;
    analytics_total +|= 1;
    return 0;
}

/// Flush buffered analytics events.
/// Returns the number of events flushed, or -1 if no engine.
export fn analytics_flush() i32 {
    if (!analytics_active) return -1;
    const count = analytics_count;
    analytics_head = (analytics_head + count) % ANALYTICS_BUFFER_CAP;
    analytics_count = 0;
    return @intCast(count);
}

/// Get total events tracked (including flushed).
/// Returns count, or 0 if no engine.
export fn analytics_total_events() u32 {
    return analytics_total;
}

/// Get number of buffered (unflushed) events.
export fn analytics_buffered_count() u32 {
    return analytics_count;
}

/// Destroy the analytics engine.
export fn analytics_destroy() void {
    analytics_active = false;
    analytics_count = 0;
    analytics_total = 0;
    analytics_head = 0;
}

// ============================================================================
// WASM-native Search
//
// Minimal keyword search using a flat document store with linear scan.
// Suitable for small corpora in WASM environments.
// ============================================================================

const SEARCH_MAX_DOCS = 64;

const SearchDoc = struct {
    id: ?[]const u8 = null,
    content: ?[]const u8 = null,
    occupied: bool = false,
};

var search_docs: [SEARCH_MAX_DOCS]SearchDoc = [_]SearchDoc{.{}} ** SEARCH_MAX_DOCS;
var search_num_docs: u32 = 0;
var search_active: bool = false;

/// Initialize the search subsystem.
/// Returns 0 on success, -1 on failure.
export fn search_init() i32 {
    if (search_active) return -1;
    search_docs = [_]SearchDoc{.{}} ** SEARCH_MAX_DOCS;
    search_num_docs = 0;
    search_active = true;
    return 0;
}

/// Tear down the search subsystem.
export fn search_deinit() void {
    search_docs = [_]SearchDoc{.{}} ** SEARCH_MAX_DOCS;
    search_num_docs = 0;
    search_active = false;
}

/// Index a document. The `idx_name` parameter is accepted for API
/// compatibility but ignored (single implicit index in WASM mode).
/// Returns 0 on success, -1 on failure.
export fn search_index_document(
    _: [*]const u8, // idx_name_ptr (ignored)
    _: usize, // idx_name_len
    doc_ptr: [*]const u8,
    doc_len: usize,
    content_ptr: [*]const u8,
    content_len: usize,
) i32 {
    if (!search_active) return -1;
    if (search_num_docs >= SEARCH_MAX_DOCS) return -1;

    // Find a free slot
    for (&search_docs) |*doc| {
        if (!doc.occupied) {
            doc.id = doc_ptr[0..doc_len];
            doc.content = content_ptr[0..content_len];
            doc.occupied = true;
            search_num_docs += 1;
            return 0;
        }
    }
    return -1;
}

/// Search for documents containing the query substring.
/// Returns the number of matching document IDs written to `out_ptr`
/// (as newline-separated strings), or -1 on error.
export fn search_query(
    query_ptr: [*]const u8,
    query_len: usize,
    out_ptr: [*]u8,
    out_max: usize,
) i32 {
    if (!search_active) return -1;
    const needle = query_ptr[0..query_len];
    var written: usize = 0;
    var matches: i32 = 0;

    for (&search_docs) |*doc| {
        if (!doc.occupied) continue;
        const content = doc.content orelse continue;
        if (std.mem.indexOf(u8, content, needle) != null) {
            const id = doc.id orelse continue;
            // Write "id\n" into output
            if (written + id.len + 1 > out_max) break;
            @memcpy(out_ptr[written..][0..id.len], id);
            out_ptr[written + id.len] = '\n';
            written += id.len + 1;
            matches += 1;
        }
    }
    return matches;
}

/// Return the number of indexed documents.
export fn search_doc_count() u32 {
    return search_num_docs;
}

// ============================================================================
// Panic handler (required for freestanding)
// ============================================================================

pub fn panic(msg: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = msg;
    @trap();
}

test {
    std.testing.refAllDecls(@This());
}
