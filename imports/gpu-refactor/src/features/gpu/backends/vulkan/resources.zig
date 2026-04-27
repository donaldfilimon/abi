//! Vulkan Resource Management
//!
//! Shader cache and command pool utilities for the Vulkan backend.

const std = @import("std");

// ============================================================================
// Shader Cache
// ============================================================================

/// LRU shader cache keyed by a hash of the shader source code.
///
/// Caches compiled VkShaderModule + VkPipeline handles to avoid
/// redundant compilations for the same shader source.
pub const ShaderCache = struct {
    entries: std.AutoHashMapUnmanaged(u64, CacheEntry) = .empty,
    allocator: std.mem.Allocator,
    max_entries: u32 = 64,
    hits: u64 = 0,
    misses: u64 = 0,

    pub const CacheEntry = struct {
        shader_module: u64, // VkShaderModule handle
        pipeline: u64, // VkPipeline handle
        last_used: i64, // timestamp (epoch seconds)
    };

    /// Look up a cached entry by source hash. Returns null on miss.
    /// Updates the last_used timestamp and hit/miss counters.
    pub fn lookup(self: *ShaderCache, source_hash: u64) ?CacheEntry {
        if (self.entries.getPtr(source_hash)) |entry| {
            self.hits += 1;
            // Update last_used timestamp
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.REALTIME, &ts);
            entry.last_used = @intCast(ts.sec);
            return entry.*;
        }
        self.misses += 1;
        return null;
    }

    /// Insert a new cache entry. Evicts the oldest entry if at capacity.
    pub fn insert(self: *ShaderCache, source_hash: u64, entry: CacheEntry) !void {
        if (!self.entries.contains(source_hash)) {
            if (self.entries.count() >= self.max_entries) {
                self.evictOldest();
            }
        }
        try self.entries.put(self.allocator, source_hash, entry);
    }

    /// Evict the entry with the oldest last_used timestamp.
    pub fn evictOldest(self: *ShaderCache) void {
        if (self.entries.count() == 0) return;

        var oldest_key: u64 = 0;
        var oldest_time: i64 = std.math.maxInt(i64);
        var found = false;

        var it = self.entries.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.last_used < oldest_time) {
                oldest_time = kv.value_ptr.last_used;
                oldest_key = kv.key_ptr.*;
                found = true;
            }
        }

        if (found) {
            _ = self.entries.fetchRemove(oldest_key);
        }
    }

    /// Release all cache storage. Does NOT destroy the Vulkan handles
    /// (caller is responsible for that).
    pub fn deinit(self: *ShaderCache) void {
        self.entries.deinit(self.allocator);
    }
};

// ============================================================================
// Command Pool
// ============================================================================

/// Pool of reusable VkCommandBuffer handles to avoid allocation overhead.
///
/// When a command buffer is no longer in flight, call `release` to return
/// it to the pool. The next `acquire` will re-use it instead of asking
/// Vulkan for a fresh allocation.
pub const CommandPool = struct {
    free_buffers: std.ArrayList(u64) = .empty,
    active_count: u32 = 0,
    total_allocated: u32 = 0,
    allocator: std.mem.Allocator,

    /// Acquire a command buffer handle. Re-uses a previously released one
    /// if available; otherwise mints a new pseudo-handle.
    pub fn acquire(self: *CommandPool) !u64 {
        if (self.free_buffers.items.len > 0) {
            const handle = self.free_buffers.pop();
            self.active_count += 1;
            return handle;
        }

        self.total_allocated += 1;
        self.active_count += 1;

        var buf: [8]u8 = undefined;
        std.c.arc4random_buf(&buf, buf.len);
        return std.mem.readInt(u64, &buf, .little);
    }

    /// Return a command buffer to the free list for later re-use.
    pub fn release(self: *CommandPool, buffer: u64) void {
        if (self.active_count > 0) {
            self.active_count -= 1;
        }
        self.free_buffers.append(self.allocator, buffer) catch {
            // If we can't track it, it's simply leaked — acceptable
            // for a best-effort pool.
        };
    }

    /// Free the pool storage. Does NOT call vkFreeCommandBuffers.
    pub fn deinit(self: *CommandPool) void {
        self.free_buffers.deinit(self.allocator);
    }
};
