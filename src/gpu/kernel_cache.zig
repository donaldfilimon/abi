//! GPU kernel cache.
//!
//! Provides caching for compiled GPU kernels to avoid
//! redundant compilation and improve startup time.
//!
//! Disk persistence functions (persistToDisk, loadFromDisk)
//! use std.Io.Dir.cwd() with proper I/O context for full Zig 0.16 compliance.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");

// ============================================================================
// Constants
// ============================================================================

/// Maximum length for cache keys (filesystem-safe identifiers).
pub const MAX_CACHE_KEY_LEN: usize = 200;

/// Maximum length for kernel entry point names.
pub const ENTRY_POINT_MAX_LEN: usize = 64;

/// Default maximum cache size in bytes (256 MB).
pub const DEFAULT_MAX_CACHE_SIZE: usize = 256 * 1024 * 1024;

/// Default maximum number of cache entries.
pub const DEFAULT_MAX_ENTRIES: usize = 1000;

/// Default filename buffer size for cache file paths.
pub const FILENAME_BUF_SIZE: usize = 256;

/// Kernel compilation errors
pub const KernelError = std.mem.Allocator.Error || error{
    CompilationFailed,
    InvalidSource,
    UnsupportedSourceType,
    InvalidOptions,
    CacheCorrupted,
    InvalidCacheKey,
};

/// Validate a cache key for safe filesystem use.
/// Rejects keys containing path traversal sequences or unsafe characters.
fn isValidCacheKey(key: []const u8) bool {
    if (key.len == 0 or key.len > MAX_CACHE_KEY_LEN) return false;

    // Reject path traversal attempts
    if (std.mem.indexOf(u8, key, "..") != null) return false;
    if (std.mem.indexOf(u8, key, "/") != null) return false;
    if (std.mem.indexOf(u8, key, "\\") != null) return false;

    // Reject Windows drive letters (e.g., "C:")
    if (key.len >= 2 and key[1] == ':') return false;

    // Only allow alphanumeric, underscore, hyphen, and dot
    for (key) |c| {
        const valid = std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.';
        if (!valid) return false;
    }

    return true;
}

/// Kernel source type.
pub const KernelSourceType = enum {
    glsl,
    spirv,
    hlsl,
    metal,
    cuda,
    opencl,
    wgsl,
    /// Portable kernel DSL IR (auto-compiles to target backend).
    dsl_ir,
};

/// Kernel compilation options.
pub const CompileOptions = struct {
    /// Target architecture.
    target_arch: []const u8 = "",
    /// Optimization level (0-3).
    opt_level: u8 = 2,
    /// Enable debug info.
    debug_info: bool = false,
    /// Preprocessor defines.
    defines: []const []const u8 = &.{},
    /// Include paths.
    include_paths: []const []const u8 = &.{},
};

/// Cache entry metadata.
pub const CacheEntryMeta = struct {
    source_hash: u64,
    options_hash: u64,
    compile_time_ns: i64,
    binary_size: usize,
    source_type: KernelSourceType,
    entry_point: [ENTRY_POINT_MAX_LEN]u8 = [_]u8{0} ** ENTRY_POINT_MAX_LEN,
    entry_point_len: usize = 0,

    pub fn getEntryPoint(self: *const CacheEntryMeta) []const u8 {
        return self.entry_point[0..self.entry_point_len];
    }
};

/// Cached kernel binary.
pub const CachedKernel = struct {
    meta: CacheEntryMeta,
    binary: []const u8,
};

/// Kernel cache configuration.
pub const KernelCacheConfig = struct {
    /// Maximum cache size in bytes.
    max_cache_size: usize = DEFAULT_MAX_CACHE_SIZE,
    /// Maximum number of entries.
    max_entries: usize = DEFAULT_MAX_ENTRIES,
    /// Enable disk persistence.
    enable_persistence: bool = true,
    /// Cache directory path.
    cache_dir: ?[]const u8 = null,
    /// Enable LRU eviction.
    enable_lru: bool = true,
};

/// Kernel cache for compiled GPU kernels.
pub const KernelCache = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: KernelCacheConfig,
    entries: std.StringHashMapUnmanaged(CacheEntry),
    lru_head: ?*LruNode,
    lru_tail: ?*LruNode,
    current_size: usize,
    mutex: std.Thread.Mutex,
    stats: CacheStats,

    /// LRU doubly-linked list node for O(1) eviction
    const LruNode = struct {
        key: []const u8,
        prev: ?*LruNode,
        next: ?*LruNode,
    };

    const CacheEntry = struct {
        meta: CacheEntryMeta,
        binary: []u8,
        access_count: u64,
        last_access: i64,
        lru_node: *LruNode,
    };

    /// Initialize the kernel cache.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, config: KernelCacheConfig) KernelCache {
        return .{
            .allocator = allocator,
            .io = io,
            .config = config,
            .entries = .{},
            .lru_head = null,
            .lru_tail = null,
            .current_size = 0,
            .mutex = .{},
            .stats = .{},
        };
    }

    /// Deinitialize the cache.
    pub fn deinit(self: *KernelCache) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.binary);
            self.allocator.destroy(entry.value_ptr.lru_node);
        }
        self.entries.deinit(self.allocator);
        self.lru_head = null;
        self.lru_tail = null;
        self.* = undefined;
    }

    /// Get a cached kernel or compile and cache it.
    pub fn getOrCompile(
        self: *KernelCache,
        source: []const u8,
        source_type: KernelSourceType,
        entry_point: []const u8,
        options: CompileOptions,
        compiler: *const fn ([]const u8, KernelSourceType, CompileOptions) KernelError![]u8,
    ) !CachedKernel {
        const key = try self.computeKey(source, source_type, entry_point, options);
        defer self.allocator.free(key);

        // Check cache
        self.mutex.lock();
        if (self.entries.getPtr(key)) |entry| {
            entry.access_count += 1;
            entry.last_access = time.unixSeconds();
            self.stats.hits += 1;
            // Move to front of LRU (most recently used)
            self.moveToFront(entry.lru_node);
            self.mutex.unlock();

            return CachedKernel{
                .meta = entry.meta,
                .binary = entry.binary,
            };
        }
        self.mutex.unlock();

        // Cache miss - compile
        self.stats.misses += 1;

        var timer = std.time.Timer.start() catch return error.TimerFailed;
        const binary = try compiler(source, source_type, options);
        const compile_time = timer.read();

        // Create entry
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);

        // Create LRU node
        const lru_node = try self.allocator.create(LruNode);
        errdefer self.allocator.destroy(lru_node);
        lru_node.* = .{ .key = owned_key, .prev = null, .next = null };

        var entry_point_buf: [ENTRY_POINT_MAX_LEN]u8 = [_]u8{0} ** ENTRY_POINT_MAX_LEN;
        const ep_len = @min(entry_point.len, ENTRY_POINT_MAX_LEN);
        @memcpy(entry_point_buf[0..ep_len], entry_point[0..ep_len]);

        const entry = CacheEntry{
            .meta = .{
                .source_hash = std.hash.Wyhash.hash(0, source),
                .options_hash = self.hashOptions(options),
                .compile_time_ns = @intCast(compile_time),
                .binary_size = binary.len,
                .source_type = source_type,
                .entry_point = entry_point_buf,
                .entry_point_len = ep_len,
            },
            .binary = binary,
            .access_count = 1,
            .last_access = time.unixSeconds(),
            .lru_node = lru_node,
        };

        // Insert into cache
        self.mutex.lock();
        defer self.mutex.unlock();

        // Evict if necessary
        try self.evictIfNeeded(binary.len);

        try self.entries.put(self.allocator, owned_key, entry);
        self.addToFront(lru_node);
        self.current_size += binary.len;

        return CachedKernel{
            .meta = entry.meta,
            .binary = entry.binary,
        };
    }

    /// Get a cached kernel by key.
    pub fn get(self: *KernelCache, source: []const u8, source_type: KernelSourceType, entry_point: []const u8, options: CompileOptions) ?CachedKernel {
        const key = self.computeKey(source, source_type, entry_point, options) catch return null;
        defer self.allocator.free(key);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.getPtr(key)) |entry| {
            entry.access_count += 1;
            entry.last_access = time.unixSeconds();
            self.stats.hits += 1;
            // Move to front of LRU (most recently used)
            self.moveToFront(entry.lru_node);

            return CachedKernel{
                .meta = entry.meta,
                .binary = entry.binary,
            };
        }

        self.stats.misses += 1;
        return null;
    }

    /// Put a pre-compiled kernel in the cache.
    pub fn put(
        self: *KernelCache,
        source: []const u8,
        source_type: KernelSourceType,
        entry_point: []const u8,
        options: CompileOptions,
        binary: []const u8,
    ) !void {
        const key = try self.computeKey(source, source_type, entry_point, options);
        errdefer self.allocator.free(key);

        const owned_binary = try self.allocator.dupe(u8, binary);
        errdefer self.allocator.free(owned_binary);

        // Create LRU node
        const lru_node = try self.allocator.create(LruNode);
        errdefer self.allocator.destroy(lru_node);
        lru_node.* = .{ .key = key, .prev = null, .next = null };

        var entry_point_buf: [ENTRY_POINT_MAX_LEN]u8 = [_]u8{0} ** ENTRY_POINT_MAX_LEN;
        const ep_len = @min(entry_point.len, ENTRY_POINT_MAX_LEN);
        @memcpy(entry_point_buf[0..ep_len], entry_point[0..ep_len]);

        const entry = CacheEntry{
            .meta = .{
                .source_hash = std.hash.Wyhash.hash(0, source),
                .options_hash = self.hashOptions(options),
                .compile_time_ns = 0,
                .binary_size = binary.len,
                .source_type = source_type,
                .entry_point = entry_point_buf,
                .entry_point_len = ep_len,
            },
            .binary = owned_binary,
            .access_count = 0,
            .last_access = time.unixSeconds(),
            .lru_node = lru_node,
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.evictIfNeeded(binary.len);

        if (self.entries.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value.binary);
            self.removeFromLru(removed.value.lru_node);
            self.allocator.destroy(removed.value.lru_node);
            self.current_size -= removed.value.meta.binary_size;
        }

        try self.entries.put(self.allocator, key, entry);
        self.addToFront(lru_node);
        self.current_size += binary.len;
    }

    /// Remove a kernel from the cache.
    pub fn remove(
        self: *KernelCache,
        source: []const u8,
        source_type: KernelSourceType,
        entry_point: []const u8,
        options: CompileOptions,
    ) bool {
        const key = self.computeKey(source, source_type, entry_point, options) catch return false;
        defer self.allocator.free(key);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value.binary);
            self.removeFromLru(removed.value.lru_node);
            self.allocator.destroy(removed.value.lru_node);
            self.current_size -= removed.value.meta.binary_size;
            self.stats.evictions += 1;
            return true;
        }
        return false;
    }

    /// Clear all cached kernels.
    pub fn clear(self: *KernelCache) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.binary);
            self.allocator.destroy(entry.value_ptr.lru_node);
        }
        self.entries.clearRetainingCapacity();
        self.lru_head = null;
        self.lru_tail = null;
        self.current_size = 0;
    }

    /// Get cache statistics.
    pub fn getStats(self: *const KernelCache) CacheStats {
        return CacheStats{
            .hits = self.stats.hits,
            .misses = self.stats.misses,
            .evictions = self.stats.evictions,
            .entry_count = self.entries.count(),
            .total_size = self.current_size,
            .max_size = self.config.max_cache_size,
        };
    }

    /// Save cache to disk.
    pub fn persistToDisk(self: *KernelCache) !void {
        if (!self.config.enable_persistence) return;
        const cache_dir = self.config.cache_dir orelse return;

        self.mutex.lock();
        defer self.mutex.unlock();

        // Create cache directory if it doesn't exist (Zig 0.16 pattern)
        std.Io.Dir.cwd().createDirPath(self.io, cache_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        // Save each entry as a separate file
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const cache_entry = entry.value_ptr.*;

            // Validate key to prevent path traversal attacks
            if (!isValidCacheKey(key)) {
                std.log.warn("Skipping invalid cache key: {s}", .{key});
                continue;
            }

            // Construct filename from key
            var filename_buf: [FILENAME_BUF_SIZE]u8 = undefined;
            const filename = try std.fmt.bufPrint(&filename_buf, "{s}/{s}.bin", .{ cache_dir, key });

            // Write binary to file
            var file = try std.Io.Dir.cwd().createFile(self.io, filename, .{ .truncate = true });
            defer file.close(self.io);

            // Build cache entry data in buffer for Zig 0.16 compatibility
            var entry_buf = std.ArrayListUnmanaged(u8).empty;
            defer entry_buf.deinit(self.allocator);

            // Write metadata
            try entry_buf.appendSlice(self.allocator, std.mem.asBytes(&cache_entry.meta));

            // Write binary data
            try entry_buf.appendSlice(self.allocator, cache_entry.binary);

            // Write using writeStreamingAll for Zig 0.16 compatibility
            try file.writeStreamingAll(self.io, entry_buf.items);
        }
    }

    /// Load cache from disk.
    pub fn loadFromDisk(self: *KernelCache) !void {
        if (!self.config.enable_persistence) return;
        const cache_dir = self.config.cache_dir orelse return;

        self.mutex.lock();
        defer self.mutex.unlock();

        // Open cache directory (Zig 0.16 pattern)
        var dir = std.Io.Dir.cwd().openDir(self.io, cache_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return;
            return err;
        };
        defer dir.close(self.io);

        // Iterate over .bin files
        var iter = dir.iterate(self.io);
        while (try iter.next(self.io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".bin")) continue;

            // Read file
            var file = try dir.openFile(self.io, entry.name, .{});
            defer file.close(self.io);

            var reader = file.reader(self.io);

            // Read metadata
            var meta: CacheEntryMeta = undefined;
            const meta_bytes = try reader.readAll(std.mem.asBytes(&meta));
            if (meta_bytes != @sizeOf(CacheEntryMeta)) continue;

            // Read binary data
            const binary_size = (try file.stat(self.io)).size - @sizeOf(CacheEntryMeta);
            const binary = try self.allocator.alloc(u8, binary_size);
            errdefer self.allocator.free(binary);

            const read_bytes = try reader.readAll(binary);
            if (read_bytes != binary_size) {
                self.allocator.free(binary);
                continue;
            }

            // Extract key from filename (remove .bin extension)
            const key_len = entry.name.len - 4;
            const key_slice = entry.name[0..key_len];

            // Validate key to ensure it's safe
            if (!isValidCacheKey(key_slice)) {
                std.log.warn("Skipping cache file with invalid key: {s}", .{entry.name});
                self.allocator.free(binary);
                continue;
            }

            const key = try self.allocator.dupe(u8, key_slice);
            errdefer self.allocator.free(key);

            // Create LRU node
            const lru_node = try self.allocator.create(LruNode);
            errdefer self.allocator.destroy(lru_node);
            lru_node.* = .{ .key = key, .prev = null, .next = null };

            // Add to cache
            const cache_entry = CacheEntry{
                .meta = meta,
                .binary = binary,
                .access_count = 0,
                .last_access = time.unixSeconds(),
                .lru_node = lru_node,
            };

            try self.entries.put(self.allocator, key, cache_entry);
            self.addToFront(lru_node);
            self.current_size += binary.len;
        }
    }

    // Internal helpers
    fn computeKey(
        self: *KernelCache,
        source: []const u8,
        source_type: KernelSourceType,
        entry_point: []const u8,
        options: CompileOptions,
    ) ![]u8 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(source);
        hasher.update(std.mem.asBytes(&source_type));
        hasher.update(entry_point);
        hasher.update(std.mem.asBytes(&self.hashOptions(options)));

        const hash = hasher.final();
        return std.fmt.allocPrint(self.allocator, "{x:0>16}", .{hash});
    }

    fn hashOptions(self: *const KernelCache, options: CompileOptions) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(options.target_arch);
        hasher.update(std.mem.asBytes(&options.opt_level));
        hasher.update(std.mem.asBytes(&options.debug_info));
        for (options.defines) |define| {
            hasher.update(define);
        }
        return hasher.final();
    }

    fn evictIfNeeded(self: *KernelCache, new_size: usize) !void {
        // Check entry count
        while (self.entries.count() >= self.config.max_entries and self.lru_tail != null) {
            self.evictLru();
        }

        // Check size
        while (self.current_size + new_size > self.config.max_cache_size and self.lru_tail != null) {
            self.evictLru();
        }
    }

    fn evictLru(self: *KernelCache) void {
        // Evict from tail (least recently used) - O(1) operation
        const tail = self.lru_tail orelse return;
        const key = tail.key;

        if (self.entries.fetchRemove(key)) |removed| {
            self.current_size -= removed.value.meta.binary_size;
            self.removeFromLru(removed.value.lru_node);
            self.allocator.free(removed.value.binary);
            self.allocator.destroy(removed.value.lru_node);
            self.allocator.free(removed.key);
            self.stats.evictions += 1;
        }
    }

    /// Add node to front of LRU list (most recently used) - O(1)
    fn addToFront(self: *KernelCache, node: *LruNode) void {
        node.prev = null;
        node.next = self.lru_head;

        if (self.lru_head) |head| {
            head.prev = node;
        }
        self.lru_head = node;

        if (self.lru_tail == null) {
            self.lru_tail = node;
        }
    }

    /// Remove node from LRU list - O(1)
    fn removeFromLru(self: *KernelCache, node: *LruNode) void {
        if (node.prev) |prev| {
            prev.next = node.next;
        } else {
            self.lru_head = node.next;
        }

        if (node.next) |next| {
            next.prev = node.prev;
        } else {
            self.lru_tail = node.prev;
        }

        node.prev = null;
        node.next = null;
    }

    /// Move node to front (most recently used) - O(1)
    fn moveToFront(self: *KernelCache, node: *LruNode) void {
        if (self.lru_head == node) return; // Already at front
        self.removeFromLru(node);
        self.addToFront(node);
    }
};

/// Cache statistics.
pub const CacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,
    entry_count: usize = 0,
    total_size: usize = 0,
    max_size: usize = 0,

    pub fn hitRate(self: CacheStats) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn utilizationPercent(self: CacheStats) f64 {
        if (self.max_size == 0) return 0;
        return @as(f64, @floatFromInt(self.total_size)) / @as(f64, @floatFromInt(self.max_size)) * 100.0;
    }
};

test "kernel cache basic" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var cache = KernelCache.init(allocator, io, .{});
    defer cache.deinit();

    const dummy_compiler = struct {
        fn compile(source: []const u8, _: KernelSourceType, _: CompileOptions) KernelError![]u8 {
            return std.testing.allocator.dupe(u8, source);
        }
    }.compile;

    // First call - miss, compile
    const result1 = try cache.getOrCompile(
        "kernel code",
        .glsl,
        "main",
        .{},
        dummy_compiler,
    );
    try std.testing.expectEqualStrings("kernel code", result1.binary);

    // Second call - hit
    const result2 = try cache.getOrCompile(
        "kernel code",
        .glsl,
        "main",
        .{},
        dummy_compiler,
    );
    try std.testing.expectEqualStrings("kernel code", result2.binary);

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
    try std.testing.expectEqual(@as(u64, 1), stats.misses);
}

test "kernel cache eviction" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var cache = KernelCache.init(allocator, io, .{
        .max_entries = 2,
        .max_cache_size = 1024,
    });
    defer cache.deinit();

    try cache.put("source1", .glsl, "main", .{}, "binary1");
    try cache.put("source2", .glsl, "main", .{}, "binary2");
    try cache.put("source3", .glsl, "main", .{}, "binary3"); // Should evict source1

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.entry_count);
}

test "cache stats" {
    var stats = CacheStats{
        .hits = 80,
        .misses = 20,
        .total_size = 512,
        .max_size = 1024,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 0.8), stats.hitRate(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), stats.utilizationPercent(), 0.001);
}
