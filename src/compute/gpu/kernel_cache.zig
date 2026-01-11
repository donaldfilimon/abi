//! GPU kernel cache.
//!
//! Provides caching for compiled GPU kernels to avoid
//! redundant compilation and improve startup time.

const std = @import("std");

/// Kernel source type.
pub const KernelSourceType = enum {
    glsl,
    spirv,
    hlsl,
    metal,
    cuda,
    opencl,
    wgsl,
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
    entry_point: [64]u8 = [_]u8{0} ** 64,
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
    max_cache_size: usize = 256 * 1024 * 1024,
    /// Maximum number of entries.
    max_entries: usize = 1000,
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
    config: KernelCacheConfig,
    entries: std.StringHashMapUnmanaged(CacheEntry),
    lru_order: std.ArrayListUnmanaged([]const u8),
    current_size: usize,
    mutex: std.Thread.Mutex,
    stats: CacheStats,

    const CacheEntry = struct {
        meta: CacheEntryMeta,
        binary: []u8,
        access_count: u64,
        last_access: i64,
    };

    /// Initialize the kernel cache.
    pub fn init(allocator: std.mem.Allocator, config: KernelCacheConfig) KernelCache {
        return .{
            .allocator = allocator,
            .config = config,
            .entries = .{},
            .lru_order = .{},
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
        }
        self.entries.deinit(self.allocator);

        for (self.lru_order.items) |key| {
            _ = key; // Keys already freed above
        }
        self.lru_order.deinit(self.allocator);
        self.* = undefined;
    }

    /// Get a cached kernel or compile and cache it.
    pub fn getOrCompile(
        self: *KernelCache,
        source: []const u8,
        source_type: KernelSourceType,
        entry_point: []const u8,
        options: CompileOptions,
        compiler: *const fn ([]const u8, KernelSourceType, CompileOptions) anyerror![]u8,
    ) !CachedKernel {
        const key = try self.computeKey(source, source_type, entry_point, options);
        defer self.allocator.free(key);

        // Check cache
        self.mutex.lock();
        if (self.entries.get(key)) |*entry| {
            entry.access_count += 1;
            entry.last_access = std.time.timestamp();
            self.stats.hits += 1;
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

        var entry_point_buf: [64]u8 = [_]u8{0} ** 64;
        const ep_len = @min(entry_point.len, 64);
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
            .last_access = std.time.timestamp(),
        };

        // Insert into cache
        self.mutex.lock();
        defer self.mutex.unlock();

        // Evict if necessary
        try self.evictIfNeeded(binary.len);

        try self.entries.put(self.allocator, owned_key, entry);
        try self.lru_order.append(self.allocator, owned_key);
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
            entry.last_access = std.time.timestamp();
            self.stats.hits += 1;

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

        var entry_point_buf: [64]u8 = [_]u8{0} ** 64;
        const ep_len = @min(entry_point.len, 64);
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
            .last_access = std.time.timestamp(),
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.evictIfNeeded(binary.len);

        if (self.entries.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value.binary);
            self.current_size -= removed.value.meta.binary_size;
        }

        try self.entries.put(self.allocator, key, entry);
        try self.lru_order.append(self.allocator, key);
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
        }
        self.entries.clearRetainingCapacity();
        self.lru_order.clearRetainingCapacity();
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

        // In a real implementation, serialize entries to disk
        _ = cache_dir;
    }

    /// Load cache from disk.
    pub fn loadFromDisk(self: *KernelCache) !void {
        if (!self.config.enable_persistence) return;
        const cache_dir = self.config.cache_dir orelse return;

        self.mutex.lock();
        defer self.mutex.unlock();

        // In a real implementation, deserialize entries from disk
        _ = cache_dir;
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
        while (self.entries.count() >= self.config.max_entries and self.lru_order.items.len > 0) {
            try self.evictLru();
        }

        // Check size
        while (self.current_size + new_size > self.config.max_cache_size and self.lru_order.items.len > 0) {
            try self.evictLru();
        }
    }

    fn evictLru(self: *KernelCache) !void {
        if (self.lru_order.items.len == 0) return;

        // Simple LRU - remove oldest
        const key = self.lru_order.orderedRemove(0);

        if (self.entries.fetchRemove(key)) |removed| {
            self.current_size -= removed.value.meta.binary_size;
            self.allocator.free(removed.value.binary);
            self.allocator.free(removed.key);
            self.stats.evictions += 1;
        }
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
    var cache = KernelCache.init(allocator, .{});
    defer cache.deinit();

    const dummy_compiler = struct {
        fn compile(source: []const u8, _: KernelSourceType, _: CompileOptions) anyerror![]u8 {
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
    var cache = KernelCache.init(allocator, .{
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
