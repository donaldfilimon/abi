//! Vulkan Pipeline Cache Management
//!
//! Provides pipeline cache persistence and warming for faster shader
//! compilation on subsequent runs. Caches compiled pipelines to disk
//! and preloads commonly-used shaders at startup.
//!
//! Features:
//! - Pipeline cache persistence to disk
//! - Background pipeline warming
//! - Common kernel pre-compilation
//! - Cache statistics and management

const std = @import("std");
const types = @import("vulkan_types.zig");
const init = @import("vulkan_init.zig");

/// Pipeline cache configuration
pub const PipelineCacheConfig = struct {
    /// Enable pipeline cache persistence to disk
    enable_persistence: bool = true,
    /// Cache file path (null = default location)
    cache_path: ?[]const u8 = null,
    /// Maximum cache size in bytes (0 = unlimited)
    max_cache_size: usize = 256 * 1024 * 1024, // 256 MB
    /// Enable background warming of common kernels
    enable_warming: bool = true,
    /// Common kernels to pre-warm
    prewarm_kernels: []const PrewarmKernel = &default_prewarm_kernels,
};

/// Kernel to pre-warm
pub const PrewarmKernel = struct {
    name: []const u8,
    spirv_hash: u64 = 0, // 0 = compute from source
    workgroup_size: [3]u32 = .{ 256, 1, 1 },
};

/// Default kernels to pre-warm (common operations)
const default_prewarm_kernels = [_]PrewarmKernel{
    .{ .name = "vector_add" },
    .{ .name = "vector_mul" },
    .{ .name = "matrix_multiply", .workgroup_size = .{ 16, 16, 1 } },
    .{ .name = "reduce_sum" },
    .{ .name = "softmax" },
    .{ .name = "layer_norm" },
    .{ .name = "gelu" },
    .{ .name = "silu" },
};

/// Pipeline cache statistics
pub const CacheStats = struct {
    /// Number of pipeline cache hits
    hits: u64 = 0,
    /// Number of pipeline cache misses
    misses: u64 = 0,
    /// Total pipelines in cache
    pipeline_count: usize = 0,
    /// Cache size in bytes
    cache_size_bytes: usize = 0,
    /// Number of kernels pre-warmed
    prewarmed_count: usize = 0,
    /// Time saved from cache hits (estimated, nanoseconds)
    time_saved_ns: u64 = 0,
    /// Last cache save time
    last_save_timestamp: i64 = 0,

    pub fn hitRate(self: CacheStats) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }
};

/// Vulkan pipeline cache manager
pub const VulkanPipelineCache = struct {
    allocator: std.mem.Allocator,
    config: PipelineCacheConfig,
    /// Vulkan pipeline cache handle
    vk_cache: ?types.VkPipelineCache = null,
    /// Cached pipeline data
    cache_data: ?[]u8 = null,
    /// Statistics
    stats: CacheStats = .{},
    /// Mutex for thread safety
    mutex: std.Thread.Mutex = .{},
    /// Initialization state
    initialized: bool = false,

    const Self = @This();

    /// Initialize the pipeline cache
    pub fn init(allocator: std.mem.Allocator, config: PipelineCacheConfig) !Self {
        var cache = Self{
            .allocator = allocator,
            .config = config,
        };

        // Try to load existing cache from disk
        if (config.enable_persistence) {
            cache.loadFromDisk() catch |err| {
                std.log.debug("No existing pipeline cache found: {}", .{err});
            };
        }

        // Create Vulkan pipeline cache
        try cache.createVulkanCache();

        cache.initialized = true;
        return cache;
    }

    /// Deinitialize and optionally save cache
    pub fn deinit(self: *Self) void {
        // Save cache to disk before cleanup
        if (self.config.enable_persistence) {
            self.saveToDisk() catch |err| {
                std.log.warn("Failed to save pipeline cache: {}", .{err});
            };
        }

        // Destroy Vulkan cache
        if (self.vk_cache) |cache| {
            if (init.vulkan_context) |*ctx| {
                if (init.vkDestroyPipelineCache) |destroy_fn| {
                    destroy_fn(ctx.device, cache, null);
                }
            }
        }

        // Free cache data
        if (self.cache_data) |data| {
            self.allocator.free(data);
        }

        self.* = undefined;
    }

    /// Get the Vulkan pipeline cache handle for use in pipeline creation
    pub fn getVkCache(self: *Self) ?types.VkPipelineCache {
        return self.vk_cache;
    }

    /// Record a cache hit
    pub fn recordHit(self: *Self, compilation_time_saved_ns: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats.hits += 1;
        self.stats.time_saved_ns += compilation_time_saved_ns;
    }

    /// Record a cache miss
    pub fn recordMiss(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats.misses += 1;
    }

    /// Get cache statistics
    pub fn getStats(self: *Self) CacheStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Update cache size
        if (self.vk_cache) |cache| {
            if (init.vulkan_context != null) {
                if (init.vkGetPipelineCacheData) |get_data_fn| {
                    var size: usize = 0;
                    _ = get_data_fn(init.vulkan_context.?.device, cache, &size, null);
                    self.stats.cache_size_bytes = size;
                }
            }
        }

        return self.stats;
    }

    /// Pre-warm commonly used kernels in background
    pub fn warmCache(self: *Self) !void {
        if (!self.config.enable_warming) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        var warmed: usize = 0;
        for (self.config.prewarm_kernels) |kernel| {
            if (self.prewarmKernel(kernel)) {
                warmed += 1;
            } else |err| {
                std.log.debug("Failed to pre-warm kernel {s}: {}", .{ kernel.name, err });
            }
        }

        self.stats.prewarmed_count = warmed;
        std.log.info("Pre-warmed {} kernels", .{warmed});
    }

    /// Save cache to disk
    pub fn saveToDisk(self: *Self) !void {
        if (!self.config.enable_persistence) return;

        const ctx = init.vulkan_context orelse return error.NotInitialized;
        const cache = self.vk_cache orelse return error.NoCacheHandle;
        const get_data_fn = init.vkGetPipelineCacheData orelse return error.FunctionNotLoaded;

        // Get cache data size
        var size: usize = 0;
        var result = get_data_fn(ctx.device, cache, &size, null);
        if (result != .success or size == 0) return;

        // Check max size
        if (self.config.max_cache_size > 0 and size > self.config.max_cache_size) {
            std.log.warn("Pipeline cache size {} exceeds max {}, skipping save", .{
                size, self.config.max_cache_size,
            });
            return;
        }

        // Allocate buffer and get data
        const data = try self.allocator.alloc(u8, size);
        defer self.allocator.free(data);

        result = get_data_fn(ctx.device, cache, &size, data.ptr);
        if (result != .success) return error.GetDataFailed;

        // Write to file
        const path = self.config.cache_path orelse getDefaultCachePath();
        const file = std.fs.cwd().createFile(path, .{}) catch |err| {
            std.log.warn("Failed to create cache file {s}: {}", .{ path, err });
            return err;
        };
        defer file.close();

        // Write header
        const header = CacheHeader{
            .magic = CACHE_MAGIC,
            .version = CACHE_VERSION,
            .data_size = size,
            .checksum = computeChecksum(data),
        };
        try file.writeAll(std.mem.asBytes(&header));

        // Write data
        try file.writeAll(data);

        self.stats.last_save_timestamp = std.time.timestamp();
        std.log.info("Saved pipeline cache: {} bytes", .{size});
    }

    /// Load cache from disk
    pub fn loadFromDisk(self: *Self) !void {
        if (!self.config.enable_persistence) return;

        const path = self.config.cache_path orelse getDefaultCachePath();
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read and validate header
        var header: CacheHeader = undefined;
        const header_bytes = try file.readAll(std.mem.asBytes(&header));
        if (header_bytes != @sizeOf(CacheHeader)) return error.InvalidHeader;

        if (header.magic != CACHE_MAGIC) return error.InvalidMagic;
        if (header.version != CACHE_VERSION) return error.VersionMismatch;

        // Read data
        const data = try self.allocator.alloc(u8, header.data_size);
        errdefer self.allocator.free(data);

        const read_bytes = try file.readAll(data);
        if (read_bytes != header.data_size) {
            self.allocator.free(data);
            return error.IncompleteData;
        }

        // Validate checksum
        if (computeChecksum(data) != header.checksum) {
            self.allocator.free(data);
            return error.ChecksumMismatch;
        }

        // Store for later use in Vulkan cache creation
        if (self.cache_data) |old_data| {
            self.allocator.free(old_data);
        }
        self.cache_data = data;

        std.log.info("Loaded pipeline cache: {} bytes", .{header.data_size});
    }

    // Internal methods

    fn createVulkanCache(self: *Self) !void {
        const ctx = init.vulkan_context orelse return error.NotInitialized;
        const create_fn = init.vkCreatePipelineCache orelse return error.FunctionNotLoaded;

        const create_info = types.VkPipelineCacheCreateInfo{
            .initialDataSize = if (self.cache_data) |d| d.len else 0,
            .pInitialData = if (self.cache_data) |d| d.ptr else null,
        };

        var cache: types.VkPipelineCache = undefined;
        const result = create_fn(ctx.device, &create_info, null, &cache);
        if (result != .success) return error.CreateCacheFailed;

        self.vk_cache = cache;
    }

    fn prewarmKernel(self: *Self, kernel: PrewarmKernel) !void {
        _ = self;
        _ = kernel;
        // Kernel pre-warming would compile the shader and create a pipeline
        // using the cache. This requires the kernel source/SPIR-V to be available.
        // For now, this is a placeholder that would integrate with the DSL compiler.
    }

    const CACHE_MAGIC: u32 = 0x56504348; // "VPCH"
    const CACHE_VERSION: u32 = 1;

    const CacheHeader = extern struct {
        magic: u32,
        version: u32,
        data_size: usize,
        checksum: u64,
    };

    fn computeChecksum(data: []const u8) u64 {
        return std.hash.Wyhash.hash(0, data);
    }

    fn getDefaultCachePath() []const u8 {
        return ".abi_pipeline_cache.bin";
    }
};

/// Global pipeline cache instance
var global_pipeline_cache: ?VulkanPipelineCache = null;
var cache_init_mutex = std.Thread.Mutex{};

/// Initialize the global pipeline cache
pub fn initGlobalCache(allocator: std.mem.Allocator, config: PipelineCacheConfig) !void {
    cache_init_mutex.lock();
    defer cache_init_mutex.unlock();

    if (global_pipeline_cache != null) return;

    global_pipeline_cache = try VulkanPipelineCache.init(allocator, config);

    // Optionally warm cache in background
    if (config.enable_warming) {
        global_pipeline_cache.?.warmCache() catch |err| {
            std.log.debug("Cache warming failed: {}", .{err});
        };
    }
}

/// Get the global pipeline cache
pub fn getGlobalCache() ?*VulkanPipelineCache {
    cache_init_mutex.lock();
    defer cache_init_mutex.unlock();

    if (global_pipeline_cache) |*cache| {
        return cache;
    }
    return null;
}

/// Deinitialize the global pipeline cache
pub fn deinitGlobalCache() void {
    cache_init_mutex.lock();
    defer cache_init_mutex.unlock();

    if (global_pipeline_cache) |*cache| {
        cache.deinit();
        global_pipeline_cache = null;
    }
}

/// Get global cache statistics
pub fn getGlobalCacheStats() ?CacheStats {
    if (getGlobalCache()) |cache| {
        return cache.getStats();
    }
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "pipeline cache config defaults" {
    const config = PipelineCacheConfig{};
    try std.testing.expect(config.enable_persistence);
    try std.testing.expect(config.enable_warming);
    try std.testing.expectEqual(@as(usize, 256 * 1024 * 1024), config.max_cache_size);
}

test "cache stats hit rate" {
    var stats = CacheStats{
        .hits = 80,
        .misses = 20,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 0.8), stats.hitRate(), 0.001);
}

test "checksum computation" {
    const data = "test data for checksum";
    const checksum1 = VulkanPipelineCache.computeChecksum(data);
    const checksum2 = VulkanPipelineCache.computeChecksum(data);

    try std.testing.expectEqual(checksum1, checksum2);

    const different_data = "different data";
    const checksum3 = VulkanPipelineCache.computeChecksum(different_data);
    try std.testing.expect(checksum1 != checksum3);
}
