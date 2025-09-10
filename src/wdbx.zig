// ----------  wdbx.zig  ----------
// This file unifies the four original modules:
//
//   * wdbx_cli.zig
//   * wdbx_enhanced.zig
//   * wdbx_http_server.zig
//   * wdbx_production.zig
//
// All imports are consolidated at the top and duplicate type definitions are
// removed.  The HTTP server and CLI refer to the `WdbxProduction` type via
// an alias named `database.Db` so that the original code compiles unchanged.
//
// ---------------------------------------------------------------
// 1. Imports & global constants
// ---------------------------------------------------------------
const std = @import("std");
const builtin = @import("builtin");
const simd = @import("simd/mod.zig");
const http = std.http;

// ---------------------------------------------------------------
// 2. Common data structures
// ---------------------------------------------------------------
const PRODUCTION_DEFAULTS = struct {
    const MAX_VECTORS_PER_SHARD = 1_000_000;
    const SHARD_COUNT = 16;
    const L1_CACHE_SIZE_MB = 256;
    const L2_CACHE_SIZE_MB = 1024;
    const L3_CACHE_SIZE_MB = 4096;
    const CHECKPOINT_INTERVAL_MS = 60_000;
    const HEALTH_CHECK_INTERVAL_MS = 5_000;
    const METRICS_EXPORT_INTERVAL_MS = 10_000;
    const MAX_CONCURRENT_OPERATIONS = 10_000;
    const COMPRESSION_BATCH_SIZE = 1000;
    const RECOVERY_RETRY_ATTEMPTS = 5;
    const BACKUP_RETENTION_DAYS = 30;
};

pub const Metrics = struct {
    // Performance metrics
    operations_total: std.atomic.Value(u64),
    operations_failed: std.atomic.Value(u64),
    latency_histogram: LatencyHistogram,
    throughput_rate: std.atomic.Value(f64),

    // Resource metrics
    memory_used_bytes: std.atomic.Value(usize),
    memory_peak_bytes: std.atomic.Value(usize),
    cache_hit_rate: std.atomic.Value(f64),
    compression_ratio: std.atomic.Value(f64),

    // Health metrics
    health_score: std.atomic.Value(f32),
    last_checkpoint_time: std.atomic.Value(i64),
    error_rate: std.atomic.Value(f64),
    recovery_count: std.atomic.Value(u32),

    // Shard metrics
    shard_distribution: []ShardMetrics,
    rebalance_operations: std.atomic.Value(u64),

    const LatencyHistogram = struct {
        buckets: [20]std.atomic.Value(u64),
        sum: std.atomic.Value(f64),
        count: std.atomic.Value(u64),

        fn record(self: *LatencyHistogram, latency_ms: f64) void {
            const bucket = @min(19, @as(usize, @intFromFloat(std.math.log2(latency_ms * 10))));
            _ = self.buckets[bucket].fetchAdd(1, .monotonic);
            _ = self.sum.fetchAdd(latency_ms, .monotonic);
            _ = self.count.fetchAdd(1, .monotonic);
        }

        fn percentile(self: *const LatencyHistogram, p: f64) f64 {
            const total = self.count.load(.monotonic);
            const target = @as(u64, @intFromFloat(@as(f64, @floatFromInt(total)) * p / 100.0));
            var sum: u64 = 0;
            for (self.buckets, 0..) |*bucket, i| {
                sum += bucket.load(.monotonic);
                if (sum >= target) {
                    return std.math.exp2(@as(f64, @floatFromInt(i))) / 10.0;
                }
            }
            return 10000.0; // Max 10s
        }
    };

    const ShardMetrics = struct {
        vector_count: std.atomic.Value(u64),
        size_bytes: std.atomic.Value(usize),
        load_factor: std.atomic.Value(f32),
        last_access: std.atomic.Value(i64),
    };

    pub fn init(allocator: std.mem.Allocator, shard_count: usize) !*Metrics {
        const self = try allocator.create(Metrics);
        self.* = .{
            .operations_total = std.atomic.Value(u64).init(0),
            .operations_failed = std.atomic.Value(u64).init(0),
            .latency_histogram = std.mem.zeroes(LatencyHistogram),
            .throughput_rate = std.atomic.Value(f64).init(0),
            .memory_used_bytes = std.atomic.Value(usize).init(0),
            .memory_peak_bytes = std.atomic.Value(usize).init(0),
            .cache_hit_rate = std.atomic.Value(f64).init(0),
            .compression_ratio = std.atomic.Value(f64).init(1.0),
            .health_score = std.atomic.Value(f32).init(1.0),
            .last_checkpoint_time = std.atomic.Value(i64).init(0),
            .error_rate = std.atomic.Value(f64).init(0),
            .recovery_count = std.atomic.Value(u32).init(0),
            .shard_distribution = try allocator.alloc(ShardMetrics, shard_count),
            .rebalance_operations = std.atomic.Value(u64).init(0),
        };

        for (self.shard_distribution) |*shard| {
            shard.* = std.mem.zeroes(ShardMetrics);
        }

        return self;
    }

    pub fn exportPrometheus(self: *const Metrics, writer: anytype) !void {
        try writer.print("# HELP wdbx_operations_total Total number of operations\n", .{});
        try writer.print("# TYPE wdbx_operations_total counter\n", .{});
        try writer.print("wdbx_operations_total {d}\n", .{self.operations_total.load(.monotonic)});

        try writer.print("# HELP wdbx_operations_failed Failed operations\n", .{});
        try writer.print("# TYPE wdbx_operations_failed counter\n", .{});
        try writer.print("wdbx_operations_failed {d}\n", .{self.operations_failed.load(.monotonic)});

        try writer.print("# HELP wdbx_latency_p50 50th percentile latency in ms\n", .{});
        try writer.print("# TYPE wdbx_latency_p50 gauge\n", .{});
        try writer.print("wdbx_latency_p50 {d:.2}\n", .{self.latency_histogram.percentile(50)});

        try writer.print("# HELP wdbx_latency_p99 99th percentile latency in ms\n", .{});
        try writer.print("# TYPE wdbx_latency_p99 gauge\n", .{});
        try writer.print("wdbx_latency_p99 {d:.2}\n", .{self.latency_histogram.percentile(99)});

        try writer.print("# HELP wdbx_memory_bytes Memory usage in bytes\n", .{});
        try writer.print("# TYPE wdbx_memory_bytes gauge\n", .{});
        try writer.print("wdbx_memory_bytes {d}\n", .{self.memory_used_bytes.load(.monotonic)});
    }
};

/// Runtime SIMD capabilities
pub const SimdCapabilities = struct {
    has_sse: bool = false,
    has_sse2: bool = false,
    has_sse3: bool = false,
    has_ssse3: bool = false,
    has_sse41: bool = false,
    has_sse42: bool = false,
    has_avx: bool = false,
    has_avx2: bool = false,
    has_avx512: bool = false,
    has_neon: bool = false, // ARM

    pub fn detect() SimdCapabilities {
        var caps = SimdCapabilities{};
        if (builtin.cpu.arch == .x86_64) {
            if (std.Target.x86.featureSetHas(builtin.cpu.features, .sse)) caps.has_sse = true;
            if (std.Target.x86.featureSetHas(builtin.cpu.features, .sse2)) caps.has_sse2 = true;
            if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx)) caps.has_avx = true;
            if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) caps.has_avx2 = true;
        } else if (builtin.cpu.arch == .aarch64) {
            caps.has_neon = true;
        }
        return caps;
    }
};

// ---------------------------------------------------------------
// 3. Compression
// ---------------------------------------------------------------
pub const CompressionType = enum {
    none,
    quantization_8bit,
    quantization_4bit,
    pq_compression, // Product Quantization
    zstd,
    delta_encoding,

    pub fn compress(self: CompressionType, data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        return switch (self) {
            .none => try serializeF32(data, allocator),
            .quantization_8bit => try quantize8Bit(data, allocator),
            .quantization_4bit => try quantize4Bit(data, allocator),
            .pq_compression => try productQuantization(data, allocator),
            .zstd => try zstdCompress(data, allocator),
            .delta_encoding => try deltaEncode(data, allocator),
        };
    }

    fn serializeF32(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        const bytes = try allocator.alloc(u8, data.len * @sizeOf(f32));
        @memcpy(bytes, std.mem.sliceAsBytes(data));
        return bytes;
    }

    fn quantize8Bit(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        var min: f32 = std.math.inf(f32);
        var max: f32 = -std.math.inf(f32);
        for (data) |v| {
            min = @min(min, v);
            max = @max(max, v);
        }
        const range = max - min;
        const scale = if (range > 0) 255.0 / range else 1.0;
        const result = try allocator.alloc(u8, 8 + data.len);
        @memcpy(result[0..4], &std.mem.toBytes(min));
        @memcpy(result[4..8], &std.mem.toBytes(scale));
        for (data, 0..) |v, i| {
            const normalized = (v - min) * scale;
            result[8 + i] = @intFromFloat(@min(255, @max(0, normalized)));
        }
        return result;
    }

    fn quantize4Bit(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        var min: f32 = std.math.inf(f32);
        var max: f32 = -std.math.inf(f32);
        for (data) |v| {
            min = @min(min, v);
            max = @max(max, v);
        }
        const range = max - min;
        const scale = if (range > 0) 15.0 / range else 1.0; // 4-bit = 16 values (0-15)
        const result = try allocator.alloc(u8, 8 + (data.len + 1) / 2); // Pack 2 values per byte
        @memcpy(result[0..4], &std.mem.toBytes(min));
        @memcpy(result[4..8], &std.mem.toBytes(scale));

        var byte_idx: usize = 8;
        var bit_offset: u3 = 0;
        for (data) |v| {
            const normalized = (v - min) * scale;
            const quantized = @as(u8, @intFromFloat(@min(15, @max(0, normalized))));

            if (bit_offset == 0) {
                result[byte_idx] = @as(u8, quantized);
                bit_offset = 4;
            } else {
                result[byte_idx] |= @as(u8, quantized) << 4;
                byte_idx += 1;
                bit_offset = 0;
            }
        }
        return result;
    }

    fn productQuantization(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        // Simple product quantization: divide vector into subvectors and quantize each
        const subvector_size = 4; // Each subvector has 4 dimensions
        const num_subvectors = (data.len + subvector_size - 1) / subvector_size;
        const result = try allocator.alloc(u8, 4 + num_subvectors); // 4 bytes for metadata

        // Store original length and subvector size
        @memcpy(result[0..4], &std.mem.toBytes(@as(u32, @intCast(data.len))));

        for (0..num_subvectors) |i| {
            const start = i * subvector_size;
            const end = @min(start + subvector_size, data.len);
            const subvector = data[start..end];

            // Simple quantization: average the subvector values
            var sum: f32 = 0;
            for (subvector) |v| sum += v;
            const avg = sum / @as(f32, @floatFromInt(subvector.len));
            result[4 + i] = @intFromFloat(@min(255, @max(0, (avg + 1.0) * 127.5))); // Map [-1,1] to [0,255]
        }
        return result;
    }

    fn zstdCompress(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        // For now, just serialize as raw bytes (zstd would require external library)
        return serializeF32(data, allocator);
    }

    fn deltaEncode(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
        if (data.len == 0) return try allocator.alloc(u8, 0);

        const result = try allocator.alloc(u8, 4 + (data.len - 1) * @sizeOf(f32));
        @memcpy(result[0..4], &std.mem.toBytes(data[0])); // Store first value

        for (data[1..], 0..) |value, i| {
            const delta = value - data[i];
            @memcpy(result[4 + i * @sizeOf(f32) .. 4 + (i + 1) * @sizeOf(f32)], &std.mem.toBytes(delta));
        }
        return result;
    }
};

/// Helper to compress a vector
fn compressVector(data: []const f32, allocator: std.mem.Allocator) ![]u8 {
    return CompressionType.quantization_8bit.compress(data, allocator);
}

/// Helper to decompress a vector
fn decompressVector(data: []const u8, allocator: std.mem.Allocator) ![]f32 {
    // Basic placeholder implementation that interprets data as raw f32 bytes.
    const len = data.len;
    if (len % @sizeOf(f32) != 0) return error.InvalidCompressedData;
    const float_count = len / @sizeOf(f32);
    const floats = try allocator.alloc(f32, float_count);
    std.mem.copyForwards(floats, std.mem.bytesToSlice(f32, data));
    return floats;
}

// ---------------------------------------------------------------
// 4. LSH Index
// ---------------------------------------------------------------
pub const LshIndex = struct {
    allocator: std.mem.Allocator,
    dimension: usize,
    num_bands: usize,
    buckets: std.StringHashMap([]u64),

    pub fn init(allocator: std.mem.Allocator, dimension: usize, num_bands: usize) !*LshIndex {
        const self = try allocator.create(LshIndex);
        self.* = .{
            .allocator = allocator,
            .dimension = dimension,
            .num_bands = num_bands,
            .buckets = std.StringHashMap([]u64).init(allocator),
        };
        return self;
    }

    pub fn insert(self: *LshIndex, key: []const u8, id: u64) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        const entry = try self.buckets.getOrPut(key_copy);
        if (entry.found_existing) {
            // Extend the existing array
            const old_ids = entry.value_ptr.*;
            const new_ids = try self.allocator.realloc(old_ids, old_ids.len + 1);
            new_ids[old_ids.len] = id;
            entry.value_ptr.* = new_ids;
        } else {
            // Create new array with the ID
            const new_ids = try self.allocator.alloc(u64, 1);
            new_ids[0] = id;
            entry.value_ptr.* = new_ids;
        }
    }

    pub fn query(self: *LshIndex, key: []const u8) []u64 {
        const ids = self.buckets.get(key) orelse return &.{};
        return ids;
    }

    pub fn deinit(self: *LshIndex) void {
        var iterator = self.buckets.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.buckets.deinit();
        self.allocator.destroy(self);
    }
};

// ---------------------------------------------------------------
// 5. Cache system
// ---------------------------------------------------------------
pub const CacheSystem = struct {
    allocator: std.mem.Allocator,
    l1_cache_size: usize,
    l2_cache_size: usize,
    l3_cache_size: usize,

    pub fn init(allocator: std.mem.Allocator, config: ProductionConfig) !*CacheSystem {
        const self = try allocator.create(CacheSystem);
        self.* = .{
            .allocator = allocator,
            .l1_cache_size = config.l1_cache_size_mb * 1024 * 1024,
            .l2_cache_size = config.l2_cache_size_mb * 1024 * 1024,
            .l3_cache_size = config.l3_cache_size_mb * 1024 * 1024,
        };
        return self;
    }
};

// ---------------------------------------------------------------
// 6. Health monitor
// ---------------------------------------------------------------
pub const HealthMonitor = struct {
    allocator: std.mem.Allocator,
    db: *WdbxProduction,

    pub fn init(allocator: std.mem.Allocator, db: *WdbxProduction) !*HealthMonitor {
        const self = try allocator.create(HealthMonitor);
        self.* = .{ .allocator = allocator, .db = db };
        return self;
    }

    pub fn checkHealth(self: *HealthMonitor) bool {
        return self.db != null;
    }
};

// ---------------------------------------------------------------
// 7. Production configuration
// ---------------------------------------------------------------
pub const ProductionConfig = struct {
    // Scalability
    shard_count: usize = PRODUCTION_DEFAULTS.SHARD_COUNT,
    max_vectors_per_shard: usize = PRODUCTION_DEFAULTS.MAX_VECTORS_PER_SHARD,
    auto_rebalance: bool = true,

    // Reliability
    checkpoint_interval_ms: u64 = PRODUCTION_DEFAULTS.CHECKPOINT_INTERVAL_MS,
    health_check_interval_ms: u64 = PRODUCTION_DEFAULTS.HEALTH_CHECK_INTERVAL_MS,
    recovery_retry_attempts: u32 = PRODUCTION_DEFAULTS.RECOVERY_RETRY_ATTEMPTS,
    enable_auto_recovery: bool = true,

    // Efficiency
    l1_cache_size_mb: usize = PRODUCTION_DEFAULTS.L1_CACHE_SIZE_MB,
    l2_cache_size_mb: usize = PRODUCTION_DEFAULTS.L2_CACHE_SIZE_MB,
    l3_cache_size_mb: usize = PRODUCTION_DEFAULTS.L3_CACHE_SIZE_MB,
    compression_type: CompressionType = .quantization_8bit,
    compression_batch_size: usize = PRODUCTION_DEFAULTS.COMPRESSION_BATCH_SIZE,

    // Observability
    enable_metrics: bool = true,
    metrics_export_interval_ms: u64 = PRODUCTION_DEFAULTS.METRICS_EXPORT_INTERVAL_MS,
    enable_tracing: bool = true,
    log_level: LogLevel = .info,

    // Performance
    max_concurrent_operations: usize = PRODUCTION_DEFAULTS.MAX_CONCURRENT_OPERATIONS,
    thread_pool_size: usize = 0, // 0 = auto‑detect
    enable_simd: bool = true,
    enable_gpu: bool = false,

    // Persistence
    data_dir: []const u8 = "./wdbx_data",
    backup_dir: []const u8 = "./wdbx_backups",
    backup_retention_days: u32 = PRODUCTION_DEFAULTS.BACKUP_RETENTION_DAYS,
    enable_wal: bool = true, // Write‑ahead logging

    const LogLevel = enum {
        debug,
        info,
        warning,
        err,
        critical,
    };
};

// ---------------------------------------------------------------
// 8. Production database implementation (WdbxProduction)
// ---------------------------------------------------------------
pub const WdbxProduction = struct {
    allocator: std.mem.Allocator,
    config: ProductionConfig,
    dimension: u16 = 0,
    row_count: u64 = 0,
    metrics: *Metrics = undefined,
    vectors: std.ArrayList([]f32),
    lsh_index: ?*LshIndex = null,

    pub fn init(allocator: std.mem.Allocator, cfg: ProductionConfig) !*WdbxProduction {
        const self = try allocator.create(WdbxProduction);
        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .dimension = 0,
            .row_count = 0,
            .vectors = .{},
        };
        // Initialize vectors ArrayList
        self.vectors = .{ .items = &.{}, .capacity = 0 };

        self.metrics = try Metrics.init(allocator, cfg.shard_count);

        // Initialize LSH index if enabled
        if (cfg.enable_simd) {
            self.lsh_index = try LshIndex.init(allocator, 128, 8); // Default 128-dim, 8 bands
        }

        return self;
    }

    pub fn open(path: []const u8, read_only: bool) !*WdbxProduction {
        // In this unified file we treat `WdbxProduction` as the database type.
        // The original code expected a separate `database.Db`; an alias is
        // provided later in the file.
        _ = path;
        _ = read_only;
        return init(std.heap.page_allocator, ProductionConfig{});
    }

    pub fn close(self: *WdbxProduction) void {
        // Clean up vectors
        for (self.vectors.items) |vector| {
            self.allocator.free(vector);
        }
        self.vectors.deinit(self.allocator);

        // Clean up LSH index
        if (self.lsh_index) |lsh| {
            lsh.deinit();
        }

        // Clean up metrics
        self.allocator.free(self.metrics.shard_distribution);
        self.allocator.destroy(self.metrics);

        self.allocator.destroy(self);
    }

    pub fn addEmbedding(self: *WdbxProduction, vector: []const f32) !u64 {
        // Set dimension on first vector
        if (self.dimension == 0) {
            self.dimension = @intCast(vector.len);
        } else if (vector.len != self.dimension) {
            return error.InvalidDimension;
        }

        // Store the vector
        const vector_copy = try self.allocator.dupe(f32, vector);
        try self.vectors.append(self.allocator, vector_copy);

        const id = self.row_count;
        self.row_count += 1;

        // Add to LSH index if available
        if (self.lsh_index) |lsh| {
            // Create a simple hash key from the vector (in real implementation, use proper LSH)
            var hash_key: [32]u8 = undefined;
            for (vector, 0..) |v, i| {
                if (i < 8) { // Use first 8 values for hash
                    const bytes = std.mem.toBytes(v);
                    @memcpy(hash_key[i * 4 .. (i + 1) * 4], &bytes);
                }
            }
            try lsh.insert(&hash_key, id);
        }

        // Update metrics
        _ = self.metrics.operations_total.fetchAdd(1, .monotonic);

        return id;
    }

    pub fn search(self: *WdbxProduction, vector: []const f32, k: usize, allocator: std.mem.Allocator) ![]SearchResult {
        if (vector.len != self.dimension) {
            return error.InvalidDimension;
        }

        if (self.vectors.items.len == 0) {
            return try allocator.alloc(SearchResult, 0);
        }

        // Define a named struct for distance results
        const DistanceResult = struct { id: u64, distance: f32 };

        // Calculate distances to all vectors
        var distances = try std.ArrayList(DistanceResult).initCapacity(allocator, self.vectors.items.len);
        defer distances.deinit(allocator);

        for (self.vectors.items, 0..) |stored_vector, i| {
            const distance = self.calculateDistance(vector, stored_vector);
            try distances.append(allocator, .{ .id = @intCast(i), .distance = distance });
        }

        // Sort by distance (ascending)
        std.sort.block(DistanceResult, distances.items, {}, struct {
            pub fn lessThan(_: void, a: DistanceResult, b: DistanceResult) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        // Return top k results
        const result_count = @min(k, distances.items.len);
        const results = try allocator.alloc(SearchResult, result_count);

        for (0..result_count) |i| {
            const dist = distances.items[i];
            results[i] = .{
                .index = dist.id,
                .score = dist.distance,
                .metadata = null,
            };
        }

        // Update metrics
        _ = self.metrics.operations_total.fetchAdd(1, .monotonic);

        return results;
    }

    fn calculateDistance(self: *WdbxProduction, a: []const f32, b: []const f32) f32 {
        _ = self;
        var sum: f32 = 0;
        for (a, b) |va, vb| {
            const diff = va - vb;
            sum += diff * diff;
        }
        return @sqrt(sum); // Euclidean distance
    }

    pub fn getStats(self: *WdbxProduction) Stats {
        _ = self;
        return .{
            .search_count = 0,
            .write_count = 0,
            .initialization_count = 0,
        };
    }

    pub fn getRowCount(self: *WdbxProduction) u64 {
        return self.row_count;
    }

    pub fn getDimension(self: *WdbxProduction) u16 {
        return self.dimension;
    }

    const Stats = struct {
        search_count: u64,
        write_count: u64,
        initialization_count: u64,

        fn getAverageSearchTime(self: *const Stats) u64 {
            if (self.search_count == 0) return 0;
            return 0; // placeholder
        }
    };

    const SearchResult = struct {
        index: u64,
        score: f32,
        metadata: ?[]const u8,
    };
};

// ---------------------------------------------------------------
// 9. Alias to preserve original database.Db references
// ---------------------------------------------------------------
const database = struct {
    pub const Db = WdbxProduction;
};

// ---------------------------------------------------------------
// 10. HTTP server (unchanged, but now uses database.Db alias)
// ---------------------------------------------------------------
pub const WdbxHttpServer = struct {
    allocator: std.mem.Allocator,
    config: ServerConfig,
    db: ?*database.Db,
    rate_limiter: RateLimiter,

    const Self = @This();

    const ServerConfig = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 8080,
        max_request_size: usize = 1024 * 1024,
        rate_limit: usize = 1000,
        enable_cors: bool = true,
        enable_auth: bool = true,
        jwt_secret: []const u8 = "",
    };

    const RateLimiter = struct {
        requests: std.HashMap(u32, u64, std.hash_map.AutoContext(u32), 80),
        last_reset: i64,
        max_requests: usize,
        window_ms: i64 = 60 * 1000,

        pub fn init(allocator: std.mem.Allocator, max_requests: usize) RateLimiter {
            return .{
                .requests = std.HashMap(u32, u64, std.hash_map.AutoContext(u32), 80).init(allocator),
                .last_reset = std.time.milliTimestamp(),
                .max_requests = max_requests,
            };
        }

        pub fn deinit(self: *RateLimiter) void {
            self.requests.deinit();
        }

        pub fn checkLimit(self: *RateLimiter, ip: u32) !bool {
            const now = std.time.milliTimestamp();

            if (now - self.last_reset > self.window_ms) {
                self.requests.clearRetainingCapacity();
                self.last_reset = now;
            }

            const current = self.requests.get(ip) orelse 0;
            if (current >= self.max_requests) return false;
            try self.requests.put(ip, current + 1);
            return true;
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .db = null,
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.db) |db| db.close();
        self.rate_limiter.deinit();
        self.allocator.destroy(self);
    }

    pub fn openDatabase(self: *Self, path: []const u8) !void {
        // Create a fresh production instance for the HTTP server.
        self.db = try database.Db.open(path, true);
    }

    pub fn start(self: *Self) !void {
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        const server = http.Server.init(.{ .allocator = self.allocator, .reuse_address = true });
        try server.listen(address);
        std.debug.print("WDBX HTTP server listening on {s}:{d}\n", .{ self.config.host, self.config.port });

        while (true) {
            var response = try server.accept(.{});
            defer response.deinit();
            try self.handleRequest(&response);
        }
    }

    fn handleRequest(self: *Self, response: *http.Server.Response) !void {
        const request = response.request;
        const uri = request.target;

        const client_ip = self.getClientIP(response);
        if (!try self.rate_limiter.checkLimit(client_ip)) {
            try self.sendError(response, 429, "Too Many Requests");
            return;
        }

        if (self.config.enable_cors) try self.addCorsHeaders(response);

        if (std.mem.eql(u8, uri, "/")) {
            try self.handleRoot(response);
        } else if (std.mem.eql(u8, uri, "/health")) {
            try self.handleHealth(response);
        } else if (std.mem.eql(u8, uri, "/stats")) {
            try self.handleStats(response);
        } else if (std.mem.startsWith(u8, uri, "/add")) {
            try self.handleAdd(response);
        } else if (std.mem.startsWith(u8, uri, "/query")) {
            try self.handleQuery(response);
        } else if (std.mem.startsWith(u8, uri, "/knn")) {
            try self.handleKnn(response);
        } else if (std.mem.startsWith(u8, uri, "/monitor")) {
            try self.handleMonitor(response);
        } else {
            try self.sendError(response, 404, "Not Found");
        }
    }

    fn handleRoot(_: *Self, response: *http.Server.Response) !void {
        const html =
            \\<!DOCTYPE html>
            \\<html>
            \\<head>
            \\    <title>WDBX Vector Database</title>
            \\    <style>
            \\        body { font-family: Arial, sans-serif; margin: 40px; }
            \\        .container { max-width: 800px; margin: 0 auto; }
            \\        h1 { color: #333; }
            \\        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            \\        .method { color: #007acc; font-weight: bold; }
            \\        .url { font-family: monospace; background: #e8e8e8; padding: 2px 6px; }
            \\    </style>
            \\</head>
            \\<body>
            \\    <div class="container">
            \\        <h1>WDBX Vector Database API</h1>
            \\        <p>Welcome to the WDBX Vector Database HTTP API.</p>
            \\        <h2>Available Endpoints</h2>
            \\        <div class="endpoint"><span class="method">GET</span> <span class="url">/health</span><p>Check server health status</p></div>
            \\        <div class="endpoint"><span class="method">GET</span> <span class="url">/stats</span><p>Get database statistics</p></div>
            \\        <div class="endpoint"><span class="method">POST</span> <span class="url">/add</span><p>Add a vector to the database (requires admin token)</p></div>
            \\        <div class="endpoint"><span class="method">GET</span> <span class="url">/query?vec=1.0,2.0,3.0</span><p>Query nearest neighbor</p></div>
            \\        <div class="endpoint"><span class="method">GET</span> <span class="url">/knn?vec=1.0,2.0,3.0&k=5</span><p>Query k‑nearest neighbors</p></div>
            \\        <div class="endpoint"><span class="method">GET</span> <span class="url">/monitor</span><p>Get performance metrics</p></div>
            \\        <h2>Authentication</h2>
            \\        <p>Admin operations require a JWT token in the Authorization header:</p>
            \\        <code>Authorization: Bearer &lt;your‑jwt‑token&gt;</code>
            \\        <h2>Vector Format</h2>
            \\        <p>Vectors should be comma‑separated float values, e.g.: <code>1.0,2.0,3.0,4.0</code></p>
            \\    </div>
            \\</body>
            \\</html>
        ;
        try response.headers.append("Content-Type", "text/html");
        try response.do();
        try response.writer().writeAll(html);
    }

    fn handleHealth(self: *Self, response: *http.Server.Response) !void {
        const health = .{
            .status = "healthy",
            .version = "WDBX Vector Database v1.0.0",
            .timestamp = std.time.milliTimestamp(),
            .database_connected = self.db != null,
        };
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"status\":\"{s}\",\"version\":\"{s}\",\"timestamp\":{d},\"database_connected\":{any}}}", .{ health.status, health.version, health.timestamp, health.database_connected });
    }

    fn handleStats(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }
        const db = self.db.?;
        const stats = db.getStats();
        const db_stats = .{
            .vectors_stored = db.getRowCount(),
            .vector_dimension = db.getDimension(),
            .searches_performed = stats.search_count,
            .average_search_time_us = stats.getAverageSearchTime(),
            .writes_performed = stats.write_count,
            .initializations = stats.initialization_count,
        };
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"vectors_stored\":{d},\"vector_dimension\":{d},\"searches_performed\":{d},\"average_search_time_us\":{d},\"writes_performed\":{d},\"initializations\":{d}}}", .{ db_stats.vectors_stored, db_stats.vector_dimension, db_stats.searches_performed, db_stats.average_search_time_us, db_stats.writes_performed, db_stats.initializations });
    }

    fn handleAdd(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }
        if (self.config.enable_auth) {
            const auth_header = response.request.headers.get("authorization") orelse {
                try self.sendError(response, 401, "Authorization required");
                return;
            };
            if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
                try self.sendError(response, 401, "Invalid authorization format");
                return;
            }
            const token = auth_header[7..];
            if (!self.validateJWT(token)) {
                try self.sendError(response, 403, "Invalid or expired token");
                return;
            }
        }
        const body = try self.readRequestBody(response);
        defer self.allocator.free(body);
        const vector = try self.parseVector(body);
        defer self.allocator.free(vector);
        const db = self.db.?;
        const row_id = try db.addEmbedding(vector);
        std.debug.print("Vector added successfully at row {d}\n", .{row_id});
    }

    fn handleQuery(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }
        const query = response.request.target;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendError(response, 400, "Missing 'vec' parameter");
            return;
        };
        const vec_end = std.mem.indexOfScalar(u8, query[vec_start..], '&') orelse query.len;
        const vector_str = query[vec_start + 4 .. vec_start + vec_end];
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);
        const db = self.db.?;
        const results = try db.search(vector, 1, self.allocator);
        defer self.allocator.free(results);
        if (results.len > 0) {
            const result = .{
                .success = true,
                .nearest_neighbor = .{
                    .index = results[0].index,
                    .distance = results[0].score,
                },
            };
            try response.headers.append("Content-Type", "application/json");
            try response.do();
            try response.writer().print("{{\"success\":{any},\"nearest_neighbor\":{{\"index\":{d},\"distance\":{d}}}}}", .{ result.success, result.nearest_neighbor.index, result.nearest_neighbor.distance });
        } else {
            try self.sendError(response, 404, "No vectors found in database");
        }
    }

    fn handleKnn(self: *Self, response: *http.Server.Response) !void {
        if (self.db == null) {
            try self.sendError(response, 503, "Database not connected");
            return;
        }
        const query = response.request.target;
        const vec_start = std.mem.indexOf(u8, query, "vec=") orelse {
            try self.sendError(response, 400, "Missing 'vec' parameter");
            return;
        };
        const vec_end = std.mem.indexOfScalar(u8, query[vec_start..], '&') orelse query.len;
        const vector_str = query[vec_start + 4 .. vec_start + vec_end];
        var k: usize = 5;
        if (std.mem.indexOf(u8, query, "k=")) |k_start| {
            const k_end = std.mem.indexOfScalar(u8, query[k_start..], '&') orelse query.len;
            const k_str = query[k_start + 2 .. k_end];
            k = try std.fmt.parseInt(usize, k_str, 10);
        }
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);
        const db = self.db.?;
        const results = try db.search(vector, k, self.allocator);
        defer self.allocator.free(results);
        var neighbors = try std.ArrayList(struct { index: u64, distance: f32 }).initCapacity(self.allocator, results.len);
        defer neighbors.deinit(self.allocator);
        for (results) |result| {
            try neighbors.append(self.allocator, .{ .index = result.index, .distance = result.score });
        }
        const result = .{
            .success = true,
            .k = k,
            .neighbors = neighbors.items,
        };
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"success\":{any},\"k\":{d},\"neighbors\":[{s}]}}", .{ result.success, result.k, try self.formatNeighbors(result.neighbors) });
    }

    fn handleMonitor(_: *Self, response: *http.Server.Response) !void {
        response.status = 501;
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"error\":\"Performance monitoring not yet implemented\"}}", .{});
    }

    fn sendError(_: *Self, response: *http.Server.Response, status: u16, message: []const u8) !void {
        response.status = status;
        try response.headers.append("Content-Type", "application/json");
        try response.do();
        try response.writer().print("{{\"error\":{any},\"status\":{d},\"message\":\"{s}\"}}", .{ true, status, message });
    }

    fn addCorsHeaders(_: *Self, response: *http.Server.Response) !void {
        try response.headers.append("Access-Control-Allow-Origin", "*");
        try response.headers.append("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        try response.headers.append("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }

    fn getClientIP(_: *Self, _: *http.Server.Response) u32 {
        return 0x7f000001;
    }

    fn readRequestBody(self: *Self, response: *http.Server.Response) ![]u8 {
        const content_length = response.request.headers.get("content-length") orelse "0";
        const length = try std.fmt.parseInt(usize, content_length, 10);
        if (length > self.config.max_request_size) return error.RequestTooLarge;
        const body = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(body);
        var total_read: usize = 0;
        while (total_read < length) {
            const read = try response.reader().read(body[total_read..]);
            if (read == 0) break;
            total_read += read;
        }
        return body[0..total_read];
    }

    fn parseVector(self: *Self, vector_str: []const u8) ![]f32 {
        var list = try std.ArrayList(f32).initCapacity(self.allocator, 8);
        defer list.deinit(self.allocator);
        var iter = std.mem.splitSequence(u8, vector_str, ",");
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\n\r");
            if (trimmed.len > 0) {
                const value = try std.fmt.parseFloat(f32, trimmed);
                try list.append(self.allocator, value);
            }
        }
        return try list.toOwnedSlice(self.allocator);
    }

    fn validateJWT(_: *Self, token: []const u8) bool {
        return token.len > 0;
    }

    fn formatNeighbors(self: *Self, neighbors: []const struct { index: u64, distance: f32 }) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit(self.allocator);
        for (neighbors, 0..) |neighbor, i| {
            if (i > 0) try buffer.appendSlice(",");
            try buffer.writer().print("{{\"index\":{d},\"distance\":{d}}}", .{ neighbor.index, neighbor.distance });
        }
        return try buffer.toOwnedSlice(self.allocator);
    }
};

// ---------------------------------------------------------------
// 8. CLI implementation
// ---------------------------------------------------------------
pub const WdbxCLI = struct {
    allocator: std.mem.Allocator,
    options: Options,

    const Options = struct {
        command: Command = .help,
        verbose: bool = false,
        quiet: bool = false,
        db_path: ?[]const u8 = null,
        port: u16 = 8080,
        host: []const u8 = "127.0.0.1",
        k: usize = 5,
        vector: ?[]const u8 = null,
        role: []const u8 = "admin",
        output_format: OutputFormat = .text,

        pub fn deinit(self: *Options, allocator: std.mem.Allocator) void {
            if (self.db_path) |path| allocator.free(path);
            if (self.vector) |vec| allocator.free(vec);
        }
    };

    const OutputFormat = enum {
        text,
        json,
        csv,

        pub fn toString(self: OutputFormat) []const u8 {
            return @tagName(self);
        }
    };

    const Command = enum {
        help,
        version,
        knn,
        query,
        add,
        stats,
        monitor,
        optimize,
        save,
        load,
        http,
        tcp,
        ws,
        gen_token,

        pub fn fromString(str: []const u8) ?Command {
            inline for (std.meta.fields(Command)) |field| {
                if (std.mem.eql(u8, str, field.name)) {
                    return @enumFromInt(field.value);
                }
            }
            return null;
        }

        pub fn getDescription(self: Command) []const u8 {
            return switch (self) {
                .help => "Show help information",
                .version => "Show version information",
                .knn => "Query k‑nearest neighbors",
                .query => "Query nearest neighbor",
                .add => "Add vector to database",
                .stats => "Show database statistics",
                .monitor => "Show performance metrics",
                .optimize => "Run ML optimization",
                .save => "Save database to file",
                .load => "Load database from file",
                .http => "Start HTTP REST API server",
                .tcp => "Start TCP binary protocol server",
                .ws => "Start WebSocket server",
                .gen_token => "Generate JWT authentication token",
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator, opts: Options) !WdbxCLI {
        return WdbxCLI{ .allocator = allocator, .options = opts };
    }

    pub fn deinit(self: *WdbxCLI) void {
        self.options.deinit(self.allocator);
    }

    pub fn run(self: *WdbxCLI) !void {
        switch (self.options.command) {
            .help => try self.showHelp(),
            .version => try self.showVersion(),
            .knn => try self.runKnn(),
            .query => try self.runQuery(),
            .add => try self.runAdd(),
            .stats => try self.runStats(),
            .monitor => try self.runMonitor(),
            .optimize => try self.runOptimize(),
            .save => try self.runSave(),
            .load => try self.runLoad(),
            .http => try self.runHttpServer(),
            .tcp => try self.runTcpServer(),
            .ws => try self.runWebSocketServer(),
            .gen_token => try self.runGenToken(),
        }
    }

    fn showHelp(_: *WdbxCLI) !void {
        std.debug.print(
            \\WDBX Vector Database - Command Line Interface
            \\
            \\Usage: wdbx <command> [options]
            \\
            \\Commands:
            \\  knn <vector> [k]     Query k‑nearest neighbors (default k=5)
            \\  query <vector>       Query nearest neighbor
            \\  add <vector>         Add vector to database
            \\  stats                Show database statistics
            \\  monitor              Show performance metrics
            \\  optimize             Run ML optimization
            \\  save <file>          Save database to file
            \\  load <file>          Load database from file
            \\  http [port]          Start HTTP REST API server
            \\  tcp [port]           Start TCP binary protocol server
            \\  ws [port]            Start WebSocket server
            \\  gen_token [role]     Generate JWT authentication token
            \\
            \\Options:
            \\  --db <path>          Database file path
            \\  --host <host>        Server host (default: 127.0.0.1)
            \\  --port <port>        Server port (default: 8080)
            \\  --role <role>        User role for token generation
            \\  --format <format>    Output format: text, json, csv
            \\  --verbose            Enable verbose output
            \\  --quiet              Suppress output
            \\
        , .{});
    }

    fn showVersion(_: *WdbxCLI) !void {
        std.debug.print("{s}\n", .{"WDBX Vector Database v1.0.0"});
    }

    fn runKnn(self: *WdbxCLI) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for knn command\n", .{});
            return;
        }
        const vector_str = self.options.vector.?;
        const k = self.options.k;
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);
        const db = try WdbxProduction.init(self.allocator, ProductionConfig{});
        defer db.close();
        const results = try db.search(vector, k, self.allocator);
        defer self.allocator.free(results);
        std.debug.print("=== K-Nearest Neighbors (k={d}) ===\n", .{k});
        for (results, 0..) |result, i| {
            std.debug.print("{d:2}. Index: {d:6}, Distance: {d:.6}\n", .{ i + 1, result.index, result.score });
        }
    }

    fn runQuery(self: *WdbxCLI) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for query command\n", .{});
            return;
        }
        const vector_str = self.options.vector.?;
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);
        const db = try WdbxProduction.init(self.allocator, ProductionConfig{});
        defer db.close();
        const results = try db.search(vector, 1, self.allocator);
        defer self.allocator.free(results);
        if (results.len > 0) {
            std.debug.print("=== Nearest Neighbor ===\n", .{});
            std.debug.print("Index: {d}, Distance: {d:.6}\n", .{ results[0].index, results[0].score });
        } else {
            std.debug.print("No vectors found in database\n", .{});
        }
    }

    fn runAdd(self: *WdbxCLI) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for add command\n", .{});
            return;
        }
        const vector_str = self.options.vector.?;
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);
        const db = try WdbxProduction.init(self.allocator, ProductionConfig{});
        defer db.close();
        const row_id = try db.addEmbedding(vector);
        std.debug.print("Vector added successfully at row {d}\n", .{row_id});
    }

    fn runStats(self: *WdbxCLI) !void {
        const db = try WdbxProduction.init(self.allocator, ProductionConfig{});
        defer db.close();
        const stats = db.getStats();
        std.debug.print(
            \\=== WDBX Database Statistics ===
            \\Vectors stored: {d}
            \\Vector dimension: {d}
            \\Searches performed: {d}
            \\Average search time: {d}µs
            \\Writes performed: {d}
            \\Initializations: {d}
            \\
        , .{
            db.getRowCount(),
            db.getDimension(),
            stats.search_count,
            stats.getAverageSearchTime(),
            stats.write_count,
            stats.initialization_count,
        });
    }

    fn runMonitor(_: *WdbxCLI) !void {
        std.debug.print("Performance monitoring not yet implemented\n", .{});
    }

    fn runOptimize(_: *WdbxCLI) !void {
        std.debug.print("ML optimization not yet implemented\n", .{});
    }

    fn runSave(_: *WdbxCLI) !void {
        std.debug.print("Database save not yet implemented\n", .{});
    }

    fn runLoad(_: *WdbxCLI) !void {
        std.debug.print("Database load not yet implemented\n", .{});
    }

    fn runHttpServer(self: *WdbxCLI) !void {
        std.debug.print("Starting HTTP server for CLI at {s}:{d}\n", .{ self.options.host, self.options.port });
        // Placeholder: just print, actual server launch is handled in WdbxHttpServer
    }

    fn runTcpServer(_: *WdbxCLI) !void {
        std.debug.print("TCP server not yet implemented\n", .{});
    }

    fn runWebSocketServer(_: *WdbxCLI) !void {
        std.debug.print("WebSocket server not yet implemented\n", .{});
    }

    fn runGenToken(self: *WdbxCLI) !void {
        std.debug.print("Generating JWT token for role {s}\n", .{self.options.role});
    }

    fn parseVector(self: *WdbxCLI, vector_str: []const u8) ![]f32 {
        var list = try std.ArrayList(f32).initCapacity(self.allocator, 8);
        defer list.deinit(self.allocator);
        var iter = std.mem.splitSequence(u8, vector_str, ",");
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\n\r");
            if (trimmed.len > 0) {
                const value = try std.fmt.parseFloat(f32, trimmed);
                try list.append(self.allocator, value);
            }
        }
        return try list.toOwnedSlice(self.allocator);
    }
};

// ---------------------------------------------------------------
// Note: This module provides WDBX database functionality but is not a standalone executable.
// Use the main CLI application (src/cli/main.zig) for command-line usage.
// ---------------------------------------------------------------
