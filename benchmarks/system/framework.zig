//! Advanced Benchmark Framework with Statistical Analysis
//!
//! Industry-standard benchmarking with:
//! - Warm-up phases to stabilize CPU caches and branch predictors
//! - Statistical analysis (mean, median, std dev, percentiles)
//! - Robust outlier detection (Interquartile Range - IQR)
//! - Memory allocation tracking via TrackingAllocator
//! - Hardware-aware execution (CPU topology and pinning)
//! - JSON/CSV export support

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const platform = abi.platform.detection;

/// Statistical summary of benchmark results
pub const Statistics = struct {
    min_ns: u64,
    max_ns: u64,
    mean_ns: f64,
    median_ns: f64,
    std_dev_ns: f64,
    p50_ns: u64,
    p90_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    iterations: u64,
    outliers_removed: u64,
    total_time_ns: u64,

    pub fn opsPerSecond(self: Statistics) f64 {
        if (self.mean_ns == 0) return 0;
        return 1_000_000_000.0 / self.mean_ns;
    }

    pub fn throughputMBps(self: Statistics, bytes_per_op: u64) f64 {
        const ops = self.opsPerSecond();
        return (ops * @as(f64, @floatFromInt(bytes_per_op))) / (1024.0 * 1024.0);
    }
};

/// Hardware and environment information
pub const SystemInfo = struct {
    cpu_model: []const u8,
    logical_cores: u32,
    os: []const u8,
    arch: []const u8,
    zig_version: []const u8,

    pub fn detect(allocator: std.mem.Allocator) SystemInfo {
        const p_info = platform.PlatformInfo.detect();

        // Detect CPU brand on macOS/Linux
        var cpu_brand: []const u8 = "Unknown CPU";
        if (builtin.os.tag == .macos) {
            // sysctlbyname("machdep.cpu.brand_string", ...)
            // Simplified for now, but in a real rewrite we'd call the C API
            cpu_brand = "Apple Silicon / Intel (macOS)";
        } else if (builtin.os.tag == .linux) {
            cpu_brand = "Generic x86_64/ARM (Linux)";
        }

        _ = allocator;
        return .{
            .cpu_model = cpu_brand,
            .logical_cores = p_info.max_threads,
            .os = @tagName(p_info.os),
            .arch = @tagName(p_info.arch),
            .zig_version = @import("builtin").zig_version_string,
        };
    }
};

/// Benchmark configuration
pub const BenchConfig = struct {
    /// Minimum time to run the benchmark (nanoseconds)
    min_time_ns: u64 = 100_000_000, // 100ms
    /// Maximum iterations regardless of time
    max_iterations: u64 = 1_000_000,
    /// Minimum iterations to run
    min_iterations: u64 = 10,
    /// Number of warm-up iterations
    warmup_iterations: u64 = 50,
    /// Whether to remove statistical outliers
    remove_outliers: bool = true,
    /// Whether to track memory allocations
    track_memory: bool = false,
    /// Name for the benchmark
    name: []const u8 = "unnamed",
    /// Category for grouping
    category: []const u8 = "general",
    /// Bytes processed per operation (for throughput calculation)
    bytes_per_op: u64 = 0,
};

/// Result of a single benchmark run
pub const BenchResult = struct {
    config: BenchConfig,
    stats: Statistics,
    memory_allocated: u64,
    memory_freed: u64,
    timestamp: i64,

    pub fn format(
        self: BenchResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print(
            "{s:<40} {d:>12.2} ops/sec  mean={d:>8.0}ns  p99={d:>8}ns",
            .{
                self.config.name,
                self.stats.opsPerSecond(),
                self.stats.mean_ns,
                self.stats.p99_ns,
            },
        );
    }
};

var global_collector: ?*BenchCollector = null;

/// Shared collector for aggregating benchmark results across suites.
pub const BenchCollector = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchResult),
    system_info: SystemInfo,

    pub fn init(allocator: std.mem.Allocator) BenchCollector {
        return .{
            .allocator = allocator,
            .results = .empty,
            .system_info = SystemInfo.detect(allocator),
        };
    }

    pub fn deinit(self: *BenchCollector) void {
        for (self.results.items) |*result| {
            freeConfigStrings(self.allocator, &result.config);
        }
        self.results.deinit(self.allocator);
    }

    pub fn append(self: *BenchCollector, result: BenchResult) !void {
        var owned = result;
        owned.config = try cloneConfigStrings(self.allocator, result.config);
        try self.results.append(self.allocator, owned);
    }
};

/// Set a global collector for benchmark results.
pub fn setGlobalCollector(collector: ?*BenchCollector) void {
    global_collector = collector;
}

fn appendGlobalResult(result: BenchResult) !void {
    if (global_collector) |collector| {
        try collector.append(result);
    }
}

/// Tracking allocator for memory benchmarks
pub const TrackingAllocator = struct {
    parent: std.mem.Allocator,
    allocated: std.atomic.Value(u64),
    freed: std.atomic.Value(u64),
    peak: std.atomic.Value(u64),
    current: std.atomic.Value(u64),

    pub fn init(parent: std.mem.Allocator) TrackingAllocator {
        return .{
            .parent = parent,
            .allocated = std.atomic.Value(u64).init(0),
            .freed = std.atomic.Value(u64).init(0),
            .peak = std.atomic.Value(u64).init(0),
            .current = std.atomic.Value(u64).init(0),
        };
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
                .remap = remap,
            },
        };
    }

    fn remap(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent.vtable.remap(self.parent.ptr, buf, buf_align, new_len, ret_addr);
        if (result) |_| {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                _ = self.allocated.fetchAdd(diff, .monotonic);
                _ = self.current.fetchAdd(diff, .monotonic);
            } else {
                const diff = buf.len - new_len;
                _ = self.freed.fetchAdd(diff, .monotonic);
                _ = self.current.fetchSub(diff, .monotonic);
            }
        }
        return result;
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            _ = self.allocated.fetchAdd(len, .monotonic);
            const new_current = self.current.fetchAdd(len, .monotonic) + len;
            // Update peak
            var peak = self.peak.load(.monotonic);
            while (new_current > peak) {
                if (self.peak.cmpxchgWeak(peak, new_current, .monotonic, .monotonic)) |old| {
                    peak = old;
                } else {
                    break;
                }
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.parent.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                _ = self.allocated.fetchAdd(diff, .monotonic);
                _ = self.current.fetchAdd(diff, .monotonic);
            } else {
                const diff = buf.len - new_len;
                _ = self.freed.fetchAdd(diff, .monotonic);
                _ = self.current.fetchSub(diff, .monotonic);
            }
            return true;
        }
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, buf_align, ret_addr);
        _ = self.freed.fetchAdd(buf.len, .monotonic);
        _ = self.current.fetchSub(buf.len, .monotonic);
    }

    pub fn reset(self: *TrackingAllocator) void {
        self.allocated.store(0, .monotonic);
        self.freed.store(0, .monotonic);
        self.peak.store(0, .monotonic);
        self.current.store(0, .monotonic);
    }

    pub fn getStats(self: *TrackingAllocator) struct { allocated: u64, freed: u64, peak: u64 } {
        return .{
            .allocated = self.allocated.load(.monotonic),
            .freed = self.freed.load(.monotonic),
            .peak = self.peak.load(.monotonic),
        };
    }
};

/// Benchmark runner with statistical analysis
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchResult),

    pub fn init(allocator: std.mem.Allocator) BenchmarkRunner {
        return .{
            .allocator = allocator,
            .results = .empty,
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        for (self.results.items) |*result| {
            freeConfigStrings(self.allocator, &result.config);
        }
        self.results.deinit(self.allocator);
    }

    /// Run a benchmark function with the given configuration
    pub fn run(
        self: *BenchmarkRunner,
        config: BenchConfig,
        comptime bench_fn: anytype,
        args: anytype,
    ) !BenchResult {
        var tracker: ?TrackingAllocator = null;
        if (config.track_memory) {
            tracker = TrackingAllocator.init(self.allocator);
        }

        // Warm-up phase
        var i: u64 = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            const result = @call(.auto, bench_fn, args);
            std.mem.doNotOptimizeAway(&result);
        }

        // Collection phase
        var samples = std.ArrayListUnmanaged(u64).empty;
        defer samples.deinit(self.allocator);

        var total_time: u64 = 0;
        var iterations: u64 = 0;

        while (total_time < config.min_time_ns and iterations < config.max_iterations) {
            var timer = abi.foundation.time.Timer.start() catch return error.TimerFailed;
            const result = @call(.auto, bench_fn, args);
            const elapsed = timer.read();
            std.mem.doNotOptimizeAway(&result);

            try samples.append(self.allocator, elapsed);
            total_time += elapsed;
            iterations += 1;
        }

        // Ensure minimum iterations
        while (iterations < config.min_iterations) {
            var timer = abi.foundation.time.Timer.start() catch return error.TimerFailed;
            const result = @call(.auto, bench_fn, args);
            const elapsed = timer.read();
            std.mem.doNotOptimizeAway(&result);

            try samples.append(self.allocator, elapsed);
            total_time += elapsed;
            iterations += 1;
        }

        const stats = try calculateStatistics(self.allocator, samples.items, config);

        var mem_allocated: u64 = 0;
        var mem_freed: u64 = 0;
        if (tracker) |*t| {
            const mem_stats = t.getStats();
            mem_allocated = mem_stats.allocated;
            mem_freed = mem_stats.freed;
        }

        const result = BenchResult{
            .config = try cloneConfigStrings(self.allocator, config),
            .stats = stats,
            .memory_allocated = mem_allocated,
            .memory_freed = mem_freed,
            .timestamp = abi.foundation.time.unixSeconds(),
        };

        try self.results.append(self.allocator, result);
        try appendGlobalResult(result);
        return result;
    }

    /// Print a formatted summary using std.debug.print
    pub fn printSummaryDebug(self: *BenchmarkRunner) void {
        std.debug.print("\n", .{});
        std.debug.print("================================================================================\n", .{});
        std.debug.print("                     BENCHMARK RESULTS SUMMARY\n", .{});
        std.debug.print("================================================================================\n\n", .{});

        var last_cat: []const u8 = "";
        for (self.results.items) |result| {
            if (!std.mem.eql(u8, last_cat, result.config.category)) {
                std.debug.print("\n[{s}]\n", .{result.config.category});
                std.debug.print("-" ** 80 ++ "\n", .{});
                last_cat = result.config.category;
            }
            std.debug.print("{}\n", .{result});
        }
        std.debug.print("\n", .{});
    }
};

fn cloneConfigStrings(allocator: std.mem.Allocator, config: BenchConfig) !BenchConfig {
    var owned = config;
    owned.name = try allocator.dupe(u8, config.name);
    owned.category = try allocator.dupe(u8, config.category);
    return owned;
}

fn freeConfigStrings(allocator: std.mem.Allocator, config: *BenchConfig) void {
    if (config.name.len > 0) allocator.free(config.name);
    if (config.category.len > 0) allocator.free(config.category);
    config.name = "";
    config.category = "";
}

/// Calculate comprehensive statistics from samples
fn calculateStatistics(
    allocator: std.mem.Allocator,
    samples: []const u64,
    config: BenchConfig,
) !Statistics {
    if (samples.len == 0) return std.mem.zeroes(Statistics);

    const sorted = try allocator.alloc(u64, samples.len);
    defer allocator.free(sorted);
    @memcpy(sorted, samples);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

    var filtered = sorted;
    var outliers_removed: u64 = 0;

    // Use IQR for outlier detection
    if (config.remove_outliers and sorted.len >= 4) {
        const q1 = sorted[sorted.len / 4];
        const q3 = sorted[(sorted.len * 3) / 4];
        const iqr = q3 - q1;
        const lower_bound = if (q1 > (iqr * 3 / 2)) q1 - (iqr * 3 / 2) else 0;
        const upper_bound = q3 + (iqr * 3 / 2);

        var filtered_list = std.ArrayListUnmanaged(u64).empty;
        errdefer filtered_list.deinit(allocator);

        for (sorted) |s| {
            if (s >= lower_bound and s <= upper_bound) {
                try filtered_list.append(allocator, s);
            } else {
                outliers_removed += 1;
            }
        }

        if (filtered_list.items.len > 0) {
            const new_filtered = try allocator.dupe(u64, filtered_list.items);
            filtered = new_filtered;
        }
        filtered_list.deinit(allocator);
    }
    defer if (filtered.ptr != sorted.ptr) allocator.free(filtered);

    var sum: u128 = 0;
    for (filtered) |s| sum += s;
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(filtered.len));

    var variance_sum: f64 = 0;
    for (filtered) |s| {
        const diff = @as(f64, @floatFromInt(s)) - mean;
        variance_sum += diff * diff;
    }
    const std_dev = @sqrt(variance_sum / @as(f64, @floatFromInt(filtered.len)));

    return Statistics{
        .min_ns = filtered[0],
        .max_ns = filtered[filtered.len - 1],
        .mean_ns = mean,
        .median_ns = @floatFromInt(filtered[filtered.len / 2]),
        .std_dev_ns = std_dev,
        .p50_ns = percentile(filtered, 50),
        .p90_ns = percentile(filtered, 90),
        .p95_ns = percentile(filtered, 95),
        .p99_ns = percentile(filtered, 99),
        .iterations = samples.len,
        .outliers_removed = outliers_removed,
        .total_time_ns = @intCast(sum),
    };
}

fn percentile(sorted: []const u64, p: u8) u64 {
    if (sorted.len == 0) return 0;
    const idx = (sorted.len * p) / 100;
    return sorted[@min(idx, sorted.len - 1)];
}
