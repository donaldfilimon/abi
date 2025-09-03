//! Test framework utilities for WDBX-AI

const std = @import("std");
const wdbx = @import("wdbx");

/// Test context for managing test resources
pub const TestContext = struct {
    allocator: std.mem.Allocator,
    temp_dir: ?std.fs.Dir,
    temp_path: ?[]u8,
    logger: ?*wdbx.core.logging.Logger,
    
    pub fn init(allocator: std.mem.Allocator) !TestContext {
        return .{
            .allocator = allocator,
            .temp_dir = null,
            .temp_path = null,
            .logger = null,
        };
    }
    
    pub fn deinit(self: *TestContext) void {
        if (self.logger) |logger| {
            logger.deinit();
        }
        
        if (self.temp_dir) |*dir| {
            dir.close();
        }
        
        if (self.temp_path) |path| {
            std.fs.cwd().deleteTree(path) catch {};
            self.allocator.free(path);
        }
    }
    
    pub fn getTempDir(self: *TestContext) !std.fs.Dir {
        if (self.temp_dir) |dir| {
            return dir;
        }
        
        const temp_name = try std.fmt.allocPrint(self.allocator, "wdbx_test_{d}", .{std.time.timestamp()});
        defer self.allocator.free(temp_name);
        
        try std.fs.cwd().makePath(temp_name);
        self.temp_path = try self.allocator.dupe(u8, temp_name);
        self.temp_dir = try std.fs.cwd().openDir(temp_name, .{});
        
        return self.temp_dir.?;
    }
    
    pub fn getTempPath(self: *TestContext, name: []const u8) ![]u8 {
        const dir_path = self.temp_path orelse blk: {
            _ = try self.getTempDir();
            break :blk self.temp_path.?;
        };
        
        return try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, name });
    }
    
    pub fn getLogger(self: *TestContext) !*wdbx.core.logging.Logger {
        if (self.logger) |logger| {
            return logger;
        }
        
        const logger = try self.allocator.create(wdbx.core.logging.Logger);
        logger.* = try wdbx.core.logging.Logger.init(self.allocator, .{
            .level = .trace,
            .output = .stdout,
            .use_color = true,
        });
        self.logger = logger;
        
        return logger;
    }
};

/// Test data generators
pub const TestData = struct {
    /// Generate random vectors
    pub fn generateVectors(allocator: std.mem.Allocator, count: usize, dimensions: usize) ![][]f32 {
        var vectors = try allocator.alloc([]f32, count);
        errdefer allocator.free(vectors);
        
        var prng = std.rand.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const random = prng.random();
        
        for (vectors) |*vector| {
            vector.* = try allocator.alloc(f32, dimensions);
            for (vector.*) |*val| {
                val.* = random.float(f32) * 2.0 - 1.0; // [-1, 1]
            }
        }
        
        return vectors;
    }
    
    /// Generate normalized vectors
    pub fn generateNormalizedVectors(allocator: std.mem.Allocator, count: usize, dimensions: usize) ![][]f32 {
        const vectors = try generateVectors(allocator, count, dimensions);
        
        for (vectors) |vector| {
            var magnitude: f32 = 0;
            for (vector) |val| {
                magnitude += val * val;
            }
            magnitude = @sqrt(magnitude);
            
            if (magnitude > 0) {
                for (vector) |*val| {
                    val.* /= magnitude;
                }
            }
        }
        
        return vectors;
    }
    
    /// Generate clustered vectors
    pub fn generateClusteredVectors(
        allocator: std.mem.Allocator,
        clusters: usize,
        vectors_per_cluster: usize,
        dimensions: usize,
    ) ![][]f32 {
        const total = clusters * vectors_per_cluster;
        var vectors = try allocator.alloc([]f32, total);
        
        var prng = std.rand.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const random = prng.random();
        
        // Generate cluster centers
        const centers = try generateVectors(allocator, clusters, dimensions);
        defer {
            for (centers) |center| {
                allocator.free(center);
            }
            allocator.free(centers);
        }
        
        // Generate vectors around centers
        var idx: usize = 0;
        for (centers) |center| {
            var i: usize = 0;
            while (i < vectors_per_cluster) : (i += 1) {
                vectors[idx] = try allocator.alloc(f32, dimensions);
                for (vectors[idx], center) |*val, center_val| {
                    val.* = center_val + (random.float(f32) - 0.5) * 0.2; // Small variance
                }
                idx += 1;
            }
        }
        
        return vectors;
    }
    
    /// Free generated vectors
    pub fn freeVectors(allocator: std.mem.Allocator, vectors: [][]f32) void {
        for (vectors) |vector| {
            allocator.free(vector);
        }
        allocator.free(vectors);
    }
};

/// Performance benchmarking utilities
pub const Benchmark = struct {
    name: []const u8,
    iterations: usize,
    warmup_iterations: usize,
    measurements: std.ArrayList(u64),
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Benchmark {
        return .{
            .name = name,
            .iterations = 1000,
            .warmup_iterations = 10,
            .measurements = std.ArrayList(u64).init(allocator),
        };
    }
    
    pub fn deinit(self: *Benchmark) void {
        self.measurements.deinit();
    }
    
    pub fn run(
        self: *Benchmark,
        comptime func: anytype,
        args: anytype,
    ) !void {
        // Warmup
        var i: usize = 0;
        while (i < self.warmup_iterations) : (i += 1) {
            _ = try @call(.auto, func, args);
        }
        
        // Actual measurements
        i = 0;
        while (i < self.iterations) : (i += 1) {
            const start = std.time.nanoTimestamp();
            _ = try @call(.auto, func, args);
            const end = std.time.nanoTimestamp();
            
            try self.measurements.append(@as(u64, @intCast(end - start)));
        }
    }
    
    pub fn getStats(self: *Benchmark) BenchmarkStats {
        if (self.measurements.items.len == 0) {
            return BenchmarkStats{};
        }
        
        // Sort measurements
        std.mem.sort(u64, self.measurements.items, {}, std.sort.asc(u64));
        
        // Calculate statistics
        var sum: u64 = 0;
        var min: u64 = std.math.maxInt(u64);
        var max: u64 = 0;
        
        for (self.measurements.items) |measurement| {
            sum += measurement;
            min = @min(min, measurement);
            max = @max(max, measurement);
        }
        
        const avg = sum / self.measurements.items.len;
        const median = self.measurements.items[self.measurements.items.len / 2];
        const p95 = self.measurements.items[@min(
            (self.measurements.items.len * 95) / 100,
            self.measurements.items.len - 1
        )];
        const p99 = self.measurements.items[@min(
            (self.measurements.items.len * 99) / 100,
            self.measurements.items.len - 1
        )];
        
        return .{
            .min = min,
            .max = max,
            .avg = avg,
            .median = median,
            .p95 = p95,
            .p99 = p99,
            .iterations = self.iterations,
        };
    }
    
    pub fn report(self: *Benchmark, writer: anytype) !void {
        const stats = self.getStats();
        
        try writer.print("\nBenchmark: {s}\n", .{self.name});
        try writer.print("  Iterations: {d}\n", .{stats.iterations});
        try writer.print("  Min:        {}\n", .{std.fmt.fmtDuration(stats.min)});
        try writer.print("  Max:        {}\n", .{std.fmt.fmtDuration(stats.max)});
        try writer.print("  Average:    {}\n", .{std.fmt.fmtDuration(stats.avg)});
        try writer.print("  Median:     {}\n", .{std.fmt.fmtDuration(stats.median)});
        try writer.print("  P95:        {}\n", .{std.fmt.fmtDuration(stats.p95)});
        try writer.print("  P99:        {}\n", .{std.fmt.fmtDuration(stats.p99)});
    }
};

pub const BenchmarkStats = struct {
    min: u64 = 0,
    max: u64 = 0,
    avg: u64 = 0,
    median: u64 = 0,
    p95: u64 = 0,
    p99: u64 = 0,
    iterations: usize = 0,
};

/// Assertion helpers
pub const assert = struct {
    pub fn vectorsEqual(actual: []const f32, expected: []const f32, tolerance: f32) !void {
        try std.testing.expectEqual(actual.len, expected.len);
        
        for (actual, expected, 0..) |a, e, i| {
            if (@abs(a - e) > tolerance) {
                std.debug.print("Vector mismatch at index {d}: {d} != {d}\n", .{ i, a, e });
                return error.VectorMismatch;
            }
        }
    }
    
    pub fn vectorsNear(actual: []const f32, expected: []const f32, max_distance: f32) !void {
        const distance = wdbx.database.Metric.euclidean.distance(actual, expected);
        if (distance > max_distance) {
            std.debug.print("Vectors too far apart: distance = {d}, max = {d}\n", .{ distance, max_distance });
            return error.VectorsTooFarApart;
        }
    }
};

/// Mock implementations for testing
pub const Mock = struct {
    /// Mock database for testing
    pub const Database = struct {
        allocator: std.mem.Allocator,
        vectors: std.StringHashMap([]f32),
        call_count: std.StringHashMap(usize),
        
        pub fn init(allocator: std.mem.Allocator) Database {
            return .{
                .allocator = allocator,
                .vectors = std.StringHashMap([]f32).init(allocator),
                .call_count = std.StringHashMap(usize).init(allocator),
            };
        }
        
        pub fn deinit(self: *Database) void {
            var it = self.vectors.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            self.vectors.deinit();
            self.call_count.deinit();
        }
        
        pub fn insert(self: *Database, id: []const u8, vector: []const f32) !void {
            const id_copy = try self.allocator.dupe(u8, id);
            const vector_copy = try self.allocator.dupe(f32, vector);
            
            try self.vectors.put(id_copy, vector_copy);
            
            const count = self.call_count.get("insert") orelse 0;
            try self.call_count.put("insert", count + 1);
        }
        
        pub fn get(self: *Database, id: []const u8) ?[]f32 {
            const count = self.call_count.get("get") orelse 0;
            self.call_count.put("get", count + 1) catch {};
            
            return self.vectors.get(id);
        }
        
        pub fn getCallCount(self: *Database, method: []const u8) usize {
            return self.call_count.get(method) orelse 0;
        }
    };
};

/// Test runner with filtering and reporting
pub const TestRunner = struct {
    allocator: std.mem.Allocator,
    filter: ?[]const u8,
    verbose: bool,
    
    pub fn init(allocator: std.mem.Allocator) TestRunner {
        return .{
            .allocator = allocator,
            .filter = null,
            .verbose = false,
        };
    }
    
    pub fn run(self: *TestRunner) !void {
        const stdout = std.io.getStdOut().writer();
        
        try stdout.writeAll("\n");
        try stdout.writeAll("Running WDBX-AI Test Suite\n");
        try stdout.writeAll("==========================\n\n");
        
        var passed: usize = 0;
        var failed: usize = 0;
        var skipped: usize = 0;
        
        // Get all test declarations
        const test_fns = @import("builtin").test_functions;
        
        for (test_fns) |test_fn| {
            const name = test_fn.name;
            
            // Apply filter if set
            if (self.filter) |filter| {
                if (std.mem.indexOf(u8, name, filter) == null) {
                    skipped += 1;
                    continue;
                }
            }
            
            if (self.verbose) {
                try stdout.print("Running: {s}...", .{name});
            }
            
            // Run test
            test_fn.func() catch |err| {
                failed += 1;
                try stdout.print("\n❌ FAILED: {s}\n", .{name});
                try stdout.print("   Error: {}\n", .{err});
                continue;
            };
            
            passed += 1;
            if (self.verbose) {
                try stdout.print(" ✅\n", .{});
            } else {
                try stdout.print(".", .{});
            }
        }
        
        // Summary
        try stdout.writeAll("\n\n");
        try stdout.writeAll("Test Summary\n");
        try stdout.writeAll("============\n");
        try stdout.print("Passed:  {d}\n", .{passed});
        try stdout.print("Failed:  {d}\n", .{failed});
        try stdout.print("Skipped: {d}\n", .{skipped});
        try stdout.print("Total:   {d}\n", .{passed + failed + skipped});
        
        if (failed > 0) {
            return error.TestsFailed;
        }
    }
};