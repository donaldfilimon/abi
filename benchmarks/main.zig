//! WDBX-AI Performance Benchmarks
//!
//! Comprehensive benchmarking suite for all components of the WDBX-AI system

const std = @import("std");
const core = @import("../src/core/mod.zig");
const simd = @import("../src/simd/mod.zig");
const database = @import("../src/database/mod.zig");
const wdbx = @import("../src/wdbx/mod.zig");

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    warmup_iterations: usize = 100,
    benchmark_iterations: usize = 1000,
    vector_dimensions: []const usize = &[_]usize{ 64, 128, 256, 512, 1024, 2048 },
    vector_counts: []const usize = &[_]usize{ 100, 1000, 10000 },
    enable_memory_profiling: bool = true,
    enable_cpu_profiling: bool = true,
    output_csv: bool = false,
    csv_filename: []const u8 = "benchmark_results.csv",
};

/// Benchmark result
pub const BenchmarkResult = struct {
    name: []const u8,
    dimension: usize,
    vector_count: usize,
    iterations: usize,
    total_time_ms: f64,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    operations_per_second: f64,
    memory_usage_mb: f64,
    
    pub fn print(self: BenchmarkResult) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("{s} (dim={}, count={}):\n", .{ self.name, self.dimension, self.vector_count });
        try stdout.print("  Total Time: {d:.2}ms\n", .{self.total_time_ms});
        try stdout.print("  Average Time: {d:.4}ms\n", .{self.avg_time_ms});
        try stdout.print("  Min Time: {d:.4}ms\n", .{self.min_time_ms});
        try stdout.print("  Max Time: {d:.4}ms\n", .{self.max_time_ms});
        try stdout.print("  Operations/sec: {d:.0}\n", .{self.operations_per_second});
        try stdout.print("  Memory Usage: {d:.2}MB\n", .{self.memory_usage_mb});
        try stdout.print("\n");
    }
    
    pub fn toCsv(self: BenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        return try std.fmt.allocPrint(allocator,
            "{s},{},{},{},{d:.2},{d:.4},{d:.4},{d:.4},{d:.0},{d:.2}\n",
            .{
                self.name,
                self.dimension,
                self.vector_count,
                self.iterations,
                self.total_time_ms,
                self.avg_time_ms,
                self.min_time_ms,
                self.max_time_ms,
                self.operations_per_second,
                self.memory_usage_mb,
            }
        );
    }
};

/// Benchmark runner
pub const BenchmarkRunner = struct {
    config: BenchmarkConfig,
    results: std.ArrayList(BenchmarkResult),
    allocator: std.mem.Allocator,
    csv_file: ?std.fs.File = null,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !Self {
        var runner = Self{
            .config = config,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
            .allocator = allocator,
        };
        
        if (config.output_csv) {
            runner.csv_file = try std.fs.cwd().createFile(config.csv_filename, .{ .truncate = true });
            const csv_header = "Name,Dimension,VectorCount,Iterations,TotalTimeMs,AvgTimeMs,MinTimeMs,MaxTimeMs,OpsPerSec,MemoryUsageMB\n";
            _ = try runner.csv_file.?.write(csv_header);
        }
        
        return runner;
    }
    
    pub fn deinit(self: *Self) void {
        if (self.csv_file) |file| {
            file.close();
        }
        self.results.deinit();
    }
    
    pub fn runBenchmark(
        self: *Self,
        comptime name: []const u8,
        benchmark_func: anytype,
        dimension: usize,
        vector_count: usize,
    ) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("Running benchmark: {s} (dim={}, count={})...", .{ name, dimension, vector_count });
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            _ = try benchmark_func(self.allocator, dimension, vector_count);
        }
        
        // Actual benchmark
        var times = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(times);
        
        const memory_before = if (self.config.enable_memory_profiling) getCurrentMemoryUsage() else 0;
        
        for (times) |*time| {
            const start = std.time.nanoTimestamp();
            _ = try benchmark_func(self.allocator, dimension, vector_count);
            const end = std.time.nanoTimestamp();
            time.* = @as(f64, @floatFromInt(end - start)) / 1_000_000.0; // Convert to milliseconds
        }
        
        const memory_after = if (self.config.enable_memory_profiling) getCurrentMemoryUsage() else 0;
        const memory_usage_mb = @as(f64, @floatFromInt(memory_after - memory_before)) / (1024.0 * 1024.0);
        
        // Calculate statistics
        var total_time: f64 = 0;
        var min_time: f64 = std.math.inf(f64);
        var max_time: f64 = 0;
        
        for (times) |time| {
            total_time += time;
            min_time = @min(min_time, time);
            max_time = @max(max_time, time);
        }
        
        const avg_time = total_time / @as(f64, @floatFromInt(self.config.benchmark_iterations));
        const ops_per_second = if (avg_time > 0) 1000.0 / avg_time else 0;
        
        const result = BenchmarkResult{
            .name = name,
            .dimension = dimension,
            .vector_count = vector_count,
            .iterations = self.config.benchmark_iterations,
            .total_time_ms = total_time,
            .avg_time_ms = avg_time,
            .min_time_ms = min_time,
            .max_time_ms = max_time,
            .operations_per_second = ops_per_second,
            .memory_usage_mb = memory_usage_mb,
        };
        
        try self.results.append(result);
        try stdout.print(" ✅ ({d:.4}ms avg)\n", .{avg_time});
        
        if (self.csv_file) |file| {
            const csv_line = try result.toCsv(self.allocator);
            defer self.allocator.free(csv_line);
            _ = try file.write(csv_line);
        }
    }
    
    fn getCurrentMemoryUsage() usize {
        // Simple approximation - in a real implementation, this would
        // query the system for actual memory usage
        return 0;
    }
    
    pub fn runAllBenchmarks(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("Starting WDBX-AI Performance Benchmarks...\n");
        
        // Display CPU features
        const features = simd.CpuFeatures.detect();
        try features.print();
        try stdout.print("\n");
        
        // SIMD benchmarks
        try stdout.print("=== SIMD Benchmarks ===\n");
        for (self.config.vector_dimensions) |dim| {
            try self.runBenchmark("SIMD Distance Calculation", benchmarkSimdDistance, dim, 1000);
            try self.runBenchmark("SIMD Matrix Multiplication", benchmarkMatrixMultiply, dim, 100);
            try self.runBenchmark("SIMD Vector Normalization", benchmarkNormalization, dim, 1000);
        }
        
        // Database benchmarks
        try stdout.print("=== Database Benchmarks ===\n");
        for (self.config.vector_dimensions) |dim| {
            for (self.config.vector_counts) |count| {
                try self.runBenchmark("Database Insert", benchmarkDatabaseInsert, dim, count);
                try self.runBenchmark("Database Search", benchmarkDatabaseSearch, dim, count);
            }
        }
        
        // WDBX benchmarks
        try stdout.print("=== WDBX Unified Benchmarks ===\n");
        for (self.config.vector_dimensions) |dim| {
            try self.runBenchmark("WDBX Unified Operations", benchmarkWdbxUnified, dim, 1000);
        }
        
        // Print summary
        try self.printSummary();
    }
    
    pub fn printSummary(self: Self) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("\n" ++ "=" ** 60 ++ "\n");
        try stdout.print("BENCHMARK SUMMARY\n");
        try stdout.print("=" ** 60 ++ "\n");
        
        // Group results by benchmark name
        var grouped = std.HashMap([]const u8, std.ArrayList(BenchmarkResult), std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer {
            var iterator = grouped.iterator();
            while (iterator.next()) |entry| {
                entry.value_ptr.deinit();
            }
            grouped.deinit();
        }
        
        for (self.results.items) |result| {
            const entry = try grouped.getOrPut(result.name);
            if (!entry.found_existing) {
                entry.value_ptr.* = std.ArrayList(BenchmarkResult).init(self.allocator);
            }
            try entry.value_ptr.append(result);
        }
        
        // Print grouped results
        var iterator = grouped.iterator();
        while (iterator.next()) |entry| {
            const name = entry.key_ptr.*;
            const results = entry.value_ptr.items;
            
            try stdout.print("\n{s}:\n", .{name});
            
            var total_ops_per_sec: f64 = 0;
            var best_ops_per_sec: f64 = 0;
            
            for (results) |result| {
                total_ops_per_sec += result.operations_per_second;
                best_ops_per_sec = @max(best_ops_per_sec, result.operations_per_second);
                try stdout.print("  dim={:>4}, count={:>5}: {d:>8.0} ops/sec\n", .{ result.dimension, result.vector_count, result.operations_per_second });
            }
            
            const avg_ops_per_sec = total_ops_per_sec / @as(f64, @floatFromInt(results.len));
            try stdout.print("  Average: {d:.0} ops/sec\n", .{avg_ops_per_sec});
            try stdout.print("  Best: {d:.0} ops/sec\n", .{best_ops_per_sec});
        }
        
        try stdout.print("\n" ++ "=" ** 60 ++ "\n");
    }
};

// Individual benchmark functions
fn benchmarkSimdDistance(allocator: std.mem.Allocator, dimension: usize, vector_count: usize) !void {
    const query = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    defer allocator.free(query);
    
    var vectors = try allocator.alloc([]f32, vector_count);
    defer {
        for (vectors) |vector| {
            allocator.free(vector);
        }
        allocator.free(vectors);
    }
    
    for (vectors) |*vector| {
        vector.* = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    }
    
    // Benchmark distance calculations
    for (vectors) |vector| {
        _ = simd.DistanceOps.euclideanDistance(query, vector);
    }
}

fn benchmarkMatrixMultiply(allocator: std.mem.Allocator, dimension: usize, count: usize) !void {
    _ = count;
    
    const matrix_a = try core.random.vector(f32, allocator, dimension * dimension, -1.0, 1.0);
    defer allocator.free(matrix_a);
    
    const matrix_b = try core.random.vector(f32, allocator, dimension * dimension, -1.0, 1.0);
    defer allocator.free(matrix_b);
    
    var result = try allocator.alloc(f32, dimension * dimension);
    defer allocator.free(result);
    
    simd.MatrixOps.multiply(matrix_a, matrix_b, result, dimension, dimension, dimension);
}

fn benchmarkNormalization(allocator: std.mem.Allocator, dimension: usize, vector_count: usize) !void {
    var vectors = try allocator.alloc([]f32, vector_count);
    defer {
        for (vectors) |vector| {
            allocator.free(vector);
        }
        allocator.free(vectors);
    }
    
    for (vectors) |*vector| {
        vector.* = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    }
    
    for (vectors) |vector| {
        simd.NormalizationOps.normalize(vector);
    }
}

fn benchmarkDatabaseInsert(allocator: std.mem.Allocator, dimension: usize, vector_count: usize) !void {
    const test_file = "benchmark_insert.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try database.createStandard(test_file, true);
    defer db.close();
    try db.init(@intCast(dimension));
    
    var vectors = try allocator.alloc([]f32, vector_count);
    defer {
        for (vectors) |vector| {
            allocator.free(vector);
        }
        allocator.free(vectors);
    }
    
    for (vectors) |*vector| {
        vector.* = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    }
    
    for (vectors) |vector| {
        _ = try db.addEmbedding(vector);
    }
}

fn benchmarkDatabaseSearch(allocator: std.mem.Allocator, dimension: usize, vector_count: usize) !void {
    const test_file = "benchmark_search.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try database.createStandard(test_file, true);
    defer db.close();
    try db.init(@intCast(dimension));
    
    // Pre-populate database
    for (0..vector_count) |_| {
        const vector = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
        defer allocator.free(vector);
        _ = try db.addEmbedding(vector);
    }
    
    // Benchmark search operations
    const query = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    defer allocator.free(query);
    
    const results = try db.search(query, 10, allocator);
    defer allocator.free(results);
}

fn benchmarkWdbxUnified(allocator: std.mem.Allocator, dimension: usize, vector_count: usize) !void {
    const test_file = "benchmark_wdbx_unified.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var config = wdbx.UnifiedConfig.createDefault(@intCast(dimension));
    config.enable_profiling = false; // Disable for cleaner benchmarks
    
    var db = try wdbx.createUnified(allocator, test_file, config);
    defer db.deinit();
    
    // Benchmark insert operations
    for (0..vector_count) |_| {
        const vector = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
        defer allocator.free(vector);
        _ = try db.addVector(vector);
    }
    
    // Benchmark search operations
    const query = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
    defer allocator.free(query);
    
    const results = try db.search(query, 10);
    defer {
        for (results) |*result| {
            result.deinit(allocator);
        }
        allocator.free(results);
    }
}

/// Run comprehensive benchmarks
pub fn runBenchmarks(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    // Initialize core system
    try core.init(allocator);
    defer core.deinit();
    
    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();
    
    try runner.runAllBenchmarks();
}

/// Default benchmark configuration
pub fn getDefaultConfig() BenchmarkConfig {
    return BenchmarkConfig{};
}

/// Quick benchmark configuration for development
pub fn getQuickConfig() BenchmarkConfig {
    return BenchmarkConfig{
        .warmup_iterations = 10,
        .benchmark_iterations = 100,
        .vector_dimensions = &[_]usize{ 64, 256, 1024 },
        .vector_counts = &[_]usize{ 100, 1000 },
    };
}

/// Comprehensive benchmark configuration
pub fn getComprehensiveConfig() BenchmarkConfig {
    return BenchmarkConfig{
        .warmup_iterations = 1000,
        .benchmark_iterations = 10000,
        .vector_dimensions = &[_]usize{ 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048 },
        .vector_counts = &[_]usize{ 10, 100, 1000, 10000, 100000 },
        .output_csv = true,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    const config = if (args.len > 1 and std.mem.eql(u8, args[1], "comprehensive"))
        getComprehensiveConfig()
    else if (args.len > 1 and std.mem.eql(u8, args[1], "quick"))
        getQuickConfig()
    else
        getDefaultConfig();
    
    try runBenchmarks(allocator, config);
}
