//! Comprehensive test runner for WDBX-AI system

const std = @import("std");
const testing = std.testing;

// Import all test modules
const test_core = @import("test_core_integration.zig");
const test_ai = @import("test_ai.zig");
const test_database = @import("test_database.zig");
const test_database_hnsw = @import("test_database_hnsw.zig");
const test_database_integration = @import("test_database_integration.zig");
const test_memory_management = @import("test_memory_management.zig");
const test_performance_optimizations = @import("test_performance_optimizations.zig");
const test_simd_vector = @import("test_simd_vector.zig");
const test_weather = @import("test_weather.zig");
const test_web_server = @import("test_web_server.zig");
const test_cli_integration = @import("test_cli_integration.zig");

/// Test suite configuration
pub const TestConfig = struct {
    enable_performance_tests: bool = true,
    enable_integration_tests: bool = true,
    enable_stress_tests: bool = false,
    verbose_output: bool = false,
    parallel_execution: bool = true,
    timeout_seconds: u32 = 300, // 5 minutes
};

/// Test result
pub const TestResult = struct {
    name: []const u8,
    passed: bool,
    duration_ms: f64,
    error_message: ?[]const u8 = null,
    
    pub fn deinit(self: *TestResult, allocator: std.mem.Allocator) void {
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

/// Test suite results
pub const TestSuiteResults = struct {
    results: std.ArrayList(TestResult),
    total_tests: usize,
    passed_tests: usize,
    failed_tests: usize,
    total_duration_ms: f64,
    allocator: std.mem.Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .results = std.ArrayList(TestResult).init(allocator),
            .total_tests = 0,
            .passed_tests = 0,
            .failed_tests = 0,
            .total_duration_ms = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.results.items) |*result| {
            result.deinit(self.allocator);
        }
        self.results.deinit();
    }
    
    pub fn addResult(self: *Self, result: TestResult) !void {
        try self.results.append(result);
        self.total_tests += 1;
        self.total_duration_ms += result.duration_ms;
        
        if (result.passed) {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
    }
    
    pub fn getSuccessRate(self: Self) f64 {
        if (self.total_tests == 0) return 1.0;
        return @as(f64, @floatFromInt(self.passed_tests)) / @as(f64, @floatFromInt(self.total_tests));
    }
    
    pub fn printSummary(self: Self) !void {
        const stdout = std.io.getStdOut().writer();
        
        try stdout.print("\n" ++ "=" ** 60 ++ "\n");
        try stdout.print("TEST SUITE RESULTS\n");
        try stdout.print("=" ** 60 ++ "\n");
        try stdout.print("Total Tests: {}\n", .{self.total_tests});
        try stdout.print("Passed: {} ({d:.1}%)\n", .{ self.passed_tests, self.getSuccessRate() * 100.0 });
        try stdout.print("Failed: {}\n", .{self.failed_tests});
        try stdout.print("Total Duration: {d:.2}ms\n", .{self.total_duration_ms});
        try stdout.print("Average Duration: {d:.2}ms\n", .{if (self.total_tests > 0) self.total_duration_ms / @as(f64, @floatFromInt(self.total_tests)) else 0.0});
        
        // Show failed tests
        if (self.failed_tests > 0) {
            try stdout.print("\nFAILED TESTS:\n");
            for (self.results.items) |result| {
                if (!result.passed) {
                    try stdout.print("  ❌ {s} ({d:.2}ms)", .{ result.name, result.duration_ms });
                    if (result.error_message) |msg| {
                        try stdout.print(" - {s}", .{msg});
                    }
                    try stdout.print("\n");
                }
            }
        }
        
        // Show slowest tests
        try stdout.print("\nSLOWEST TESTS:\n");
        var sorted_results = try self.allocator.dupe(TestResult, self.results.items);
        defer self.allocator.free(sorted_results);
        
        std.sort.insertion(TestResult, sorted_results, {}, struct {
            fn lessThan(context: void, a: TestResult, b: TestResult) bool {
                _ = context;
                return a.duration_ms > b.duration_ms; // Descending order
            }
        }.lessThan);
        
        const slowest_count = @min(5, sorted_results.len);
        for (sorted_results[0..slowest_count]) |result| {
            const status = if (result.passed) "✅" else "❌";
            try stdout.print("  {s} {s} ({d:.2}ms)\n", .{ status, result.name, result.duration_ms });
        }
        
        try stdout.print("=" ** 60 ++ "\n");
    }
};

/// Test runner
pub const TestRunner = struct {
    config: TestConfig,
    results: TestSuiteResults,
    allocator: std.mem.Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, config: TestConfig) Self {
        return Self{
            .config = config,
            .results = TestSuiteResults.init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.results.deinit();
    }
    
    pub fn runTest(self: *Self, comptime test_name: []const u8, test_func: anytype) !void {
        const start_time = std.time.nanoTimestamp();
        
        if (self.config.verbose_output) {
            std.debug.print("Running test: {s}...", .{test_name});
        }
        
        var result = TestResult{
            .name = test_name,
            .passed = false,
            .duration_ms = 0,
        };
        
        // Run the test with timeout
        const test_result = self.runWithTimeout(test_func, self.config.timeout_seconds);
        
        const end_time = std.time.nanoTimestamp();
        result.duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        
        switch (test_result) {
            .success => {
                result.passed = true;
                if (self.config.verbose_output) {
                    std.debug.print(" ✅ ({d:.2}ms)\n", .{result.duration_ms});
                }
            },
            .failure => |err| {
                result.passed = false;
                result.error_message = try self.allocator.dupe(u8, @errorName(err));
                if (self.config.verbose_output) {
                    std.debug.print(" ❌ ({d:.2}ms) - {s}\n", .{ result.duration_ms, @errorName(err) });
                }
            },
            .timeout => {
                result.passed = false;
                result.error_message = try self.allocator.dupe(u8, "Test timeout");
                if (self.config.verbose_output) {
                    std.debug.print(" ⏰ ({d:.2}ms) - Timeout\n", .{result.duration_ms});
                }
            },
        }
        
        try self.results.addResult(result);
    }
    
    fn runWithTimeout(self: Self, test_func: anytype, timeout_seconds: u32) TestResultType {
        _ = self;
        _ = timeout_seconds;
        
        // For now, just run the test directly (timeout implementation would be more complex)
        test_func() catch |err| {
            return TestResultType{ .failure = err };
        };
        
        return TestResultType.success;
    }
    
    const TestResultType = union(enum) {
        success: void,
        failure: anyerror,
        timeout: void,
    };
    
    pub fn runAllTests(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("Starting WDBX-AI Test Suite...\n");
        try stdout.print("Configuration:\n");
        try stdout.print("  Performance Tests: {}\n", .{self.config.enable_performance_tests});
        try stdout.print("  Integration Tests: {}\n", .{self.config.enable_integration_tests});
        try stdout.print("  Stress Tests: {}\n", .{self.config.enable_stress_tests});
        try stdout.print("  Parallel Execution: {}\n", .{self.config.parallel_execution});
        try stdout.print("  Timeout: {}s\n", .{self.config.timeout_seconds});
        try stdout.print("\n");
        
        // Run core tests
        try self.runTest("Core System Initialization", testCoreInitialization);
        try self.runTest("String Utilities", testStringUtilities);
        try self.runTest("Time Utilities", testTimeUtilities);
        try self.runTest("Random Utilities", testRandomUtilities);
        try self.runTest("Performance Monitoring", testPerformanceMonitoring);
        try self.runTest("Memory Management", testMemoryManagement);
        try self.runTest("Error Handling", testErrorHandling);
        
        if (self.config.enable_performance_tests) {
            try self.runTest("SIMD Performance", testSimdPerformance);
            try self.runTest("Database Performance", testDatabasePerformance);
            try self.runTest("Memory Allocation Performance", testMemoryPerformance);
        }
        
        if (self.config.enable_integration_tests) {
            try self.runTest("Database Integration", testDatabaseIntegration);
            try self.runTest("AI Integration", testAIIntegration);
            try self.runTest("WDBX Integration", testWDBXIntegration);
        }
        
        if (self.config.enable_stress_tests) {
            try self.runTest("Memory Stress Test", testMemoryStress);
            try self.runTest("Concurrent Operations Stress", testConcurrencyStress);
            try self.runTest("Large Dataset Stress", testLargeDatasetStress);
        }
        
        try self.results.printSummary();
    }
};

// Individual test functions
fn testCoreInitialization() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    try testing.expect(core.isInitialized());
}

fn testStringUtilities() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    const trimmed = core.string.trim("  hello  ");
    try testing.expectEqualStrings("hello", trimmed);
    
    try testing.expect(core.string.startsWith("hello world", "hello"));
    try testing.expect(core.string.endsWith("hello world", "world"));
}

fn testTimeUtilities() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    const start = core.time.now();
    core.time.sleep(1);
    const end = core.time.now();
    try testing.expect(end > start);
}

fn testRandomUtilities() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    core.random.initWithSeed(42);
    const val = core.random.int(u32, 1, 100);
    try testing.expect(val >= 1 and val <= 100);
}

fn testPerformanceMonitoring() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    var timer = try core.performance.startTimer("test");
    core.time.sleep(1);
    core.performance.endTimer("test", timer);
    
    const stats = core.performance.getStats("test");
    try testing.expect(stats != null);
    try testing.expect(stats.?.count == 1);
}

fn testMemoryManagement() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    var pool = core.memory.MemoryPool.init(testing.allocator);
    defer pool.deinit();
    
    const data = try pool.allocator().alloc(u8, 100);
    try testing.expect(data.len == 100);
}

fn testErrorHandling() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    const error_info = core.errors.systemError(1001, "Test error");
    try core.errors.recordError(error_info);
    
    const stats = core.errors.getGlobalErrorStats();
    try testing.expect(stats != null);
}

fn testSimdPerformance() !void {
    const simd_mod = @import("../src/simd/mod.zig");
    
    const vector_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector_b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    
    const distance = simd_mod.VectorOps.distance(&vector_a, &vector_b);
    try testing.expect(distance > 0.0);
}

fn testDatabasePerformance() !void {
    const database_mod = @import("../src/database/mod.zig");
    
    try core.init(testing.allocator);
    defer core.deinit();
    
    const test_file = "test_db_perf.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try database_mod.createStandard(test_file, true);
    defer db.close();
    try db.init(64);
    
    // Performance test: insert many vectors
    const start_time = std.time.nanoTimestamp();
    for (0..100) |i| {
        const vector = [_]f32{@as(f32, @floatFromInt(i))} ** 64;
        _ = try db.addEmbedding(&vector);
    }
    const end_time = std.time.nanoTimestamp();
    
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    try testing.expect(duration_ms < 1000.0); // Should complete in under 1 second
}

fn testMemoryPerformance() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    try core.allocators.init(testing.allocator);
    defer core.allocators.deinit();
    
    if (core.allocators.getSmartAllocator()) |smart_alloc| {
        const start_time = std.time.nanoTimestamp();
        
        // Allocate and free many small objects
        for (0..1000) |_| {
            const data = try smart_alloc.alloc(u8, 64);
            smart_alloc.free(data);
        }
        
        const end_time = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        try testing.expect(duration_ms < 100.0); // Should complete quickly
    }
}

fn testDatabaseIntegration() !void {
    const wdbx_mod = @import("../src/wdbx/mod.zig");
    
    try core.init(testing.allocator);
    defer core.deinit();
    
    const test_file = "test_wdbx_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try wdbx_mod.createWithDefaults(testing.allocator, test_file, 4);
    defer db.deinit();
    
    const test_vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const id = try db.addVector(&test_vector);
    
    const results = try db.search(&test_vector, 1);
    defer {
        for (results) |*result| {
            result.deinit(testing.allocator);
        }
        testing.allocator.free(results);
    }
    
    try testing.expect(results.len == 1);
    try testing.expect(results[0].id == id);
}

fn testAIIntegration() !void {
    const ai_mod = @import("../src/ai/mod.zig");
    
    try core.init(testing.allocator);
    defer core.deinit();
    
    var network = try ai_mod.NeuralNetwork.init(testing.allocator, &[_]usize{4}, &[_]usize{2});
    defer network.deinit();
    
    try network.addDenseLayer(8, .relu);
    try network.addDenseLayer(2, .softmax);
    try network.compile();
    
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const output = try testing.allocator.alloc(f32, 2);
    defer testing.allocator.free(output);
    
    try network.forward(&input, output);
    try testing.expect(output.len == 2);
}

fn testWDBXIntegration() !void {
    const wdbx_mod = @import("../src/wdbx/mod.zig");
    
    try core.init(testing.allocator);
    defer core.deinit();
    
    const config = wdbx_mod.UnifiedConfig.createDefault(128);
    try config.validate();
    
    const test_file = "test_wdbx_unified.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try wdbx_mod.createUnified(testing.allocator, test_file, config);
    defer db.deinit();
    
    const test_vector = [_]f32{0.5} ** 128;
    const id = try db.addVector(&test_vector);
    
    const results = try db.search(&test_vector, 1);
    defer {
        for (results) |*result| {
            result.deinit(testing.allocator);
        }
        testing.allocator.free(results);
    }
    
    try testing.expect(results.len == 1);
    try testing.expect(results[0].id == id);
    try testing.expect(results[0].distance == 0.0);
}

fn testMemoryStress() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    try core.allocators.init(testing.allocator);
    defer core.allocators.deinit();
    
    if (core.allocators.getSmartAllocator()) |smart_alloc| {
        var allocations = std.ArrayList([]u8).init(testing.allocator);
        defer {
            for (allocations.items) |allocation| {
                smart_alloc.free(allocation);
            }
            allocations.deinit();
        }
        
        // Stress test: allocate many objects of varying sizes
        for (0..1000) |i| {
            const size = (i % 1000) + 16; // 16 to 1015 bytes
            const data = try smart_alloc.alloc(u8, size);
            try allocations.append(data);
        }
        
        try testing.expect(allocations.items.len == 1000);
    }
}

fn testConcurrencyStress() !void {
    try core.init(testing.allocator);
    defer core.deinit();
    
    var counter = std.atomic.Value(u32).init(0);
    var threads: [4]std.Thread = undefined;
    
    const worker = struct {
        fn run(c: *std.atomic.Value(u32)) void {
            for (0..250) |_| {
                _ = c.fetchAdd(1, .monotonic);
            }
        }
    }.run;
    
    // Start worker threads
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&counter});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    try testing.expect(counter.load(.monotonic) == 1000); // 4 threads * 250 increments
}

fn testLargeDatasetStress() !void {
    const wdbx_mod = @import("../src/wdbx/mod.zig");
    
    try core.init(testing.allocator);
    defer core.deinit();
    
    const test_file = "test_large_dataset.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var config = wdbx_mod.UnifiedConfig.createDefault(64);
    config.max_vectors = 10000;
    
    var db = try wdbx_mod.createUnified(testing.allocator, test_file, config);
    defer db.deinit();
    
    // Insert many vectors
    for (0..1000) |i| {
        const vector = [_]f32{@as(f32, @floatFromInt(i)) / 1000.0} ** 64;
        _ = try db.addVector(&vector);
    }
    
    // Perform searches
    const query = [_]f32{0.5} ** 64;
    const results = try db.search(&query, 10);
    defer {
        for (results) |*result| {
            result.deinit(testing.allocator);
        }
        testing.allocator.free(results);
    }
    
    try testing.expect(results.len <= 10);
}

/// Main test runner function
pub fn runTests(allocator: std.mem.Allocator, config: TestConfig) !void {
    var runner = TestRunner.init(allocator, config);
    defer runner.deinit();
    
    try runner.runAllTests();
    
    // Exit with error code if tests failed
    if (runner.results.failed_tests > 0) {
        std.process.exit(1);
    }
}

/// Default test configuration
pub fn getDefaultConfig() TestConfig {
    return TestConfig{};
}

/// Performance-focused test configuration
pub fn getPerformanceConfig() TestConfig {
    return TestConfig{
        .enable_performance_tests = true,
        .enable_integration_tests = false,
        .enable_stress_tests = false,
        .verbose_output = true,
    };
}

/// Full test suite configuration
pub fn getFullConfig() TestConfig {
    return TestConfig{
        .enable_performance_tests = true,
        .enable_integration_tests = true,
        .enable_stress_tests = true,
        .verbose_output = true,
        .timeout_seconds = 600, // 10 minutes for stress tests
    };
}

test "Test runner functionality" {
    var runner = TestRunner.init(testing.allocator, getDefaultConfig());
    defer runner.deinit();
    
    try runner.runTest("Sample Test", testCoreInitialization);
    
    try testing.expect(runner.results.total_tests == 1);
    try testing.expect(runner.results.passed_tests == 1);
    try testing.expect(runner.results.failed_tests == 0);
}
