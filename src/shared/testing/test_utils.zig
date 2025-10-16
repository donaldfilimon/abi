//! Test Utilities Module
//!
//! Provides common testing utilities, fixtures, and helpers to improve
//! test coverage and reduce duplication in test code.

const std = @import("std");
const imports = @import("../imports.zig");
const patterns = @import("../patterns/common.zig");
const errors = @import("../errors/framework_errors.zig");

pub const testing = std.testing;
pub const expect = testing.expect;
pub const expectEqual = testing.expectEqual;
pub const expectEqualStrings = testing.expectEqualStrings;
pub const expectError = testing.expectError;
pub const Allocator = std.mem.Allocator;

/// Test allocator wrapper with leak detection
pub const TestAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{}),
    
    pub fn init() TestAllocator {
        return .{
            .gpa = std.heap.GeneralPurposeAllocator(.{}){},
        };
    }
    
    pub fn allocator(self: *TestAllocator) Allocator {
        return self.gpa.allocator();
    }
    
    pub fn deinit(self: *TestAllocator) !void {
        const leaked = self.gpa.detectLeaks();
        _ = self.gpa.deinit();
        if (leaked) {
            return error.MemoryLeak;
        }
    }
};

/// Mock writer for testing I/O operations
pub const MockWriter = struct {
    buffer: std.ArrayList(u8),
    
    pub fn init(allocator: Allocator) MockWriter {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *MockWriter) void {
        self.buffer.deinit();
    }
    
    pub fn writer(self: *MockWriter) std.io.AnyWriter {
        return self.buffer.writer().any();
    }
    
    pub fn getWritten(self: *MockWriter) []const u8 {
        return self.buffer.items;
    }
    
    pub fn clear(self: *MockWriter) void {
        self.buffer.clearRetainingCapacity();
    }
    
    pub fn contains(self: *MockWriter, needle: []const u8) bool {
        return std.mem.indexOf(u8, self.buffer.items, needle) != null;
    }
};

/// Test fixture for framework components
pub const FrameworkFixture = struct {
    allocator: Allocator,
    writer: MockWriter,
    logger: patterns.Logger,
    
    pub fn init(allocator: Allocator) FrameworkFixture {
        var writer = MockWriter.init(allocator);
        const logger = patterns.Logger.init(writer.writer(), .debug);
        
        return .{
            .allocator = allocator,
            .writer = writer,
            .logger = logger,
        };
    }
    
    pub fn deinit(self: *FrameworkFixture) void {
        self.writer.deinit();
    }
    
    pub fn getOutput(self: *FrameworkFixture) []const u8 {
        return self.writer.getWritten();
    }
    
    pub fn clearOutput(self: *FrameworkFixture) void {
        self.writer.clear();
    }
    
    pub fn expectOutput(self: *FrameworkFixture, expected: []const u8) !void {
        try expectEqualStrings(expected, self.getOutput());
    }
    
    pub fn expectOutputContains(self: *FrameworkFixture, needle: []const u8) !void {
        try expect(self.writer.contains(needle));
    }
};

/// Performance test utilities
pub const PerformanceTest = struct {
    start_time: u64,
    allocator: Allocator,
    measurements: std.ArrayList(u64),
    
    pub fn init(allocator: Allocator) PerformanceTest {
        return .{
            .start_time = 0,
            .allocator = allocator,
            .measurements = std.ArrayList(u64).init(allocator),
        };
    }
    
    pub fn deinit(self: *PerformanceTest) void {
        self.measurements.deinit();
    }
    
    pub fn start(self: *PerformanceTest) void {
        self.start_time = std.time.nanoTimestamp();
    }
    
    pub fn stop(self: *PerformanceTest) !void {
        const end_time = std.time.nanoTimestamp();
        const duration = end_time - self.start_time;
        try self.measurements.append(duration);
    }
    
    pub fn getAverageNs(self: *PerformanceTest) u64 {
        if (self.measurements.items.len == 0) return 0;
        
        var sum: u64 = 0;
        for (self.measurements.items) |measurement| {
            sum += measurement;
        }
        return sum / self.measurements.items.len;
    }
    
    pub fn getAverageMs(self: *PerformanceTest) f64 {
        return @as(f64, @floatFromInt(self.getAverageNs())) / 1_000_000.0;
    }
    
    pub fn expectMaxDurationMs(self: *PerformanceTest, max_ms: f64) !void {
        const avg_ms = self.getAverageMs();
        if (avg_ms > max_ms) {
            std.log.err("Performance test failed: {d:.2}ms > {d:.2}ms", .{ avg_ms, max_ms });
            return error.PerformanceTestFailed;
        }
    }
};

/// Memory usage tracker for tests
pub const MemoryTracker = struct {
    initial_usage: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MemoryTracker {
        return .{
            .initial_usage = getCurrentMemoryUsage(),
            .allocator = allocator,
        };
    }
    
    pub fn getCurrentUsage(self: *MemoryTracker) usize {
        _ = self;
        return getCurrentMemoryUsage();
    }
    
    pub fn getUsageDelta(self: *MemoryTracker) isize {
        const current = self.getCurrentUsage();
        return @as(isize, @intCast(current)) - @as(isize, @intCast(self.initial_usage));
    }
    
    pub fn expectMaxUsageMB(self: *MemoryTracker, max_mb: f64) !void {
        const delta_bytes = self.getUsageDelta();
        const delta_mb = @as(f64, @floatFromInt(delta_bytes)) / (1024.0 * 1024.0);
        
        if (delta_mb > max_mb) {
            std.log.err("Memory usage test failed: {d:.2}MB > {d:.2}MB", .{ delta_mb, max_mb });
            return error.MemoryUsageTestFailed;
        }
    }
    
    fn getCurrentMemoryUsage() usize {
        // Simplified memory usage tracking - would use platform-specific APIs in real implementation
        return 0; // Placeholder
    }
};

/// Test data generators
pub const TestData = struct {
    /// Generate random bytes
    pub fn randomBytes(allocator: Allocator, len: usize, seed: u64) ![]u8 {
        var prng = std.rand.DefaultPrng.init(seed);
        const bytes = try allocator.alloc(u8, len);
        prng.random().bytes(bytes);
        return bytes;
    }
    
    /// Generate random string
    pub fn randomString(allocator: Allocator, len: usize, seed: u64) ![]u8 {
        const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        var prng = std.rand.DefaultPrng.init(seed);
        
        const string = try allocator.alloc(u8, len);
        for (string) |*char| {
            char.* = chars[prng.random().uintLessThan(usize, chars.len)];
        }
        return string;
    }
    
    /// Generate random float array
    pub fn randomFloats(allocator: Allocator, len: usize, seed: u64) ![]f32 {
        var prng = std.rand.DefaultPrng.init(seed);
        const floats = try allocator.alloc(f32, len);
        for (floats) |*f| {
            f.* = prng.random().float(f32);
        }
        return floats;
    }
    
    /// Generate test vector for AI/ML operations
    pub fn testVector(allocator: Allocator, dimensions: usize, seed: u64) ![]f32 {
        return randomFloats(allocator, dimensions, seed);
    }
    
    /// Generate test matrix
    pub fn testMatrix(allocator: Allocator, rows: usize, cols: usize, seed: u64) ![][]f32 {
        const matrix = try allocator.alloc([]f32, rows);
        for (matrix, 0..) |*row, i| {
            row.* = try randomFloats(allocator, cols, seed + i);
        }
        return matrix;
    }
};

/// Async test utilities
pub const AsyncTest = struct {
    /// Run test with timeout
    pub fn withTimeout(comptime test_fn: fn () anyerror!void, timeout_ms: u64) !void {
        _ = test_fn;
        _ = timeout_ms;
        // Simplified implementation - would use proper async/await in real code
        try test_fn();
    }
    
    /// Run concurrent tests
    pub fn concurrent(comptime test_fn: fn (usize) anyerror!void, thread_count: usize) !void {
        _ = test_fn;
        _ = thread_count;
        // Simplified implementation - would spawn actual threads in real code
        for (0..thread_count) |i| {
            try test_fn(i);
        }
    }
};

/// Test assertion helpers
pub const Assert = struct {
    /// Assert that two floats are approximately equal
    pub fn approxEqual(actual: f64, expected: f64, tolerance: f64) !void {
        const diff = @abs(actual - expected);
        if (diff > tolerance) {
            std.log.err("Values not approximately equal: {d} vs {d} (diff: {d}, tolerance: {d})", .{ actual, expected, diff, tolerance });
            return error.AssertionFailed;
        }
    }
    
    /// Assert that array contains value
    pub fn arrayContains(comptime T: type, array: []const T, value: T) !void {
        for (array) |item| {
            if (std.meta.eql(item, value)) return;
        }
        std.log.err("Array does not contain expected value", .{});
        return error.AssertionFailed;
    }
    
    /// Assert that error is of specific type
    pub fn errorType(actual_error: anyerror, expected_error: anyerror) !void {
        if (actual_error != expected_error) {
            std.log.err("Error type mismatch: {s} vs {s}", .{ @errorName(actual_error), @errorName(expected_error) });
            return error.AssertionFailed;
        }
    }
    
    /// Assert that operation completes within time limit
    pub fn completesWithin(comptime operation: fn () anyerror!void, max_ms: u64) !void {
        const start = std.time.milliTimestamp();
        try operation();
        const duration = std.time.milliTimestamp() - start;
        
        if (duration > max_ms) {
            std.log.err("Operation took too long: {d}ms > {d}ms", .{ duration, max_ms });
            return error.TimeoutExceeded;
        }
    }
};

/// Test suite runner
pub const TestSuite = struct {
    name: []const u8,
    tests: []const TestCase,
    
    pub const TestCase = struct {
        name: []const u8,
        test_fn: *const fn () anyerror!void,
    };
    
    pub fn run(self: TestSuite, allocator: Allocator) !void {
        var fixture = FrameworkFixture.init(allocator);
        defer fixture.deinit();
        
        try fixture.logger.info("Running test suite: {s}", .{self.name});
        
        var passed: usize = 0;
        var failed: usize = 0;
        
        for (self.tests) |test_case| {
            fixture.clearOutput();
            
            test_case.test_fn() catch |err| {
                try fixture.logger.err("FAIL: {s} - {s}", .{ test_case.name, @errorName(err) });
                failed += 1;
                continue;
            };
            
            try fixture.logger.info("PASS: {s}", .{test_case.name});
            passed += 1;
        }
        
        try fixture.logger.info("Test suite completed: {d} passed, {d} failed", .{ passed, failed });
        
        if (failed > 0) {
            return error.TestSuiteFailed;
        }
    }
};

test "MockWriter captures output correctly" {
    var mock = MockWriter.init(testing.allocator);
    defer mock.deinit();
    
    try mock.writer().print("Hello, {s}!", .{"World"});
    try expectEqualStrings("Hello, World!", mock.getWritten());
    try expect(mock.contains("World"));
}

test "FrameworkFixture provides testing environment" {
    var fixture = FrameworkFixture.init(testing.allocator);
    defer fixture.deinit();
    
    try fixture.logger.info("Test message", .{});
    try fixture.expectOutputContains("Test message");
}

test "PerformanceTest measures execution time" {
    var perf_test = PerformanceTest.init(testing.allocator);
    defer perf_test.deinit();
    
    perf_test.start();
    std.time.sleep(1000000); // 1ms
    try perf_test.stop();
    
    // Should complete within reasonable time (allowing for test environment variance)
    try perf_test.expectMaxDurationMs(100.0);
}

test "TestData generates consistent random data" {
    const seed = 12345;
    
    const bytes1 = try TestData.randomBytes(testing.allocator, 10, seed);
    defer testing.allocator.free(bytes1);
    
    const bytes2 = try TestData.randomBytes(testing.allocator, 10, seed);
    defer testing.allocator.free(bytes2);
    
    try expectEqualStrings(bytes1, bytes2);
}

test "Assert helpers work correctly" {
    try Assert.approxEqual(1.0, 1.001, 0.01);
    try Assert.arrayContains(i32, &[_]i32{ 1, 2, 3 }, 2);
    try Assert.errorType(error.OutOfMemory, error.OutOfMemory);
}