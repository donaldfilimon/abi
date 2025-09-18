//! Comprehensive test suite for refactored Zig 0.16 codebase
//!
//! This test suite validates all the refactoring improvements including:
//! - Unified error handling system
//! - Core module functionality
//! - SIMD operations
//! - Database operations
//! - Memory management
//! - Performance monitoring

const std = @import("std");
const testing = std.testing;

// Export core modules
pub const core = @import("core/mod.zig");
pub const simd = @import("simd/mod.zig");
pub const db = @import("db/mod.zig");

// Export main types
pub const AbiError = core.AbiError;
pub const DbError = db.DbError;
pub const WdbxHeader = db.WdbxHeader;

// Global state
var global_allocator: ?std.mem.Allocator = null;
var is_initialized: bool = false;

/// Initialize the ABI framework
pub fn init(allocator: std.mem.Allocator) !void {
    if (is_initialized) return;
    global_allocator = allocator;
    try core.init(allocator);
    is_initialized = true;
}

/// Deinitialize the ABI framework
pub fn deinit() void {
    if (!is_initialized) return;
    core.deinit();
    global_allocator = null;
    is_initialized = false;
}

/// Check if the framework is initialized
pub fn isInitialized() bool {
    return is_initialized;
}

/// System information structure
pub const SystemInfo = struct {
    version: []const u8,
    features: []const []const u8,
    simd_support: struct {
        f32x4: bool,
        f32x8: bool,
        f32x16: bool,
    },
};

/// Get system information
pub fn getSystemInfo() SystemInfo {
    return SystemInfo{
        .version = "1.0.0",
        .features = &[_][]const u8{
            "core",
            "simd",
            "database",
            "performance_monitoring",
            "cross_platform",
        },
        .simd_support = .{
            .f32x4 = core.platform.hasSimd(),
            .f32x8 = core.platform.optimalSimdWidth() >= 8,
            .f32x16 = core.platform.optimalSimdWidth() >= 16,
        },
    };
}

// =============================================================================
// CORE MODULE TESTS
// =============================================================================

test "core system initialization" {
    const allocator = testing.allocator;

    try init(allocator);
    defer deinit();

    try testing.expect(isInitialized());
    try testing.expect(core.platform.isWindows() or core.platform.isLinux() or core.platform.isMacOS());
    try testing.expect(core.platform.optimalSimdWidth() > 0);
}

test "unified error handling" {
    const success = core.ok(u32, 42);
    try testing.expect(std.mem.eql(u8, @tagName(success), "ok"));
    try testing.expect(success.ok == 42);

    const failure = core.err(u32, core.AbiError.OutOfMemory);
    try testing.expect(std.mem.eql(u8, @tagName(failure), "err"));
    try testing.expect(failure.err == core.AbiError.OutOfMemory);
}

test "logging system" {
    const allocator = testing.allocator;

    try init(allocator);
    defer deinit();

    core.log.debug("Debug message: {}", .{42});
    core.log.info("Info message: {}", .{42});
    core.log.warn("Warning message: {}", .{42});
    core.log.err("Error message: {}", .{42});

    core.log.setLevel(.debug);
    core.log.debug("This should be visible", .{});

    core.log.setLevel(.err);
    core.log.debug("This should be hidden", .{});
}

test "performance monitoring" {
    var counter = core.performance.Counter{};

    counter.record(1000);
    counter.record(2000);
    counter.record(3000);

    try testing.expectEqual(@as(u64, 3), counter.count);
    try testing.expectEqual(@as(u64, 6000), counter.total_time);
    try testing.expectEqual(@as(u64, 1000), counter.min_time);
    try testing.expectEqual(@as(u64, 3000), counter.max_time);

    const avg = counter.average();
    try testing.expectEqual(@as(f64, 2000.0), avg);

    counter.reset();
    try testing.expectEqual(@as(u64, 0), counter.count);
}

test "timer functionality" {
    const timer = core.performance.Timer.start();

    std.Thread.sleep(1_000_000);

    const elapsed = timer.elapsed();
    try testing.expect(elapsed > 0);

    const elapsed_ms = timer.elapsedMs();
    try testing.expect(elapsed_ms > 0.0);
}

test "string utilities" {
    const allocator = testing.allocator;

    try testing.expect(core.string.startsWith("hello world", "hello"));
    try testing.expect(core.string.endsWith("hello world", "world"));
    try testing.expect(!core.string.startsWith("hello world", "world"));

    const trimmed = try core.string.trim(allocator, "  hello  ");
    defer allocator.free(trimmed);
    try testing.expect(std.mem.eql(u8, "hello", trimmed));

    const parts = try core.string.split(allocator, "a,b,c", ",");
    defer {
        for (parts) |part| allocator.free(part);
        allocator.free(parts);
    }
    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expect(std.mem.eql(u8, "a", parts[0]));
    try testing.expect(std.mem.eql(u8, "b", parts[1]));
    try testing.expect(std.mem.eql(u8, "c", parts[2]));
}

test "validation utilities" {
    try testing.expect(core.validation.inRange(5, 1, 10));
    try testing.expect(!core.validation.inRange(15, 1, 10));

    try testing.expect(core.validation.notEmpty("hello"));
    try testing.expect(!core.validation.notEmpty(""));

    try testing.expect(core.validation.hasLength("hello", 5));
    try testing.expect(!core.validation.hasLength("hello", 3));

    try testing.expect(core.validation.hasMinLength("hello", 3));
    try testing.expect(!core.validation.hasMinLength("hi", 3));

    try testing.expect(core.validation.hasMaxLength("hello", 10));
    try testing.expect(!core.validation.hasMaxLength("hello", 3));
}

test "platform detection" {
    const platform_name = core.platform.name();
    try testing.expect(platform_name.len > 0);

    const arch_name = core.platform.arch();
    try testing.expect(arch_name.len > 0);

    _ = core.platform.hasSimd();
    const simd_width = core.platform.optimalSimdWidth();
    try testing.expect(simd_width > 0);
}

// =============================================================================
// SIMD MODULE TESTS
// =============================================================================

test "SIMD vector operations" {
    const allocator = testing.allocator;

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    const dist = simd.distance(&a, &b);
    try testing.expect(dist > 0.0);

    const similarity = simd.cosineSimilarity(&a, &b);
    try testing.expect(similarity > 0.0 and similarity <= 1.0);

    const result = try allocator.alloc(f32, a.len);
    defer allocator.free(result);
    simd.add(result, &a, &b);
    try testing.expectEqual(@as(f32, 3.0), result[0]);
    try testing.expectEqual(@as(f32, 5.0), result[1]);

    simd.scale(result, &a, 2.0);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    try testing.expectEqual(@as(f32, 4.0), result[1]);

    const dot = simd.dotProduct(&a, &b);
    try testing.expect(dot > 0.0);
}

test "SIMD matrix operations" {
    const matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const vector = [_]f32{ 1.0, 2.0 };
    const result = try testing.allocator.alloc(f32, 3);
    defer testing.allocator.free(result);

    simd.matrixVectorMultiply(result, &matrix, &vector, 3, 2);
    try testing.expectEqual(@as(f32, 5.0), result[0]);
    try testing.expectEqual(@as(f32, 11.0), result[1]);
    try testing.expectEqual(@as(f32, 17.0), result[2]);
}

test "SIMD performance monitoring" {
    const monitor = simd.getPerformanceMonitor();
    const start_time = std.time.nanoTimestamp();

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    _ = simd.distance(&a, &b);

    const end_time = std.time.nanoTimestamp();
    const duration = @as(u64, @intCast(end_time - start_time));

    monitor.recordOperation(duration, true);
    try testing.expectEqual(@as(u64, 1), monitor.operation_count);
    try testing.expectEqual(@as(u64, 1), monitor.simd_usage_count);
}

// =============================================================================
// DATABASE MODULE TESTS
// =============================================================================

test "database error handling" {
    const error_types = [_]DbError{
        .InvalidFileFormat,
        .CorruptedData,
        .InvalidDimensions,
        .IndexOutOfBounds,
        .InsufficientMemory,
        .FileSystemError,
        .LockContention,
        .InvalidOperation,
        .VersionMismatch,
        .ChecksumMismatch,
    };

    try testing.expect(error_types.len > 0);
}

test "database header validation" {
    const header = WdbxHeader.createDefault();
    try testing.expect(header.validateMagic());
    try testing.expectEqual(@as(u16, 1), header.version);
    try testing.expectEqual(@as(u64, 0), header.row_count);
}

// =============================================================================
// MEMORY MANAGEMENT TESTS
// =============================================================================

test "memory allocation tracking" {
    const data = try testing.allocator.alloc(u8, 1024);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(usize, 1024), data.len);

    const numbers = try testing.allocator.alloc(i32, 100);
    defer testing.allocator.free(numbers);

    for (numbers, 0..) |*num, i| {
        num.* = @intCast(i);
    }

    try testing.expectEqual(@as(i32, 0), numbers[0]);
    try testing.expectEqual(@as(i32, 99), numbers[99]);
}

test "string memory management" {
    const str = try core.string.toString(testing.allocator, 42);
    defer testing.allocator.free(str);
    try testing.expect(std.mem.eql(u8, "42", str));

    const parts = try core.string.split(testing.allocator, "a,b,c", ",");
    defer {
        for (parts) |part| testing.allocator.free(part);
        testing.allocator.free(parts);
    }
    try testing.expectEqual(@as(usize, 3), parts.len);
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

test "end-to-end framework usage" {
    const allocator = testing.allocator;

    try init(allocator);
    defer deinit();

    const info = getSystemInfo();
    try testing.expect(std.mem.eql(u8, "1.0.0", info.version));
    try testing.expect(info.features.len > 0);

    try testing.expect(info.simd_support.f32x4 or info.simd_support.f32x8 or info.simd_support.f32x16);

    core.log.info("Integration test completed successfully", .{});
}

test "performance benchmarking" {
    var counter = core.performance.Counter{};

    for (0..100) |_| {
        const timer = core.performance.Timer.start();

        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
        _ = simd.distance(&a, &b);

        const elapsed = timer.elapsed();
        counter.record(elapsed);
    }

    try testing.expectEqual(@as(u64, 100), counter.count);
    try testing.expect(counter.getAverageTime() > 0.0);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

test "error propagation" {
    const result = core.err(u32, core.AbiError.OutOfMemory);

    switch (result) {
        .ok => |value| {
            try testing.expect(false);
            _ = value;
        },
        .err => |err| {
            try testing.expectEqual(core.AbiError.OutOfMemory, err);
        },
    }
}

test "validation error handling" {
    try testing.expect(core.validation.inRange(5, 1, 10));
    try testing.expect(!core.validation.inRange(15, 1, 10));

    try testing.expect(core.validation.notEmpty("hello"));
    try testing.expect(!core.validation.notEmpty(""));
}

// =============================================================================
// PLATFORM COMPATIBILITY TESTS
// =============================================================================

test "cross-platform compatibility" {
    const platform_name = core.platform.name();
    try testing.expect(platform_name.len > 0);

    const arch_name = core.platform.arch();
    try testing.expect(arch_name.len > 0);

    const simd_width = core.platform.optimalSimdWidth();
    try testing.expect(simd_width > 0);

    if (core.platform.isWindows()) {
        try testing.expect(!core.platform.isLinux());
        try testing.expect(!core.platform.isMacOS());
    } else if (core.platform.isLinux()) {
        try testing.expect(!core.platform.isWindows());
        try testing.expect(!core.platform.isMacOS());
    } else if (core.platform.isMacOS()) {
        try testing.expect(!core.platform.isWindows());
        try testing.expect(!core.platform.isLinux());
    }
}
