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
const abi = @import("abi");

// =============================================================================
// CORE MODULE TESTS
// =============================================================================

test "core system initialization" {
    const allocator = testing.allocator;
    
    // Test initialization
    try abi.init(allocator);
    defer abi.deinit();
    
    // Verify system is initialized
    try testing.expect(abi.isInitialized());
    
    // Test core utilities
    try testing.expect(abi.core.platform.isWindows() or abi.core.platform.isLinux() or abi.core.platform.isMacOS());
    try testing.expect(abi.core.platform.optimalSimdWidth() > 0);
}

test "unified error handling" {
    // Test error creation
    const success = abi.core.ok(u32, 42);
    try testing.expect(success == .ok);
    try testing.expect(success.ok == 42);
    
    const failure = abi.core.err(u32, abi.core.AbiError.OutOfMemory);
    try testing.expect(failure == .err);
    try testing.expect(failure.err == abi.core.AbiError.OutOfMemory);
}

test "logging system" {
    const allocator = testing.allocator;
    
    // Initialize core system
    try abi.init(allocator);
    defer abi.deinit();
    
    // Test logging at different levels
    abi.core.log.debug("Debug message: {}", .{42});
    abi.core.log.info("Info message: {}", .{42});
    abi.core.log.warn("Warning message: {}", .{42});
    abi.core.log.err("Error message: {}", .{42});
    
    // Test log level setting
    abi.core.log.setLevel(abi.core.LogLevel.debug);
    abi.core.log.debug("This should be visible", .{});
    
    abi.core.log.setLevel(abi.core.LogLevel.error);
    abi.core.log.debug("This should be hidden", .{});
}

test "performance monitoring" {
    const allocator = testing.allocator;
    
    // Test performance counter
    var counter = abi.core.performance.Counter{};
    
    // Record some measurements
    counter.record(1000); // 1μs
    counter.record(2000); // 2μs
    counter.record(3000); // 3μs
    
    try testing.expectEqual(@as(u64, 3), counter.count);
    try testing.expectEqual(@as(u64, 6000), counter.total_time);
    try testing.expectEqual(@as(u64, 1000), counter.min_time);
    try testing.expectEqual(@as(u64, 3000), counter.max_time);
    
    // Test average calculation
    const avg = counter.average();
    try testing.expectEqual(@as(f64, 2000.0), avg);
    
    // Test reset
    counter.reset();
    try testing.expectEqual(@as(u64, 0), counter.count);
}

test "timer functionality" {
    // Test timer
    const timer = abi.core.performance.Timer.start();
    
    // Simulate some work
    std.Thread.sleep(1_000_000); // 1ms
    
    const elapsed = timer.elapsed();
    try testing.expect(elapsed > 0);
    
    const elapsed_ms = timer.elapsedMs();
    try testing.expect(elapsed_ms > 0.0);
}

test "string utilities" {
    const allocator = testing.allocator;
    
    // Test string operations
    try testing.expect(abi.core.string.startsWith("hello world", "hello"));
    try testing.expect(abi.core.string.endsWith("hello world", "world"));
    try testing.expect(!abi.core.string.startsWith("hello world", "world"));
    
    // Test string trimming
    const trimmed = try abi.core.string.trim(allocator, "  hello  ");
    defer allocator.free(trimmed);
    try testing.expectEqualStrings("hello", trimmed);
    
    // Test string splitting
    const parts = try abi.core.string.split(allocator, "a,b,c", ",");
    defer {
        for (parts) |part| allocator.free(part);
        allocator.free(parts);
    }
    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expectEqualStrings("a", parts[0]);
    try testing.expectEqualStrings("b", parts[1]);
    try testing.expectEqualStrings("c", parts[2]);
}

test "validation utilities" {
    // Test range validation
    try testing.expect(abi.core.validation.inRange(5, 1, 10));
    try testing.expect(!abi.core.validation.inRange(15, 1, 10));
    
    // Test string validation
    try testing.expect(abi.core.validation.notEmpty("hello"));
    try testing.expect(!abi.core.validation.notEmpty(""));
    
    try testing.expect(abi.core.validation.hasLength("hello", 5));
    try testing.expect(!abi.core.validation.hasLength("hello", 3));
    
    try testing.expect(abi.core.validation.hasMinLength("hello", 3));
    try testing.expect(!abi.core.validation.hasMinLength("hi", 3));
    
    try testing.expect(abi.core.validation.hasMaxLength("hello", 10));
    try testing.expect(!abi.core.validation.hasMaxLength("hello", 3));
}

test "platform detection" {
    // Test platform detection
    const platform_name = abi.core.platform.name();
    try testing.expect(platform_name.len > 0);
    
    const arch_name = abi.core.platform.arch();
    try testing.expect(arch_name.len > 0);
    
    // Test SIMD detection
    const has_simd = abi.core.platform.hasSimd();
    const simd_width = abi.core.platform.optimalSimdWidth();
    try testing.expect(simd_width > 0);
}

// =============================================================================
// SIMD MODULE TESTS
// =============================================================================

test "SIMD vector operations" {
    const allocator = testing.allocator;
    
    // Test vectors
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    
    // Test distance calculation
    const dist = abi.simd.distance(&a, &b);
    try testing.expect(dist > 0.0);
    
    // Test cosine similarity
    const similarity = abi.simd.cosineSimilarity(&a, &b);
    try testing.expect(similarity > 0.0 and similarity <= 1.0);
    
    // Test vector addition
    const result = try allocator.alloc(f32, a.len);
    defer allocator.free(result);
    abi.simd.add(result, &a, &b);
    try testing.expectEqual(@as(f32, 3.0), result[0]);
    try testing.expectEqual(@as(f32, 5.0), result[1]);
    
    // Test vector scaling
    abi.simd.scale(result, &a, 2.0);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    try testing.expectEqual(@as(f32, 4.0), result[1]);
    
    // Test dot product
    const dot = abi.simd.dotProduct(&a, &b);
    try testing.expect(dot > 0.0);
}

test "SIMD matrix operations" {
    const allocator = testing.allocator;
    
    // Test matrix-vector multiplication
    const matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const vector = [_]f32{ 1.0, 2.0 };
    const result = try allocator.alloc(f32, 3);
    defer allocator.free(result);
    
    abi.simd.matrixVectorMultiply(result, &matrix, &vector, 3, 2);
    try testing.expectEqual(@as(f32, 5.0), result[0]); // 1*1 + 2*2
    try testing.expectEqual(@as(f32, 11.0), result[1]); // 3*1 + 4*2
    try testing.expectEqual(@as(f32, 17.0), result[2]); // 5*1 + 6*2
}

test "SIMD performance monitoring" {
    const monitor = abi.simd.getPerformanceMonitor();
    const start_time = std.time.nanoTimestamp();
    
    // Simulate some operations
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    _ = abi.simd.distance(&a, &b);
    
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
    // Test error types
    const error_types = [_]abi.DbError{
        abi.DbError.InvalidFileFormat,
        abi.DbError.CorruptedData,
        abi.DbError.InvalidDimensions,
        abi.DbError.IndexOutOfBounds,
        abi.DbError.InsufficientMemory,
        abi.DbError.FileSystemError,
        abi.DbError.LockContention,
        abi.DbError.InvalidOperation,
        abi.DbError.VersionMismatch,
        abi.DbError.ChecksumMismatch,
    };
    
    // Verify all error types are defined
    try testing.expect(error_types.len > 0);
}

test "database header validation" {
    // Test header creation
    const header = abi.WdbxHeader.createDefault();
    try testing.expect(header.validateMagic());
    try testing.expectEqual(@as(u16, 1), header.version);
    try testing.expectEqual(@as(u64, 0), header.row_count);
}

// =============================================================================
// MEMORY MANAGEMENT TESTS
// =============================================================================

test "memory allocation tracking" {
    const allocator = testing.allocator;
    
    // Test basic allocation
    const data = try allocator.alloc(u8, 1024);
    defer allocator.free(data);
    
    try testing.expectEqual(@as(usize, 1024), data.len);
    
    // Test array allocation
    const numbers = try allocator.alloc(i32, 100);
    defer allocator.free(numbers);
    
    for (numbers, 0..) |*num, i| {
        num.* = @intCast(i);
    }
    
    try testing.expectEqual(@as(i32, 0), numbers[0]);
    try testing.expectEqual(@as(i32, 99), numbers[99]);
}

test "string memory management" {
    const allocator = testing.allocator;
    
    // Test string conversion
    const str = try abi.core.string.toString(allocator, 42);
    defer allocator.free(str);
    try testing.expectEqualStrings("42", str);
    
    // Test string splitting with proper cleanup
    const parts = try abi.core.string.split(allocator, "a,b,c", ",");
    defer {
        for (parts) |part| allocator.free(part);
        allocator.free(parts);
    }
    try testing.expectEqual(@as(usize, 3), parts.len);
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

test "end-to-end framework usage" {
    const allocator = testing.allocator;
    
    // Initialize framework
    try abi.init(allocator);
    defer abi.deinit();
    
    // Test system info
    const info = abi.getSystemInfo();
    try testing.expectEqualStrings("1.0.0", info.version);
    try testing.expect(info.features.len > 0);
    
    // Test SIMD support detection
    try testing.expect(info.simd_support.f32x4 or info.simd_support.f32x8 or info.simd_support.f32x16);
    
    // Test logging
    abi.core.log.info("Integration test completed successfully", .{});
}

test "performance benchmarking" {
    const allocator = testing.allocator;
    
    // Test performance counter
    var counter = abi.core.performance.Counter{};
    
    // Simulate multiple operations
    for (0..100) |_| {
        const timer = abi.core.performance.Timer.start();
        
        // Simulate some work
        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
        _ = abi.simd.distance(&a, &b);
        
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
    // Test error creation and handling
    const result: abi.Result(u32) = abi.core.err(u32, abi.core.AbiError.OutOfMemory);
    
    switch (result) {
        .ok => |value| {
            // This should not be reached
            try testing.expect(false);
            _ = value;
        },
        .err => |err| {
            try testing.expectEqual(abi.core.AbiError.OutOfMemory, err);
        },
    }
}

test "validation error handling" {
    // Test validation functions
    try testing.expect(abi.core.validation.inRange(5, 1, 10));
    try testing.expect(!abi.core.validation.inRange(15, 1, 10));
    
    // Test string validation
    try testing.expect(abi.core.validation.notEmpty("hello"));
    try testing.expect(!abi.core.validation.notEmpty(""));
}

// =============================================================================
// PLATFORM COMPATIBILITY TESTS
// =============================================================================

test "cross-platform compatibility" {
    // Test platform detection works on all platforms
    const platform_name = abi.core.platform.name();
    try testing.expect(platform_name.len > 0);
    
    const arch_name = abi.core.platform.arch();
    try testing.expect(arch_name.len > 0);
    
    // Test SIMD detection
    const simd_width = abi.core.platform.optimalSimdWidth();
    try testing.expect(simd_width > 0);
    
    // Test platform-specific features
    if (abi.core.platform.isWindows()) {
        try testing.expect(!abi.core.platform.isLinux());
        try testing.expect(!abi.core.platform.isMacOS());
    } else if (abi.core.platform.isLinux()) {
        try testing.expect(!abi.core.platform.isWindows());
        try testing.expect(!abi.core.platform.isMacOS());
    } else if (abi.core.platform.isMacOS()) {
        try testing.expect(!abi.core.platform.isWindows());
        try testing.expect(!abi.core.platform.isLinux());
    }
}
