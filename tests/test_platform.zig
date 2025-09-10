//! Comprehensive tests for the platform module

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("../src/platform.zig");

test "PlatformInfo detection" {
    const testing = std.testing;

    const info = platform.PlatformInfo.detect();

    // Test basic fields
    try testing.expectEqual(builtin.os.tag, info.os);
    try testing.expectEqual(builtin.cpu.arch, info.arch);

    // Test boolean fields
    try testing.expect(@TypeOf(info.supports_ansi_colors) == bool);
    try testing.expect(@TypeOf(info.supports_simd) == bool);

    // Test numeric fields
    try testing.expect(info.max_threads > 0);
    try testing.expect(info.cache_line_size > 0);

    // Test cache line size detection
    const expected_cache_line = switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => 64,
        .x86 => 32,
        else => 64,
    };
    try testing.expectEqual(expected_cache_line, info.cache_line_size);
}

test "PlatformInfo ANSI color detection" {
    const testing = std.testing;

    const info = platform.PlatformInfo.detect();

    // Test ANSI support based on OS
    const expected_ansi = switch (builtin.os.tag) {
        .windows => false,
        .linux, .macos, .freebsd, .openbsd, .netbsd => true,
        else => false,
    };
    try testing.expectEqual(expected_ansi, info.supports_ansi_colors);
}

test "Platform initialization functions" {
    const testing = std.testing;

    // Test that initialization functions don't crash
    try platform.initializePlatform();

    // Test platform-specific initialization functions
    switch (builtin.os.tag) {
        .windows => {
            // Test Windows initialization (should not crash)
            // Note: We can't easily test the actual Windows ANSI setup without Windows API
        },
        .linux => {
            // Linux initialization should work
        },
        .macos => {
            // macOS initialization should work
        },
        else => {
            // Other platforms should work too
        },
    }
}

test "FileOps functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test file operations with a temporary file
    const temp_path = "test_temp_file.txt";
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    // Test file creation
    {
        var file = try platform.FileOps.createFile(temp_path);
        defer file.close();

        const content = "Hello, World!";
        try file.writeAll(content);
    }

    // Test file existence check
    try testing.expect(platform.FileOps.fileExists(temp_path));

    // Test file opening and reading
    {
        var file = try platform.FileOps.openFile(temp_path);
        defer file.close();

        var buffer: [100]u8 = undefined;
        const bytes_read = try file.read(&buffer);
        try testing.expectEqual(@as(usize, 13), bytes_read); // "Hello, World!" is 13 bytes
        try testing.expectEqualStrings("Hello, World!", buffer[0..bytes_read]);
    }

    // Test file deletion
    try platform.FileOps.deleteFile(temp_path);
    try testing.expect(!platform.FileOps.fileExists(temp_path));
}

test "MemoryOps functionality" {
    const testing = std.testing;

    // Test page size detection
    const page_size = platform.MemoryOps.getPageSize();
    try testing.expect(page_size > 0);
    try testing.expect(page_size % 1024 == 0); // Should be power of 2, typically 4096

    // Test page alignment
    try testing.expectEqual(@as(usize, 0), platform.MemoryOps.alignToPageSize(0));
    try testing.expectEqual(page_size, platform.MemoryOps.alignToPageSize(1));
    try testing.expectEqual(page_size, platform.MemoryOps.alignToPageSize(page_size - 1));
    try testing.expectEqual(page_size * 2, platform.MemoryOps.alignToPageSize(page_size + 1));

    // Test virtual memory limit
    const vm_limit = platform.MemoryOps.getVirtualMemoryLimit();
    try testing.expect(vm_limit > 0);

    // VM limit should be reasonable for the architecture
    const expected_min = switch (builtin.cpu.arch) {
        .x86_64 => 1 << 40, // At least 1 TB
        .aarch64 => 1 << 40,
        .x86 => 1 << 30, // At least 1 GB
        else => 1 << 30,
    };
    try testing.expect(vm_limit >= expected_min);
}

test "ThreadOps functionality" {
    const testing = std.testing;

    // Test optimal thread count
    const optimal_threads = platform.ThreadOps.getOptimalThreadCount();
    try testing.expect(optimal_threads > 0);
    try testing.expect(optimal_threads <= platform.PlatformInfo.detect().max_threads);

    // Thread priority enum values
    try testing.expectEqual(@as(u2, 0), @intFromEnum(platform.ThreadPriority.low));
    try testing.expectEqual(@as(u2, 1), @intFromEnum(platform.ThreadPriority.normal));
    try testing.expectEqual(@as(u2, 2), @intFromEnum(platform.ThreadPriority.high));
    try testing.expectEqual(@as(u2, 3), @intFromEnum(platform.ThreadPriority.realtime));
}

test "PerfOps functionality" {
    const testing = std.testing;

    // Test CPU frequency detection
    const cpu_freq = platform.PerfOps.getCpuFrequency();
    try testing.expect(cpu_freq > 0);
    try testing.expect(cpu_freq >= 500_000_000); // At least 500 MHz

    // Test cache info
    const cache_info = platform.PerfOps.getCacheInfo();

    try testing.expect(cache_info.l1_cache_size > 0);
    try testing.expect(cache_info.l2_cache_size > 0);
    try testing.expect(cache_info.l3_cache_size > 0);
    try testing.expect(cache_info.cache_line_size > 0);

    // L1 should be smaller than L2, L2 smaller than L3
    try testing.expect(cache_info.l1_cache_size < cache_info.l2_cache_size);
    try testing.expect(cache_info.l2_cache_size < cache_info.l3_cache_size);

    // Cache line size should match platform detection
    const platform_info = platform.PlatformInfo.detect();
    try testing.expectEqual(platform_info.cache_line_size, cache_info.cache_line_size);
}

test "Colors functionality" {
    const testing = std.testing;

    // Test color constants are non-empty strings
    try testing.expect(platform.Colors.reset.len > 0);
    try testing.expect(platform.Colors.bold.len > 0);
    try testing.expect(platform.Colors.red.len > 0);
    try testing.expect(platform.Colors.green.len > 0);
    try testing.expect(platform.Colors.yellow.len > 0);
    try testing.expect(platform.Colors.blue.len > 0);
    try testing.expect(platform.Colors.magenta.len > 0);
    try testing.expect(platform.Colors.cyan.len > 0);
    try testing.expect(platform.Colors.white.len > 0);

    // Test that color codes start with escape sequence
    try testing.expect(std.mem.startsWith(u8, platform.Colors.red, "\x1b["));
    try testing.expect(std.mem.startsWith(u8, platform.Colors.green, "\x1b["));
    try testing.expect(std.mem.startsWith(u8, platform.Colors.blue, "\x1b["));

    // Test that reset ends color codes
    try testing.expect(std.mem.endsWith(u8, platform.Colors.red, "m"));
    try testing.expect(std.mem.endsWith(u8, platform.Colors.green, "m"));
    try testing.expect(std.mem.endsWith(u8, platform.Colors.blue, "m"));
}

test "getTempDir functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const temp_dir = try platform.getTempDir(allocator);
    defer allocator.free(temp_dir);

    // Temp directory should be non-empty
    try testing.expect(temp_dir.len > 0);

    // Should match expected paths for different platforms
    const expected_path = switch (builtin.os.tag) {
        .windows => "C:\\temp",
        .linux, .macos, .freebsd, .openbsd, .netbsd => "/tmp",
        else => "/tmp",
    };

    try testing.expectEqualStrings(expected_path, temp_dir);
}

test "sleep function" {
    const testing = std.testing;

    // Test that sleep doesn't crash and takes approximately the right amount of time
    const start = std.time.nanoTimestamp();
    platform.sleep(10); // Sleep for 10ms
    const end = std.time.nanoTimestamp();

    const elapsed_ms = @as(u64, @intCast((end - start) / std.time.ns_per_ms));

    // Should sleep for at least 5ms (allowing some tolerance for timing variations)
    try testing.expect(elapsed_ms >= 5);
    // Shouldn't sleep for more than 100ms (reasonable upper bound)
    try testing.expect(elapsed_ms < 100);
}

test "getSystemInfo functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const info_str = try platform.getSystemInfo(allocator);
    defer allocator.free(info_str);

    // System info should contain expected fields
    try testing.expect(info_str.len > 0);
    try testing.expect(std.mem.indexOf(u8, info_str, "System Information:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "OS:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "Architecture:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "ANSI Colors:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "SIMD Support:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "Max Threads:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "Cache Line Size:") != null);
    try testing.expect(std.mem.indexOf(u8, info_str, "CPU Frequency:") != null);

    // Should contain actual platform information
    const platform_info = platform.PlatformInfo.detect();
    const os_str = @tagName(platform_info.os);
    const arch_str = @tagName(platform_info.arch);

    try testing.expect(std.mem.indexOf(u8, info_str, os_str) != null);
    try testing.expect(std.mem.indexOf(u8, info_str, arch_str) != null);
}

test "PlatformError values" {
    const testing = std.testing;

    // Test that all error values exist
    try testing.expectEqual(@as(u8, 0), @intFromError(platform.PlatformError.PlatformNotSupported));
    try testing.expectEqual(@as(u8, 1), @intFromError(platform.PlatformError.FeatureNotAvailable));
    try testing.expectEqual(@as(u8, 2), @intFromError(platform.PlatformError.InsufficientPermissions));
    try testing.expectEqual(@as(u8, 3), @intFromError(platform.PlatformError.ResourceExhausted));
}

test "CacheInfo structure" {
    const testing = std.testing;

    const cache_info = platform.PerfOps.getCacheInfo();

    // Test that cache sizes are reasonable
    try testing.expect(cache_info.l1_cache_size >= 1024); // At least 1KB
    try testing.expect(cache_info.l2_cache_size >= cache_info.l1_cache_size);
    try testing.expect(cache_info.l3_cache_size >= cache_info.l2_cache_size);

    // Cache line size should be power of 2 and reasonable
    try testing.expect(cache_info.cache_line_size >= 16); // At least 16 bytes
    try testing.expect(cache_info.cache_line_size <= 256); // At most 256 bytes
    try testing.expect(std.math.isPowerOfTwo(cache_info.cache_line_size));
}

test "Thread priority enum" {
    const testing = std.testing;

    // Test enum values and ordering
    try testing.expect(@intFromEnum(platform.ThreadPriority.low) < @intFromEnum(platform.ThreadPriority.normal));
    try testing.expect(@intFromEnum(platform.ThreadPriority.normal) < @intFromEnum(platform.ThreadPriority.high));
    try testing.expect(@intFromEnum(platform.ThreadPriority.high) < @intFromEnum(platform.ThreadPriority.realtime));
}

test "Platform-specific behavior" {
    const testing = std.testing;

    // Test that platform detection works correctly for current platform
    const info = platform.PlatformInfo.detect();

    // OS should match builtin
    try testing.expectEqual(builtin.os.tag, info.os);
    try testing.expectEqual(builtin.cpu.arch, info.arch);

    // Test platform-specific thread counts
    const max_threads = switch (builtin.os.tag) {
        .windows, .linux, .macos => 8,
        else => 4,
    };
    try testing.expectEqual(max_threads, info.max_threads);
}

test "File operations error handling" {
    const testing = std.testing;

    // Test operations on non-existent files
    try testing.expect(!platform.FileOps.fileExists("non_existent_file_12345.txt"));

    // Test deleting non-existent file should fail gracefully
    const result = platform.FileOps.deleteFile("non_existent_file_12345.txt");
    try testing.expectError(std.fs.File.OpenError.FileNotFound, result);
}

test "Memory operations edge cases" {
    const testing = std.testing;

    // Test alignment with various sizes
    const page_size = platform.MemoryOps.getPageSize();

    try testing.expectEqual(@as(usize, 0), platform.MemoryOps.alignToPageSize(0));

    // Test with sizes that are already aligned
    try testing.expectEqual(page_size, platform.MemoryOps.alignToPageSize(page_size));
    try testing.expectEqual(page_size * 2, platform.MemoryOps.alignToPageSize(page_size * 2));

    // Test with very large sizes
    const large_size = page_size * 1000 + 1;
    const aligned_large = platform.MemoryOps.alignToPageSize(large_size);
    try testing.expect(aligned_large >= large_size);
    try testing.expect(aligned_large % page_size == 0);
}

test "Performance metrics validation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test that system info generation doesn't crash and produces reasonable output
    const info = try platform.getSystemInfo(allocator);
    defer allocator.free(info);

    // Info should contain multiple lines
    var line_count: usize = 0;
    var iter = std.mem.splitScalar(u8, info, '\n');
    while (iter.next()) |_| {
        line_count += 1;
    }
    try testing.expect(line_count >= 10); // Should have at least 10 lines of info

    // Should contain numeric values for sizes/frequencies
    try testing.expect(std.mem.indexOf(u8, info, "KB") != null or std.mem.indexOf(u8, info, "MB") != null);
    try testing.expect(std.mem.indexOf(u8, info, "GHz") != null);
}
