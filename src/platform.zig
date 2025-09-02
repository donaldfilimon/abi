//! Cross-platform utilities for the Abi AI framework
//!
//! @Definitions
//!
//! **PlatformInfo:**
//!   A structure describing the detected platform's capabilities and configuration. Fields include:
//!     - `os`: The operating system tag (e.g., .linux, .windows, .macos).
//!     - `arch`: The CPU architecture (e.g., .x86_64, .aarch64).
//!     - `supports_ansi_colors`: Whether the terminal supports ANSI color codes.
//!     - `supports_simd`: Whether SIMD instructions are available (from build-time detection).
//!     - `max_threads`: Maximum recommended thread count for the platform.
//!     - `cache_line_size`: The CPU's cache line size in bytes.
//!   Use `PlatformInfo.detect()` to obtain a populated instance at runtime.
//!
//! **FileOps:**
//!   Cross-platform file operations, including:
//!     - `openFile(path)`: Open a file for reading.
//!     - `createFile(path)`: Create a file for writing.
//!     - `deleteFile(path)`: Delete a file.
//!     - `fileExists(path)`: Check if a file exists.
//!   All operations use Zig's standard library and are safe for use on all supported platforms.
//!
//! **MemoryOps:**
//!   Cross-platform memory and virtual memory utilities, including:
//!     - `getPageSize()`: Returns the system's memory page size.
//!     - `alignToPageSize(size)`: Rounds up a size to the nearest page boundary.
//!     - `getVirtualMemoryLimit()`: Returns the maximum virtual memory addressable by the architecture.
//!
//! **ThreadOps:**
//!   Threading utilities for optimal concurrency, including:
//!     - `getOptimalThreadCount()`: Returns the recommended thread count for the platform.
//!     - `setThreadPriority(thread, priority)`: (Stub) Set thread priority (not implemented, platform-specific).
//!   The `ThreadPriority` enum provides standard priority levels: `low`, `normal`, `high`, `realtime`.
//!
//! **PerfOps:**
//!   Performance-related utilities, including:
//!     - `getCpuFrequency()`: Returns an estimated CPU base frequency in Hz.
//!     - `getCacheInfo()`: Returns a `CacheInfo` struct with L1/L2/L3 cache sizes and cache line size.
//!
//! **Colors:**
//!   ANSI color escape codes and colorized printing utilities. Fields include:
//!     - `reset`, `bold`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white` (ANSI codes).
//!     - `print(color, fmt, args)`: Print formatted text in color if supported by the terminal.
//!
//! **PlatformError:**
//!   Error set for platform-specific failures, including:
//!     - `PlatformNotSupported`, `FeatureNotAvailable`, `InsufficientPermissions`, `ResourceExhausted`.
//!
//! **Other Utilities:**
//!     - `getTempDir(allocator)`: Returns the path to the platform's temporary directory.
//!     - `sleep(milliseconds)`: Sleeps for the specified number of milliseconds.
//!     - `getSystemInfo(allocator)`: Returns a formatted string with system information.
//!
//! All functionality is implemented using Zig's standard library and builtins, with no C or libc dependencies.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("root.zig");
const core = @import("core/mod.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Platform capabilities and configuration
pub const PlatformInfo = struct {
    os: std.Target.Os.Tag,
    arch: std.Target.Cpu.Arch,
    supports_ansi_colors: bool,
    supports_simd: bool,
    max_threads: u32,
    cache_line_size: u32,

    pub fn detect() PlatformInfo {
        return .{
            .os = builtin.os.tag,
            .arch = builtin.cpu.arch,
            .supports_ansi_colors = detectAnsiSupport(),
            .supports_simd = abi.features.has_simd,
            .max_threads = detectMaxThreads(),
            .cache_line_size = detectCacheLineSize(),
        };
    }

    fn detectAnsiSupport() bool {
        return switch (builtin.os.tag) {
            .windows => false, // Simplified for now
            .linux, .macos, .freebsd, .openbsd, .netbsd => true,
            else => false,
        };
    }

    fn detectMaxThreads() u32 {
        return switch (builtin.os.tag) {
            .windows, .linux, .macos => 8, // Conservative default
            else => 4,
        };
    }

    fn detectCacheLineSize() u32 {
        return switch (builtin.cpu.arch) {
            .x86_64, .aarch64 => 64,
            .x86 => 32,
            else => 64, // Safe default
        };
    }
};

/// Platform-specific initialization
pub fn initializePlatform() !void {
    const info = PlatformInfo.detect();

    std.log.info("Platform initialized: {any} {any}", .{ info.os, info.arch });
    std.log.info("Features: ANSI={any}, SIMD={any}, Threads={any}", .{ info.supports_ansi_colors, info.supports_simd, info.max_threads });

    // Platform-specific setup
    switch (builtin.os.tag) {
        .windows => try initWindows(),
        .linux => try initLinux(),
        .macos => try initMacOS(),
        else => {}, // No special setup needed
    }
}

/// Windows-specific initialization
fn initWindows() !void {
    // Enable ANSI color support on Windows 10+
    if (builtin.os.tag == .windows) {
        enableWindowsAnsiColors() catch |err| {
            std.log.warn("Failed to enable ANSI colors: {}", .{err});
        };
    }
}

/// Linux-specific initialization
fn initLinux() !void {
    // Set up signal handlers or other Linux-specific features
    std.log.debug("Linux platform initialized", .{});
}

/// macOS-specific initialization
fn initMacOS() !void {
    // Set up macOS-specific features
    std.log.debug("macOS platform initialized", .{});
}

/// Enable ANSI color support on Windows (simplified)
fn enableWindowsAnsiColors() !void {
    if (builtin.os.tag != .windows) return;

    // Simplified implementation - skip Windows API calls to avoid compatibility issues
    std.log.debug("Windows ANSI color support skipped (simplified implementation)", .{});
}

/// Cross-platform file operations
pub const FileOps = struct {
    pub fn openFile(path: []const u8) !std.fs.File {
        return std.fs.cwd().openFile(path, .{});
    }

    pub fn createFile(path: []const u8) !std.fs.File {
        return std.fs.cwd().createFile(path, .{});
    }

    pub fn deleteFile(path: []const u8) !void {
        return std.fs.cwd().deleteFile(path);
    }

    pub fn fileExists(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        file.close();
        return true;
    }
};

/// Cross-platform memory operations
pub const MemoryOps = struct {
    pub fn getPageSize() usize {
        return switch (builtin.os.tag) {
            .windows => 4096,
            .linux, .macos, .freebsd, .openbsd, .netbsd => 4096,
            else => 4096,
        };
    }

    pub fn alignToPageSize(size: usize) usize {
        const page_size = getPageSize();
        return (size + page_size - 1) & ~(page_size - 1);
    }

    pub fn getVirtualMemoryLimit() usize {
        return switch (builtin.cpu.arch) {
            .x86_64 => 1 << 47, // 128 TB
            .x86 => 1 << 32, // 4 GB
            .aarch64 => 1 << 48, // 256 TB
            else => 1 << 32, // Conservative default
        };
    }
};

/// Cross-platform threading utilities
pub const ThreadOps = struct {
    pub fn getOptimalThreadCount() u32 {
        const info = PlatformInfo.detect();
        return @min(info.max_threads, std.Thread.getCpuCount() catch 4);
    }

    pub fn setThreadPriority(thread: std.Thread, priority: ThreadPriority) !void {
        _ = thread;
        _ = priority;
        // Simplified - actual implementation would be platform-specific
        std.log.debug("Thread priority setting not implemented", .{});
    }
};

pub const ThreadPriority = enum {
    low,
    normal,
    high,
    realtime,
};

/// Cross-platform performance utilities
pub const PerfOps = struct {
    pub fn getCpuFrequency() u64 {
        // Simplified - return base frequency estimate
        return switch (builtin.cpu.arch) {
            .x86_64 => 2_500_000_000, // 2.5 GHz
            .aarch64 => 2_000_000_000, // 2.0 GHz
            else => 1_000_000_000, // 1.0 GHz
        };
    }

    pub fn getCacheInfo() CacheInfo {
        return .{
            .l1_cache_size = 32 * 1024, // 32 KB
            .l2_cache_size = 256 * 1024, // 256 KB
            .l3_cache_size = 8 * 1024 * 1024, // 8 MB
            .cache_line_size = PlatformInfo.detect().cache_line_size,
        };
    }
};

pub const CacheInfo = struct {
    l1_cache_size: u32,
    l2_cache_size: u32,
    l3_cache_size: u32,
    cache_line_size: u32,
};

/// ANSI color support
pub const Colors = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";

    pub fn print(comptime color: []const u8, comptime fmt: []const u8, args: anytype) void {
        const info = PlatformInfo.detect();
        if (info.supports_ansi_colors) {
            std.debug.print("{s}", .{color});
            std.debug.print(fmt, args);
            std.debug.print("{s}", .{reset});
        } else {
            std.debug.print(fmt, args);
        }
    }
};

/// Platform-specific error handling
pub const PlatformError = error{
    PlatformNotSupported,
    FeatureNotAvailable,
    InsufficientPermissions,
    ResourceExhausted,
};

/// Get platform-specific temporary directory
pub fn getTempDir(allocator: std.mem.Allocator) ![]const u8 {
    return switch (builtin.os.tag) {
        .windows => try allocator.dupe(u8, "C:\\temp"),
        .linux, .macos, .freebsd, .openbsd, .netbsd => try allocator.dupe(u8, "/tmp"),
        else => try allocator.dupe(u8, "/tmp"),
    };
}

/// Platform-specific sleep function
pub fn sleep(milliseconds: u64) void {
    std.time.sleep(milliseconds * 1_000_000); // Convert to nanoseconds
}

/// Get system information as a formatted string
pub fn getSystemInfo(allocator: std.mem.Allocator) ![]const u8 {
    const info = PlatformInfo.detect();
    const cache_info = PerfOps.getCacheInfo();

    return try std.fmt.allocPrint(allocator,
        \\System Information:
        \\  OS: {}
        \\  Architecture: {}
        \\  ANSI Colors: {}
        \\  SIMD Support: {}
        \\  Max Threads: {}
        \\  Cache Line Size: {} bytes
        \\  L1 Cache: {} KB
        \\  L2 Cache: {} KB
        \\  L3 Cache: {} MB
        \\  CPU Frequency: {d:.1} GHz
        \\
    , .{
        info.os,
        info.arch,
        info.supports_ansi_colors,
        info.supports_simd,
        info.max_threads,
        info.cache_line_size,
        cache_info.l1_cache_size / 1024,
        cache_info.l2_cache_size / 1024,
        cache_info.l3_cache_size / (1024 * 1024),
        @as(f64, @floatFromInt(PerfOps.getCpuFrequency())) / 1_000_000_000.0,
    });
}
