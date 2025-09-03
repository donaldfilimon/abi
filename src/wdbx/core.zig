//! WDBX Core Module
//!
//! This module contains core types, errors, and utilities used across the WDBX system.

const std = @import("std");

// =============================================================================
// CORE TYPES
// =============================================================================

/// WDBX error types
pub const WdbxError = error{
    // Database errors
    AlreadyInitialized,
    NotInitialized,
    InvalidState,
    CorruptedDatabase,
    DimensionMismatch,
    VectorNotFound,
    IndexOutOfBounds,

    // Compression errors
    CompressionFailed,
    DecompressionFailed,
    InvalidCompressedData,

    // I/O errors
    OutOfMemory,
    FileBusy,
    DiskFull,
    BackupFailed,
    RestoreFailed,

    // Configuration errors
    InvalidConfiguration,
    ConfigurationValidationFailed,
    UnsupportedVersion,

    // Network errors
    ConnectionFailed,
    Timeout,
    InvalidRequest,
    RequestTooLarge,

    // Authentication errors
    AuthenticationFailed,
    RateLimitExceeded,

    // CLI errors
    InvalidCommand,
    MissingArgument,
    InvalidParameter,
};

/// WDBX version information
pub const VERSION = struct {
    pub const MAJOR = 1;
    pub const MINOR = 0;
    pub const PATCH = 0;
    pub const PRE_RELEASE = "alpha";

    pub fn string() []const u8 {
        return "1.0.0-alpha";
    }

    pub fn isCompatible(major: u32, minor: u32) bool {
        return major == MAJOR and minor <= MINOR;
    }
};

/// Output format options
pub const OutputFormat = enum {
    text,
    json,
    csv,

    pub fn fromString(s: []const u8) ?OutputFormat {
        if (std.ascii.eqlIgnoreCase(s, "text")) return .text;
        if (std.ascii.eqlIgnoreCase(s, "json")) return .json;
        if (std.ascii.eqlIgnoreCase(s, "csv")) return .csv;
        return null;
    }

    pub fn toString(self: OutputFormat) []const u8 {
        return switch (self) {
            .text => "text",
            .json => "json",
            .csv => "csv",
        };
    }
};

/// Log level enumeration
pub const LogLevel = enum(u8) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    fatal = 4,

    pub fn fromString(s: []const u8) ?LogLevel {
        if (std.ascii.eqlIgnoreCase(s, "debug")) return .debug;
        if (std.ascii.eqlIgnoreCase(s, "info")) return .info;
        if (std.ascii.eqlIgnoreCase(s, "warn")) return .warn;
        if (std.ascii.eqlIgnoreCase(s, "error")) return .err;
        if (std.ascii.eqlIgnoreCase(s, "fatal")) return .fatal;
        return null;
    }

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }

    pub fn toInt(self: LogLevel) u8 {
        return @intFromEnum(self);
    }
};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Common WDBX configuration
pub const Config = struct {
    // Basic settings
    debug_mode: bool = false,
    log_level: LogLevel = .info,
    output_format: OutputFormat = .text,

    // Database settings
    db_path: ?[]const u8 = null,
    dimension: u16 = 0,
    max_vectors: usize = 1_000_000,

    // Performance settings
    enable_simd: bool = true,
    enable_compression: bool = true,
    thread_count: ?u16 = null,

    // Network settings
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: usize = 1000,
    request_timeout_ms: u32 = 5000,

    // Security settings
    enable_auth: bool = false,
    enable_cors: bool = true,
    enable_rate_limiting: bool = true,

    pub fn init() Config {
        return .{};
    }

    pub fn validate(self: *const Config) WdbxError!void {
        if (self.port == 0) return WdbxError.InvalidConfiguration;
        if (self.max_connections == 0) return WdbxError.InvalidConfiguration;
        if (self.request_timeout_ms == 0) return WdbxError.InvalidConfiguration;
        if (self.thread_count != null and self.thread_count.? == 0) {
            return WdbxError.InvalidConfiguration;
        }
    }
};

// =============================================================================
// UTILITIES
// =============================================================================

/// Performance timer for benchmarking
pub const Timer = struct {
    start_time: i64,

    pub fn init() Timer {
        return .{ .start_time = std.time.nanoTimestamp() };
    }

    pub fn elapsed(self: *const Timer) u64 {
        const end_time = std.time.nanoTimestamp();
        return @intCast(end_time - self.start_time);
    }

    pub fn elapsedMs(self: *const Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1_000_000.0;
    }

    pub fn elapsedUs(self: *const Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1_000.0;
    }

    pub fn restart(self: *Timer) void {
        self.start_time = std.time.nanoTimestamp();
    }
};

/// Simple logging utility
pub const Logger = struct {
    allocator: std.mem.Allocator,
    level: LogLevel,

    pub fn init(allocator: std.mem.Allocator, level: LogLevel) Logger {
        return .{
            .allocator = allocator,
            .level = level,
        };
    }

    pub fn deinit(self: *Logger) void {
        _ = self;
        // Nothing to clean up for now
    }

    pub fn log(self: *Logger, level: LogLevel, comptime fmt: []const u8, args: anytype) !void {
        if (level.toInt() < self.level.toInt()) return;

        const timestamp = std.time.milliTimestamp();
        const level_str = level.toString();

        std.debug.print("[{d}] [{s}] " ++ fmt ++ "\n", .{ timestamp, level_str } ++ args);
    }

    pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.debug, fmt, args);
    }

    pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.info, fmt, args);
    }

    pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.warn, fmt, args);
    }

    pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.err, fmt, args);
    }

    pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.fatal, fmt, args);
    }
};

/// Memory usage statistics
pub const MemoryStats = struct {
    total_allocated: usize = 0,
    total_freed: usize = 0,
    current_usage: usize = 0,
    peak_usage: usize = 0,
    allocation_count: usize = 0,
    free_count: usize = 0,

    pub fn init() MemoryStats {
        return .{};
    }

    pub fn allocate(self: *MemoryStats, size: usize) void {
        self.total_allocated += size;
        self.current_usage += size;
        self.allocation_count += 1;

        if (self.current_usage > self.peak_usage) {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn deallocate(self: *MemoryStats, size: usize) void {
        self.total_freed += size;
        self.current_usage -= size;
        self.free_count += 1;
    }

    pub fn reset(self: *MemoryStats) void {
        self.* = init();
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "VERSION compatibility" {
    try std.testing.expect(VERSION.isCompatible(1, 0));
    try std.testing.expect(!VERSION.isCompatible(2, 0));
    try std.testing.expect(!VERSION.isCompatible(1, 1));
}

test "OutputFormat parsing" {
    try std.testing.expectEqual(OutputFormat.json, OutputFormat.fromString("json").?);
    try std.testing.expectEqual(OutputFormat.text, OutputFormat.fromString("TEXT").?);
    try std.testing.expectEqual(@as(?OutputFormat, null), OutputFormat.fromString("invalid"));
}

test "LogLevel ordering" {
    try std.testing.expect(LogLevel.debug.toInt() < LogLevel.info.toInt());
    try std.testing.expect(LogLevel.info.toInt() < LogLevel.warn.toInt());
    try std.testing.expect(LogLevel.warn.toInt() < LogLevel.err.toInt());
}

test "Config validation" {
    var config = Config.init();
    try config.validate();

    config.port = 0;
    try std.testing.expectError(WdbxError.InvalidConfiguration, config.validate());
}

test "Timer functionality" {
    var timer = Timer.init();
    const end = std.time.milliTimestamp() + 1; while (std.time.milliTimestamp() < end) { std.atomic.spinLoopHint(); } // 1ms
    const elapsed = timer.elapsedUs();
    try std.testing.expect(elapsed >= 1000.0); // Should be at least 1000 microseconds
}

test "MemoryStats tracking" {
    var stats = MemoryStats.init();

    stats.allocate(100);
    try std.testing.expectEqual(@as(usize, 100), stats.current_usage);
    try std.testing.expectEqual(@as(usize, 100), stats.peak_usage);
    try std.testing.expectEqual(@as(usize, 1), stats.allocation_count);

    stats.allocate(200);
    try std.testing.expectEqual(@as(usize, 300), stats.current_usage);
    try std.testing.expectEqual(@as(usize, 300), stats.peak_usage);

    stats.deallocate(100);
    try std.testing.expectEqual(@as(usize, 200), stats.current_usage);
    try std.testing.expectEqual(@as(usize, 300), stats.peak_usage); // Peak should remain
    try std.testing.expectEqual(@as(usize, 1), stats.free_count);
}
