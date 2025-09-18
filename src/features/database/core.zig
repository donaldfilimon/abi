//! WDBX Core Module
//!
//! This module contains core types, errors, and utilities used across the WDBX system.

const std = @import("std");

// =============================================================================
// CORE TYPES
// =============================================================================

/// WDBX standardized error codes with numeric IDs for consistent error handling
pub const WdbxError = error{
    // Database errors (1000-1999)
    AlreadyInitialized, // 1001
    NotInitialized, // 1002
    InvalidState, // 1003
    CorruptedDatabase, // 1004
    DimensionMismatch, // 1005
    VectorNotFound, // 1006
    IndexOutOfBounds, // 1007
    DatabaseLocked, // 1008
    IndexCorrupted, // 1009
    SchemaVersionMismatch, // 1010

    // Compression errors (2000-2999)
    CompressionFailed, // 2001
    DecompressionFailed, // 2002
    InvalidCompressedData, // 2003
    CompressionNotSupported, // 2004
    CompressionRatioTooLow, // 2005

    // I/O errors (3000-3999)
    OutOfMemory, // 3001
    FileBusy, // 3002
    DiskFull, // 3003
    BackupFailed, // 3004
    RestoreFailed, // 3005
    FilePermissionDenied, // 3006
    DirectoryNotFound, // 3007
    InvalidFilePath, // 3008

    // Configuration errors (4000-4999)
    InvalidConfiguration, // 4001
    ConfigurationValidationFailed, // 4002
    UnsupportedVersion, // 4003
    ConfigurationNotFound, // 4004
    InvalidConfigurationFormat, // 4005
    EnvironmentVariableError, // 4006

    // Network errors (5000-5999)
    ConnectionFailed, // 5001
    Timeout, // 5002
    InvalidRequest, // 5003
    RequestTooLarge, // 5004
    NetworkNotAvailable, // 5005
    PortInUse, // 5006
    BindFailed, // 5007
    SocketError, // 5008

    // Authentication errors (6000-6999)
    AuthenticationFailed, // 6001
    RateLimitExceeded, // 6002
    TokenExpired, // 6003
    InvalidCredentials, // 6004
    PermissionDenied, // 6005
    InvalidApiKey, // 6006

    // CLI errors (7000-7999)
    InvalidCommand, // 7001
    MissingArgument, // 7002
    InvalidParameter, // 7003
    CommandNotFound, // 7004
    InvalidOptionValue, // 7005
    ConflictingOptions, // 7006

    // Performance errors (8000-8999)
    PerformanceThresholdExceeded, // 8001
    ResourceExhausted, // 8002
    ThreadPoolFull, // 8003
    CacheOverflow, // 8004
    MemoryFragmentation, // 8005
    CpuThrottling, // 8006

    // Plugin errors (9000-9999)
    PluginNotFound, // 9001
    PluginLoadFailed, // 9002
    PluginVersionMismatch, // 9003
    PluginInitializationFailed, // 9004
    PluginExecutionFailed, // 9005
    InvalidPluginInterface, // 9006
};

/// Error code mapping for consistent error handling across modules
pub const ErrorCodes = struct {
    pub fn getErrorCode(err: WdbxError) u32 {
        return switch (err) {
            // Database errors (1000-1999)
            .AlreadyInitialized => 1001,
            .NotInitialized => 1002,
            .InvalidState => 1003,
            .CorruptedDatabase => 1004,
            .DimensionMismatch => 1005,
            .VectorNotFound => 1006,
            .IndexOutOfBounds => 1007,
            .DatabaseLocked => 1008,
            .IndexCorrupted => 1009,
            .SchemaVersionMismatch => 1010,

            // Compression errors (2000-2999)
            .CompressionFailed => 2001,
            .DecompressionFailed => 2002,
            .InvalidCompressedData => 2003,
            .CompressionNotSupported => 2004,
            .CompressionRatioTooLow => 2005,

            // I/O errors (3000-3999)
            .OutOfMemory => 3001,
            .FileBusy => 3002,
            .DiskFull => 3003,
            .BackupFailed => 3004,
            .RestoreFailed => 3005,
            .FilePermissionDenied => 3006,
            .DirectoryNotFound => 3007,
            .InvalidFilePath => 3008,

            // Configuration errors (4000-4999)
            .InvalidConfiguration => 4001,
            .ConfigurationValidationFailed => 4002,
            .UnsupportedVersion => 4003,
            .ConfigurationNotFound => 4004,
            .InvalidConfigurationFormat => 4005,
            .EnvironmentVariableError => 4006,

            // Network errors (5000-5999)
            .ConnectionFailed => 5001,
            .Timeout => 5002,
            .InvalidRequest => 5003,
            .RequestTooLarge => 5004,
            .NetworkNotAvailable => 5005,
            .PortInUse => 5006,
            .BindFailed => 5007,
            .SocketError => 5008,

            // Authentication errors (6000-6999)
            .AuthenticationFailed => 6001,
            .RateLimitExceeded => 6002,
            .TokenExpired => 6003,
            .InvalidCredentials => 6004,
            .PermissionDenied => 6005,
            .InvalidApiKey => 6006,

            // CLI errors (7000-7999)
            .InvalidCommand => 7001,
            .MissingArgument => 7002,
            .InvalidParameter => 7003,
            .CommandNotFound => 7004,
            .InvalidOptionValue => 7005,
            .ConflictingOptions => 7006,

            // Performance errors (8000-8999)
            .PerformanceThresholdExceeded => 8001,
            .ResourceExhausted => 8002,
            .ThreadPoolFull => 8003,
            .CacheOverflow => 8004,
            .MemoryFragmentation => 8005,
            .CpuThrottling => 8006,

            // Plugin errors (9000-9999)
            .PluginNotFound => 9001,
            .PluginLoadFailed => 9002,
            .PluginVersionMismatch => 9003,
            .PluginInitializationFailed => 9004,
            .PluginExecutionFailed => 9005,
            .InvalidPluginInterface => 9006,
        };
    }

    pub fn getErrorDescription(err: WdbxError) []const u8 {
        return switch (err) {
            // Database errors
            .AlreadyInitialized => "Database has already been initialized",
            .NotInitialized => "Database has not been initialized",
            .InvalidState => "Database is in an invalid state",
            .CorruptedDatabase => "Database file is corrupted",
            .DimensionMismatch => "Vector dimensions do not match",
            .VectorNotFound => "Vector not found in database",
            .IndexOutOfBounds => "Index is out of bounds",
            .DatabaseLocked => "Database is locked by another process",
            .IndexCorrupted => "Index data is corrupted",
            .SchemaVersionMismatch => "Database schema version mismatch",

            // Compression errors
            .CompressionFailed => "Data compression failed",
            .DecompressionFailed => "Data decompression failed",
            .InvalidCompressedData => "Invalid compressed data format",
            .CompressionNotSupported => "Compression algorithm not supported",
            .CompressionRatioTooLow => "Compression ratio below threshold",

            // I/O errors
            .OutOfMemory => "Insufficient memory available",
            .FileBusy => "File is busy or locked",
            .DiskFull => "Disk is full",
            .BackupFailed => "Backup operation failed",
            .RestoreFailed => "Restore operation failed",
            .FilePermissionDenied => "File permission denied",
            .DirectoryNotFound => "Directory not found",
            .InvalidFilePath => "Invalid file path",

            // Configuration errors
            .InvalidConfiguration => "Invalid configuration",
            .ConfigurationValidationFailed => "Configuration validation failed",
            .UnsupportedVersion => "Unsupported version",
            .ConfigurationNotFound => "Configuration file not found",
            .InvalidConfigurationFormat => "Invalid configuration file format",
            .EnvironmentVariableError => "Environment variable error",

            // Network errors
            .ConnectionFailed => "Network connection failed",
            .Timeout => "Operation timed out",
            .InvalidRequest => "Invalid request",
            .RequestTooLarge => "Request too large",
            .NetworkNotAvailable => "Network not available",
            .PortInUse => "Port already in use",
            .BindFailed => "Failed to bind to address",
            .SocketError => "Socket operation failed",

            // Authentication errors
            .AuthenticationFailed => "Authentication failed",
            .RateLimitExceeded => "Rate limit exceeded",
            .TokenExpired => "Authentication token expired",
            .InvalidCredentials => "Invalid credentials",
            .PermissionDenied => "Permission denied",
            .InvalidApiKey => "Invalid API key",

            // CLI errors
            .InvalidCommand => "Invalid command",
            .MissingArgument => "Missing required argument",
            .InvalidParameter => "Invalid parameter",
            .CommandNotFound => "Command not found",
            .InvalidOptionValue => "Invalid option value",
            .ConflictingOptions => "Conflicting command options",

            // Performance errors
            .PerformanceThresholdExceeded => "Performance threshold exceeded",
            .ResourceExhausted => "System resources exhausted",
            .ThreadPoolFull => "Thread pool is full",
            .CacheOverflow => "Cache overflow",
            .MemoryFragmentation => "Memory fragmentation detected",
            .CpuThrottling => "CPU throttling active",

            // Plugin errors
            .PluginNotFound => "Plugin not found",
            .PluginLoadFailed => "Failed to load plugin",
            .PluginVersionMismatch => "Plugin version mismatch",
            .PluginInitializationFailed => "Plugin initialization failed",
            .PluginExecutionFailed => "Plugin execution failed",
            .InvalidPluginInterface => "Invalid plugin interface",
        };
    }

    pub fn getErrorCategory(err: WdbxError) []const u8 {
        const code = getErrorCode(err);
        return switch (code / 1000) {
            1 => "Database",
            2 => "Compression",
            3 => "I/O",
            4 => "Configuration",
            5 => "Network",
            6 => "Authentication",
            7 => "CLI",
            8 => "Performance",
            9 => "Plugin",
            else => "Unknown",
        };
    }

    /// Format error for logging and display
    pub fn formatError(allocator: std.mem.Allocator, err: WdbxError, context: ?[]const u8) ![]const u8 {
        const code = getErrorCode(err);
        const category = getErrorCategory(err);
        const description = getErrorDescription(err);

        if (context) |ctx| {
            return try std.fmt.allocPrint(allocator, "[{s}:{d}] {s}: {s} (Context: {s})", .{ category, code, @errorName(err), description, ctx });
        } else {
            return try std.fmt.allocPrint(allocator, "[{s}:{d}] {s}: {s}", .{ category, code, @errorName(err), description });
        }
    }
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
    std.Thread.sleep(1000000); // 1ms
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
