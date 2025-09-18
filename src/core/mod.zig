//! Core utilities and cross-platform abstractions for Abi AI Framework
//!
//! This module provides fundamental types, utilities, and abstractions that
//! are used throughout the framework for consistent cross-platform behavior.
//!
//! ## Features
//! - Unified error handling system
//! - Cross-platform abstractions
//! - Memory management utilities
//! - Performance monitoring
//! - Logging and debugging tools
//!
//! ## Usage
//! ```zig
//! const core = @import("core");
//! const allocator = std.heap.page_allocator;
//!
//! // Initialize core systems
//! try core.init(allocator);
//! defer core.deinit();
//!
//! // Use unified error handling
//! const result = core.Result(T).ok(value);
//! ```

const std = @import("std");
const builtin = @import("builtin");

// =============================================================================
// UNIFIED ERROR SYSTEM
// =============================================================================

/// Unified error types for the entire framework
pub const AbiError = error{
    // Core system errors
    SystemNotInitialized,
    SystemAlreadyInitialized,
    InvalidConfiguration,
    ResourceExhausted,

    // Memory errors
    OutOfMemory,
    InvalidAllocation,
    MemoryLeak,
    BufferOverflow,
    BufferUnderflow,

    // I/O errors
    FileNotFound,
    PermissionDenied,
    DiskFull,
    NetworkError,
    Timeout,

    // Validation errors
    InvalidInput,
    InvalidState,
    InvalidOperation,
    DimensionMismatch,

    // Performance errors
    PerformanceThresholdExceeded,
    ResourceLimitExceeded,
    ConcurrencyError,

    // AI/ML specific errors
    ModelNotLoaded,
    InvalidModel,
    TrainingFailed,
    InferenceFailed,

    // Database errors
    DatabaseError,
    IndexError,
    QueryError,
    TransactionError,
};

/// Result type for operations that can fail
pub fn Result(comptime T: type) type {
    return std.meta.Result(T, AbiError);
}

/// Success result helper
pub fn ok(comptime T: type, value: T) Result(T) {
    return .{ .ok = value };
}

/// Error result helper
pub fn err(comptime T: type, error_type: AbiError) Result(T) {
    return .{ .err = error_type };
}

// =============================================================================
// CORE TYPES AND UTILITIES
// =============================================================================

// Re-export commonly used std types for convenience
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;
pub const HashMap = std.HashMap;
pub const StringHashMap = std.StringHashMap;
pub const AutoHashMap = std.AutoHashMap;

// =============================================================================
// CORE SYSTEM INITIALIZATION
// =============================================================================

/// Core system state
var core_initialized: bool = false;
var global_allocator: ?Allocator = null;

/// Initialize the core system
/// Must be called before using any framework functionality
pub fn init(allocator: Allocator) AbiError!void {
    if (core_initialized) {
        return AbiError.SystemAlreadyInitialized;
    }

    global_allocator = allocator;
    core_initialized = true;

    // Initialize logging system
    try log.init(allocator);

    log.info("Core system initialized", .{});
}

/// Deinitialize the core system
/// Should be called when shutting down the framework
pub fn deinit() void {
    if (!core_initialized) return;

    log.info("Core system shutting down", .{});

    // Cleanup logging
    log.deinit();

    global_allocator = null;
    core_initialized = false;
}

/// Get the global allocator
/// Returns the allocator passed to init()
pub fn getAllocator() Allocator {
    if (global_allocator) |alloc| {
        return alloc;
    }
    @panic("Core system not initialized. Call core.init() first.");
}

/// Check if core system is initialized
pub fn isInitialized() bool {
    return core_initialized;
}

// =============================================================================
// LOGGING SYSTEM
// =============================================================================

/// Log levels
pub const LogLevel = enum(u8) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    fatal = 4,

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }
};

/// Logging system
pub const log = struct {
    var allocator: ?Allocator = null;
    var current_level: LogLevel = .info;
    var enabled: bool = false;

    /// Initialize logging system
    pub fn init(alloc: Allocator) AbiError!void {
        allocator = alloc;
        enabled = true;
    }

    /// Deinitialize logging system
    pub fn deinit() void {
        enabled = false;
        allocator = null;
    }

    /// Set log level
    pub fn setLevel(level: LogLevel) void {
        current_level = level;
    }

    /// Log a debug message
    pub fn debug(comptime format: []const u8, args: anytype) void {
        logMessage(.debug, format, args);
    }

    /// Log an info message
    pub fn info(comptime format: []const u8, args: anytype) void {
        logMessage(.info, format, args);
    }

    /// Log a warning message
    pub fn warn(comptime format: []const u8, args: anytype) void {
        logMessage(.warn, format, args);
    }

    /// Log an error message
    pub fn err(comptime format: []const u8, args: anytype) void {
        logMessage(.err, format, args);
    }

    /// Log a fatal message
    pub fn fatal(comptime format: []const u8, args: anytype) void {
        logMessage(.fatal, format, args);
    }

    /// Internal logging function
    fn logMessage(level: LogLevel, comptime format: []const u8, args: anytype) void {
        if (!enabled or @intFromEnum(level) < @intFromEnum(current_level)) return;

        const timestamp = std.time.timestamp();
        std.debug.print("[{}] {s}: " ++ format ++ "\n", .{ timestamp, level.toString() } ++ args);
    }
};

// =============================================================================
// RANDOM NUMBER GENERATION
// =============================================================================

/// Random number generation utilities
pub const random = struct {
    pub const Random = std.Random;
    pub const DefaultPrng = std.Random.DefaultPrng;

    /// Get a thread-local random number generator
    pub fn getPrng() DefaultPrng {
        // Use a proper seeded PRNG instead of crypto.random for better performance
        var seed: u64 = undefined;
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        return DefaultPrng.init(seed);
    }

    /// Generate a random integer in range [min, max)
    pub fn intRangeLessThan(comptime T: type, min: T, max: T) T {
        var prng = getPrng();
        return prng.random().intRangeLessThan(T, min, max);
    }

    /// Generate a random float in range [0, 1)
    pub fn float(comptime T: type) T {
        var prng = getPrng();
        return prng.random().float(T);
    }

    /// Standard normal (mean 0, stddev 1) via Box-Muller
    pub fn normal(comptime T: type) T {
        var prng = getPrng();
        const r = prng.random();
        // Avoid 0
        var u_a: T = r.float(T);
        if (u_a <= 0) u_a = @as(T, 1.0e-7);
        const u_b: T = r.float(T);
        const two_pi: T = @as(T, 6.28318530717958647692);
        const mag = @sqrt(-2 * @log(u_a));
        const z0 = mag * @cos(two_pi * u_b);
        return z0;
    }
};

// =============================================================================
// PERFORMANCE MONITORING
// =============================================================================

/// Performance monitoring utilities
pub const performance = struct {
    /// Performance counter
    pub const Counter = struct {
        count: u64 = 0,
        total_time: u64 = 0,
        min_time: u64 = std.math.maxInt(u64),
        max_time: u64 = 0,

        /// Record a timing measurement
        pub fn record(self: *Counter, time_ns: u64) void {
            self.count += 1;
            self.total_time += time_ns;
            self.min_time = @min(self.min_time, time_ns);
            self.max_time = @max(self.max_time, time_ns);
        }

        /// Get average time
        pub fn average(self: Counter) f64 {
            if (self.count == 0) return 0.0;
            return @as(f64, @floatFromInt(self.total_time)) / @as(f64, @floatFromInt(self.count));
        }

        /// Reset counter
        pub fn reset(self: *Counter) void {
            self.count = 0;
            self.total_time = 0;
            self.min_time = std.math.maxInt(u64);
            self.max_time = 0;
        }
    };

    /// Timer for measuring execution time
    pub const Timer = struct {
        start_time: u64,

        /// Start timing
        pub fn start() Timer {
            return .{ .start_time = std.time.nanoTimestamp() };
        }

        /// Get elapsed time in nanoseconds
        pub fn elapsed(self: Timer) u64 {
            return std.time.nanoTimestamp() - self.start_time;
        }

        /// Get elapsed time in milliseconds
        pub fn elapsedMs(self: Timer) f64 {
            return @as(f64, @floatFromInt(self.elapsed())) / 1_000_000.0;
        }
    };
};

// =============================================================================
// STRING UTILITIES
// =============================================================================

/// String manipulation utilities
pub const string = struct {
    /// Check if string starts with prefix
    pub fn startsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.startsWith(u8, haystack, needle);
    }

    /// Check if string ends with suffix
    pub fn endsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.endsWith(u8, haystack, needle);
    }

    /// Trim whitespace from string
    pub fn trim(allocator: Allocator, s: []const u8) ![]u8 {
        const trimmed = std.mem.trim(u8, s, " \t\n\r");
        return allocator.dupe(u8, trimmed);
    }

    /// Split string by delimiter
    pub fn split(allocator: Allocator, s: []const u8, delimiter: []const u8) ![][]u8 {
        var parts = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (parts.items) |part| allocator.free(part);
            parts.deinit();
        }

        var iter = std.mem.split(u8, s, delimiter);
        while (iter.next()) |part| {
            const trimmed = try allocator.dupe(u8, part);
            try parts.append(trimmed);
        }

        return parts.toOwnedSlice();
    }

    /// Convert a value to string using the provided allocator
    /// Note: Consider using stack-allocated buffers for better performance
    pub fn toString(allocator: Allocator, value: anytype) ![]u8 {
        return std.fmt.allocPrint(allocator, "{}", .{value});
    }
};

// =============================================================================
// TIME UTILITIES
// =============================================================================

/// Time-related utilities
pub const time = struct {
    /// Get current timestamp in nanoseconds
    pub fn now() u64 {
        return std.time.nanoTimestamp();
    }

    /// Get current timestamp in milliseconds
    pub fn nowMs() u64 {
        return std.time.milliTimestamp();
    }

    /// Sleep for specified duration
    pub fn sleep(duration_ns: u64) void {
        std.Thread.sleep(duration_ns);
    }

    /// Format duration as human-readable string
    /// Note: Multiple allocPrint calls could be optimized with a single buffer
    pub fn formatDuration(duration_ns: u64) []const u8 {
        const allocator = if (isInitialized()) getAllocator() else std.heap.page_allocator;

        if (duration_ns < 1_000) {
            return std.fmt.allocPrint(allocator, "{}ns", .{duration_ns}) catch "0ns";
        } else if (duration_ns < 1_000_000) {
            const us_value = @as(f64, @floatFromInt(duration_ns)) / 1_000.0;
            return std.fmt.allocPrint(allocator, "{:.2}μs", .{us_value}) catch "0μs";
        } else if (duration_ns < 1_000_000_000) {
            const ms_value = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
            return std.fmt.allocPrint(allocator, "{:.2}ms", .{ms_value}) catch "0ms";
        } else {
            const s_value = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
            return std.fmt.allocPrint(allocator, "{:.2}s", .{s_value}) catch "0s";
        }
    }
};

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/// Input validation utilities
pub const validation = struct {
    /// Validate that a value is within a range
    pub fn inRange(value: anytype, min: anytype, max: anytype) bool {
        return value >= min and value <= max;
    }

    /// Validate that a string is not empty
    pub fn notEmpty(s: []const u8) bool {
        return s.len > 0;
    }

    /// Validate that a slice has the expected length
    pub fn hasLength(s: []const u8, expected: usize) bool {
        return s.len == expected;
    }

    /// Validate that a slice has at least the minimum length
    pub fn hasMinLength(s: []const u8, min: usize) bool {
        return s.len >= min;
    }

    /// Validate that a slice has at most the maximum length
    pub fn hasMaxLength(s: []const u8, max: usize) bool {
        return s.len <= max;
    }
};

// =============================================================================
// PLATFORM DETECTION
// =============================================================================

/// Platform-specific utilities
pub const platform = struct {
    /// Check if running on Windows
    pub fn isWindows() bool {
        return builtin.os.tag == .windows;
    }

    /// Check if running on Linux
    pub fn isLinux() bool {
        return builtin.os.tag == .linux;
    }

    /// Check if running on macOS
    pub fn isMacOS() bool {
        return builtin.os.tag == .macos;
    }

    /// Check if running on WebAssembly
    pub fn isWasm() bool {
        return builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
    }

    /// Get platform name as string
    pub fn name() []const u8 {
        return @tagName(builtin.os.tag);
    }

    /// Get architecture name as string
    pub fn arch() []const u8 {
        return @tagName(builtin.cpu.arch);
    }

    /// Check if SIMD is available
    pub fn hasSimd() bool {
        return !isWasm() and builtin.cpu.arch == .x86_64;
    }

    /// Get optimal SIMD width for current platform
    pub fn optimalSimdWidth() u32 {
        if (isWasm()) return 1;
        if (builtin.cpu.arch == .x86_64) return 8; // AVX2
        if (builtin.cpu.arch == .aarch64) return 4; // NEON
        return 4; // Default fallback
    }

    /// Check if SIMD operations are available
    pub fn isSimdAvailable(comptime width: usize) bool {
        return switch (width) {
            4 => true, // Basic SIMD always available
            8 => builtin.cpu.arch == .x86_64 and std.simd.suggestVectorLength(f32) >= 8,
            16 => builtin.cpu.arch == .x86_64 and std.simd.suggestVectorLength(f32) >= 16,
            else => false,
        };
    }
};
