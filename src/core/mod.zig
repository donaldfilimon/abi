<<<<<<< HEAD
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
=======
//! Core Module - Fundamental Types and Operations
//!
//! This module provides the foundational types, errors, and operations used across
//! the entire WDBX framework, including SIMD operations and error definitions.

const std = @import("std");
<<<<<<<< HEAD:src/simd/mod.zig
const builtin = @import("builtin");
const core = @import("core");
========

/// Framework-wide error set for consistent error handling
pub const FrameworkError = error{
    // Generic framework errors
    InvalidConfiguration,
    UnsupportedOperation,
    InvalidState,
    InvalidData,
    NotImplemented,
    ResourceExhausted,
    OperationFailed,

    // Memory-related errors
    OutOfMemory,
    InvalidAlignment,
    BufferTooSmall,

    // Data processing errors
    InvalidDimensions,
    TypeMismatch,
    ConversionFailed,

    // I/O errors
    FileNotFound,
    AccessDenied,
    NetworkError,

    // Computation errors
    NumericalInstability,
    ConvergenceFailure,
    DivisionByZero,
} || std.mem.Allocator.Error;

// SIMD functionality is now integrated directly into core

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;

// =============================================================================
// SIMD OPERATIONS
// =============================================================================
>>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b:src/core/mod.zig

/// SIMD vector types with automatic detection
pub const Vector = struct {
    /// 4-float SIMD vector
    pub const f32x4 = if (@hasDecl(std.simd, "f32x4")) std.simd.f32x4 else @Vector(4, f32);
    /// 8-float SIMD vector
    pub const f32x8 = if (@hasDecl(std.simd, "f32x8")) std.simd.f32x8 else @Vector(8, f32);
    /// 16-float SIMD vector
    pub const f32x16 = if (@hasDecl(std.simd, "f32x16")) std.simd.f32x16 else @Vector(16, f32);

    /// Load vector from slice (compatible with both std.simd and @Vector)
    pub fn load(comptime T: type, data: []const f32) T {
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            return std.simd.f32x16.load(data);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            return std.simd.f32x8.load(data);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            return std.simd.f32x4.load(data);
        } else {
            // Fallback for @Vector types
            var result: T = undefined;
            for (0..@typeInfo(T).vector.len) |i| {
                result[i] = data[i];
            }
            return result;
        }
    }

    /// Store vector to slice (compatible with both std.simd and @Vector)
    pub fn store(data: []f32, vec: anytype) void {
        const T = @TypeOf(vec);
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            std.simd.f32x16.store(data, vec);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            std.simd.f32x8.store(data, vec);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            std.simd.f32x4.store(data, vec);
        } else {
            // Fallback for @Vector types
            for (0..@typeInfo(T).vector.len) |i| {
                data[i] = vec[i];
            }
        }
    }

    /// Create splat vector (compatible with both std.simd and @Vector)
    pub fn splat(comptime T: type, value: f32) T {
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            return std.simd.f32x16.splat(value);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            return std.simd.f32x8.splat(value);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            return std.simd.f32x4.splat(value);
        } else {
            // Fallback for @Vector types
            return @splat(value);
        }
    }

    /// Check if SIMD is available for a given vector size
    pub fn isSimdAvailable(comptime size: usize) bool {
        return switch (size) {
            4 => @hasDecl(std.simd, "f32x4"),
            8 => @hasDecl(std.simd, "f32x8"),
            16 => @hasDecl(std.simd, "f32x16"),
            else => false,
        };
    }

    /// Get optimal SIMD vector size for given dimension
    pub fn getOptimalSize(dimension: usize) usize {
        if (dimension >= 16 and isSimdAvailable(16)) return 16;
        if (dimension >= 8 and isSimdAvailable(8)) return 8;
        if (dimension >= 4 and isSimdAvailable(4)) return 4;
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
        return 1;
    }
};

<<<<<<< HEAD

=======
/// SIMD-optimized vector operations
pub const VectorOps = struct {
    /// Calculate Euclidean distance between two vectors using SIMD
    pub fn distance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        if (a.len == 0) return 0.0;

        var acc: f32 = 0.0;
        var i: usize = 0;

        // SIMD-optimized distance calculation with unified approach
        const optimal_size = Vector.getOptimalSize(a.len);
        inline for (.{ 16, 8, 4 }) |vec_size| {
            if (optimal_size == vec_size) {
                while (i + vec_size <= a.len) : (i += vec_size) {
                    const va = @as(@Vector(vec_size, f32), a[i..][0..vec_size].*);
                    const vb = @as(@Vector(vec_size, f32), b[i..][0..vec_size].*);
                    const diff = va - vb;
                    acc += @reduce(.Add, diff * diff);
                }
                break;
            }
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            acc += diff * diff;
        }

        return @sqrt(acc);
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;
        if (a.len == 0) return 0.0;

        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        // SIMD-optimized calculations with unified approach
        inline for (.{ 16, 8, 4 }) |vec_size| {
            if (optimal_size == vec_size) {
                while (i + vec_size <= a.len) : (i += vec_size) {
                    const va = @as(@Vector(vec_size, f32), a[i..][0..vec_size].*);
                    const vb = @as(@Vector(vec_size, f32), b[i..][0..vec_size].*);

                    dot_product += @reduce(.Add, va * vb);
                    norm_a += @reduce(.Add, va * va);
                    norm_b += @reduce(.Add, vb * vb);
                }
                break;
            }
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        const denominator = @sqrt(norm_a) * @sqrt(norm_b);
        if (denominator == 0.0) return 0.0;
        return dot_product / denominator;
    }

    /// Add two vectors using SIMD
    pub fn add(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

<<<<<<<< HEAD:src/simd/mod.zig
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const sum = va + vb;
                    @memcpy(result[i..][0..16], @as([16]f32, sum)[0..]); // Bounds checked by loop condition
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const sum = va + vb;
                    @memcpy(result[i..][0..8], @as([8]f32, sum)[0..]); // Bounds checked by loop condition
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const sum = va + vb;
                    @memcpy(result[i..][0..4], @as([4]f32, sum)[0..]); // Bounds checked by loop condition
                }
            },
            else => {},
========
        inline for (.{ 16, 8, 4 }) |vec_size| {
            if (optimal_size == vec_size) {
                while (i + vec_size <= a.len) : (i += vec_size) {
                    const va = @as(@Vector(vec_size, f32), a[i..][0..vec_size].*);
                    const vb = @as(@Vector(vec_size, f32), b[i..][0..vec_size].*);
                    const s = va + vb;
                    @memcpy(result[i..][0..vec_size], @as([vec_size]f32, s)[0..]);
                }
                break;
            }
>>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b:src/core/mod.zig
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Calculate dot product of two vectors
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;

        var acc: f32 = 0.0;
        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

<<<<<<<< HEAD:src/simd/mod.zig
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..16], @as([16]f32, diff)[0..]); // Bounds checked by loop condition
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..8], @as([8]f32, diff)[0..]); // Bounds checked by loop condition
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..4], @as([4]f32, diff)[0..]); // Bounds checked by loop condition
                }
            },
            else => {},
========
        inline for (.{ 16, 8, 4 }) |vec_size| {
            if (optimal_size == vec_size) {
                while (i + vec_size <= a.len) : (i += vec_size) {
                    const va = @as(@Vector(vec_size, f32), a[i..][0..vec_size].*);
                    const vb = @as(@Vector(vec_size, f32), b[i..][0..vec_size].*);
                    acc += @reduce(.Add, va * vb);
                }
                break;
            }
>>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b:src/core/mod.zig
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            acc += a[i] * b[i];
        }

        return acc;
    }

    /// Matrix multiplication (simplified for vector ops)
    pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, m: usize, n: usize, p: usize) void {
        // Simple matrix multiplication: result[m][p] = a[m][n] * b[n][p]
        if (result.len != m * p or a.len != m * n or b.len != n * p) return;

        for (0..m) |i| {
            for (0..p) |j| {
                var sum: f32 = 0.0;
                for (0..n) |k| {
                    sum += a[i * n + k] * b[k * p + j];
                }
                result[i * p + j] = sum;
            }
        }
    }

    /// Multiply vector by scalar using SIMD
    pub fn scale(result: []f32, vector: []const f32, scalar: f32) void {
        if (result.len != vector.len) return;

        const optimal_size = Vector.getOptimalSize(vector.len);
        var i: usize = 0;

        inline for (.{ 16, 8, 4 }) |vec_size| {
            if (optimal_size == vec_size) {
                const scale_vec = @as(@Vector(vec_size, f32), @splat(scalar));
                while (i + vec_size <= vector.len) : (i += vec_size) {
                    const v = @as(@Vector(vec_size, f32), vector[i..][0..vec_size].*);
                    const scaled = v * scale_vec;
<<<<<<<< HEAD:src/simd/mod.zig
                    @memcpy(result[i..][0..16], @as([16]f32, scaled)[0..]); // Bounds checked by loop condition
                }
            },
            8 => {
                const scale_vec = @as(@Vector(8, f32), @splat(scalar));
                while (i + 8 <= vector.len) : (i += 8) {
                    const v = @as(@Vector(8, f32), vector[i..][0..8].*);
                    const scaled = v * scale_vec;
                    @memcpy(result[i..][0..8], @as([8]f32, scaled)[0..]); // Bounds checked by loop condition
                }
            },
            4 => {
                const scale_vec = @as(@Vector(4, f32), @splat(scalar));
                while (i + 4 <= vector.len) : (i += 4) {
                    const v = @as(@Vector(4, f32), vector[i..][0..4].*);
                    const scaled = v * scale_vec;
                    @memcpy(result[i..][0..4], @as([4]f32, scaled)[0..]); // Bounds checked by loop condition
                }
            },
            else => {},
========
                    @memcpy(result[i..][0..vec_size], @as([vec_size]f32, scaled)[0..]);
                }
                break;
            }
>>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b:src/core/mod.zig
        }

        // Handle remaining elements
        while (i < vector.len) : (i += 1) {
            result[i] = vector[i] * scalar;
        }
    }

    /// Normalize vector to unit length
    pub fn normalize(result: []f32, vector: []const f32) void {
        if (result.len != vector.len) return;

        const norm = @sqrt(VectorOps.dotProduct(vector, vector));
        if (norm == 0.0) {
            @memset(result, 0.0);
            return;
        }

        VectorOps.scale(result, vector, 1.0 / norm);
    }
};

/// Matrix operations with SIMD acceleration
pub const MatrixOps = struct {
    /// Matrix-vector multiplication: result = matrix * vector
    pub fn matrixVectorMultiply(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) void {
        if (result.len != rows or vector.len != cols) return;

        const optimal_size = Vector.getOptimalSize(cols);
        var row: usize = 0;

        while (row < rows) : (row += 1) {
            const row_start = row * cols;
            var acc_row: f32 = 0.0;
            var col: usize = 0;

            // SIMD-optimized row-vector multiplication
            switch (optimal_size) {
                16 => {
                    while (col + 16 <= cols) : (col += 16) {
                        const matrix_row = @as(@Vector(16, f32), matrix[row_start + col ..][0..16].*);
                        const vec_slice = @as(@Vector(16, f32), vector[col..][0..16].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                8 => {
                    while (col + 8 <= cols) : (col += 8) {
                        const matrix_row = @as(@Vector(8, f32), matrix[row_start + col ..][0..8].*);
                        const vec_slice = @as(@Vector(8, f32), vector[col..][0..8].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                4 => {
                    while (col + 4 <= cols) : (col += 4) {
                        const matrix_row = @as(@Vector(4, f32), matrix[row_start + col ..][0..4].*);
                        const vec_slice = @as(@Vector(4, f32), vector[col..][0..4].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                else => {},
            }

            // Handle remaining columns
            while (col < cols) : (col += 1) {
                acc_row += matrix[row_start + col] * vector[col];
            }

            result[row] = acc_row;
        }
    }
};

/// Performance monitoring for SIMD operations
pub const PerformanceMonitor = struct {
    operation_count: u64 = 0,
    total_time_ns: u64 = 0,
    simd_usage_count: u64 = 0,
    scalar_fallback_count: u64 = 0,

    pub fn recordOperation(self: *PerformanceMonitor, duration_ns: u64, used_simd: bool) void {
        self.operation_count += 1;
        self.total_time_ns += duration_ns;
        if (used_simd) {
            self.simd_usage_count += 1;
        } else {
            self.scalar_fallback_count += 1;
        }
    }

    pub fn getAverageTime(self: *const PerformanceMonitor) f64 {
        if (self.operation_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_time_ns)) / @as(f64, @floatFromInt(self.operation_count));
    }

    pub fn getSimdUsageRate(self: *const PerformanceMonitor) f64 {
        if (self.operation_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.simd_usage_count)) / @as(f64, @floatFromInt(self.operation_count));
    }

    pub fn printStats(self: *const PerformanceMonitor) void {
        core.log.info("SIMD Performance Statistics:", .{});
        core.log.info("  Total Operations: {d}", .{self.operation_count});
        core.log.info("  Average Time: {d:.3} ns", .{self.getAverageTime()});
        core.log.info("  SIMD Usage Rate: {d:.1}%", .{self.getSimdUsageRate() * 100.0});
        core.log.info("  SIMD Operations: {d}", .{self.simd_usage_count});
        core.log.info("  Scalar Fallbacks: {d}", .{self.scalar_fallback_count});
    }
};

/// Global performance monitor instance
var global_performance_monitor = PerformanceMonitor{};

/// Get global performance monitor
pub fn getPerformanceMonitor() *PerformanceMonitor {
    return &global_performance_monitor;
}

/// Compile-time feature detection
pub const Features = struct {
    pub const has_simd = @hasDecl(std.simd, "f32x4");
    pub const has_avx = @import("builtin").target.cpu.arch == .x86_64 and
        std.Target.x86.featureSetHas(@import("builtin").target.cpu.features, .avx);
    pub const has_neon = @import("builtin").target.cpu.arch == .aarch64 and
        std.Target.aarch64.featureSetHas(@import("builtin").target.cpu.features, .neon);
};

/// Common validation utilities
pub const Validation = struct {
    /// Validate that dimensions match
    pub fn validateDimensions(expected: usize, actual: usize) FrameworkError!void {
        if (expected != actual) {
            return FrameworkError.InvalidDimensions;
        }
    }

    /// Validate that slice is not empty
    pub fn validateNonEmpty(slice: anytype) FrameworkError!void {
        if (slice.len == 0) {
            return FrameworkError.InvalidData;
        }
    }

    /// Validate alignment requirements
    pub fn validateAlignment(ptr: anytype, alignment: usize) FrameworkError!void {
        if (@intFromPtr(ptr) % alignment != 0) {
            return FrameworkError.InvalidAlignment;
        }
    }
};

test "core framework error handling" {
    const testing = std.testing;

    // Test dimension validation
    try Validation.validateDimensions(128, 128);
    try testing.expectError(FrameworkError.InvalidDimensions, Validation.validateDimensions(128, 256));

    // Test empty slice validation
    const empty_slice: []const f32 = &.{};
    const valid_slice = &[_]f32{ 1.0, 2.0, 3.0 };

    try testing.expectError(FrameworkError.InvalidData, Validation.validateNonEmpty(empty_slice));
    try Validation.validateNonEmpty(valid_slice);
}

// =============================================================================
// SIMD CONVENIENCE RE-EXPORTS
// =============================================================================

// Re-export commonly used types
pub const f32x4 = Vector.f32x4;
pub const f32x8 = Vector.f32x8;
pub const f32x16 = Vector.f32x16;

// Re-export operations
pub const distance = VectorOps.distance;
pub const cosineSimilarity = VectorOps.cosineSimilarity;
pub const add = VectorOps.add;
pub const dotProduct = VectorOps.dotProduct;
pub const scale = VectorOps.scale;
pub const matrixVectorMultiply = MatrixOps.matrixVectorMultiply;

test "SIMD vector operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test vectors
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    // Test distance calculation
    const dist = distance(&a, &b);
    try testing.expect(dist > 0.0);

    // Test cosine similarity
    const similarity = cosineSimilarity(&a, &b);
    try testing.expect(similarity > 0.0 and similarity <= 1.0);

    // Test vector addition
    const result = try allocator.alloc(f32, a.len);
    defer allocator.free(result);
    add(result, &a, &b);
    try testing.expectEqual(@as(f32, 3.0), result[0]);
    try testing.expectEqual(@as(f32, 5.0), result[1]);

    // Test vector scaling
    scale(result, &a, 2.0);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    try testing.expectEqual(@as(f32, 4.0), result[1]);

    // Test dot product
    const dot = dotProduct(&a, &b);
    try testing.expect(dot > 0.0);
}

test "SIMD matrix operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test matrix-vector multiplication
    const matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const vector = [_]f32{ 1.0, 2.0 };
    const result = try allocator.alloc(f32, 3);
    defer allocator.free(result);

    matrixVectorMultiply(result, &matrix, &vector, 3, 2);
    try testing.expectEqual(@as(f32, 5.0), result[0]); // 1*1 + 2*2
    try testing.expectEqual(@as(f32, 11.0), result[1]); // 3*1 + 4*2
    try testing.expectEqual(@as(f32, 17.0), result[2]); // 5*1 + 6*2
}
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
