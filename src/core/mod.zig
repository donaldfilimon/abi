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

const errors = @import("errors.zig");
const lifecycle = @import("lifecycle.zig");
const logging_mod = @import("logging.zig");

// =============================================================================
// UNIFIED ERROR SYSTEM
// =============================================================================

pub const AbiError = errors.AbiError;
pub const Result = errors.Result;
pub const ok = errors.ok;
pub const err = errors.err;

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

/// Initialize the core system
pub const init = lifecycle.init;
/// Deinitialize the core system
pub const deinit = lifecycle.deinit;
/// Get the global allocator
pub const getAllocator = lifecycle.getAllocator;
/// Check if core system is initialized
pub const isInitialized = lifecycle.isInitialized;

// =============================================================================
// LOGGING SYSTEM
// =============================================================================

pub const LogLevel = logging_mod.LogLevel;
pub const log = logging_mod.log;

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
// VECTOR & MATRIX OPERATIONS
// =============================================================================

pub const VectorOps = struct {
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        const len = @min(a.len, b.len);
        var acc: f32 = 0.0;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            acc += a[i] * b[i];
        }
        return acc;
    }

    pub fn add(result: []f32, a: []const f32, b: []const f32) void {
        std.debug.assert(result.len == a.len and a.len == b.len);
        for (result, a, b) |*r, av, bv| r.* = av + bv;
    }

    pub fn scale(result: []f32, input: []const f32, factor: f32) void {
        std.debug.assert(result.len == input.len);
        for (result, input) |*r, val| r.* = val * factor;
    }

    pub fn normalize(result: []f32, input: []const f32) void {
        std.debug.assert(result.len == input.len);
        const mag = l2Norm(input);
        if (mag == 0.0) {
            @memset(result, 0.0);
            return;
        }
        const inv = 1.0 / mag;
        for (result, input) |*r, val| r.* = val * inv;
    }

    pub fn vectorNormalize(result: []f32, input: []const f32) void {
        normalize(result, input);
    }

    pub fn distance(a: []const f32, b: []const f32) f32 {
        const len = @min(a.len, b.len);
        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std.math.sqrt(sum);
    }

    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        const len = @min(a.len, b.len);
        if (len == 0) return 0.0;
        var dot: f32 = 0.0;
        var mag_a: f32 = 0.0;
        var mag_b: f32 = 0.0;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const av = a[i];
            const bv = b[i];
            dot += av * bv;
            mag_a += av * av;
            mag_b += bv * bv;
        }
        if (mag_a == 0.0 or mag_b == 0.0) return 0.0;
        return dot / (std.math.sqrt(mag_a) * std.math.sqrt(mag_b));
    }

    pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, shared_dim: usize) void {
        std.debug.assert(result.len >= rows * cols);
        std.debug.assert(a.len >= rows * shared_dim);
        std.debug.assert(b.len >= shared_dim * cols);
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            var col: usize = 0;
            while (col < cols) : (col += 1) {
                var acc: f32 = 0.0;
                var k: usize = 0;
                while (k < shared_dim) : (k += 1) {
                    const a_idx = row * shared_dim + k;
                    const b_idx = k * cols + col;
                    acc += a[a_idx] * b[b_idx];
                }
                result[row * cols + col] = acc;
            }
        }
    }

    fn l2Norm(vec: []const f32) f32 {
        var sum: f32 = 0.0;
        for (vec) |v| sum += v * v;
        return std.math.sqrt(sum);
    }
};

pub fn matrixVectorMultiply(result: []f32, weights: []const f32, input: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(result.len >= rows);
    std.debug.assert(weights.len >= rows * cols);
    std.debug.assert(input.len >= cols);
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        var acc: f32 = 0.0;
        var col: usize = 0;
        while (col < cols) : (col += 1) {
            acc += weights[row * cols + col] * input[col];
        }
        result[row] = acc;
    }
}

pub fn matrixVectorMultiplyTranspose(result: []f32, weights: []const f32, input: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(result.len >= rows);
    std.debug.assert(weights.len >= rows * cols);
    std.debug.assert(input.len >= cols);
    std.mem.set(f32, result, 0.0);
    var col: usize = 0;
    while (col < cols) : (col += 1) {
        const input_val = input[col];
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            result[row] += weights[col * rows + row] * input_val;
        }
    }
}

// =============================================================================
// TIME UTILITIES
// =============================================================================// =============================================================================

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
