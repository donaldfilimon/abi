//! Shared Utilities Module
//!
//! Provides fundamental building blocks and generic helpers used throughout
//! the ABI framework. Organized by functional sub-modules.
//!
//! # Overview
//!
//! This module provides core utilities organized into these categories:
//!
//! - **Time**: Platform-aware time functions compatible with Zig 0.16
//! - **Math**: Statistical functions (mean, variance, percentile, etc.)
//! - **String**: String manipulation and parsing utilities
//! - **Lifecycle**: Module initialization and teardown helpers
//! - **Retry**: Exponential backoff for retry logic
//! - **Binary**: Serialization and cursor-based binary I/O
//!
//! # Sub-modules
//!
//! For specialized functionality, see these sub-modules:
//!
//! | Module | Description |
//! |--------|-------------|
//! | `crypto` | Cryptographic hashing and random generation |
//! | `encoding` | Base64, hex, URL encoding/decoding |
//! | `fs` | Filesystem operations (Zig 0.16 compatible) |
//! | `http` | HTTP client with retry and connection pooling |
//! | `json` | JSON parsing and serialization |
//! | `memory` | Memory pools and specialized allocators |
//! | `net` | Network address parsing and socket utilities |
//!
//! # Usage
//!
//! ```zig
//! const utils = @import("utils.zig");
//!
//! // Time utilities
//! const now = utils.unixSeconds();
//! utils.sleepMs(100);
//!
//! // Math utilities
//! const avg = utils.math.mean(&.{1.0, 2.0, 3.0});
//! const p95 = try utils.math.percentile(allocator, values, 0.95);
//!
//! // String utilities
//! const trimmed = utils.string.trimWhitespace("  hello  ");
//! const lower = try utils.string.toLowerAscii(allocator, "HELLO");
//! ```

const std = @import("std");
const platform_time = @import("time.zig");

// ============================================================================
// Time Utilities (Zig 0.16 compatible, platform-aware)
// ============================================================================

/// Get current time as Unix timestamp in seconds.
///
/// Uses monotonic time since application start, converted to approximate
/// Unix time. Suitable for relative timing but may drift from wall clock.
///
/// For high-precision timing, use `nowNanoseconds()` instead.
pub fn unixSeconds() i64 {
    return platform_time.unixSeconds();
}

/// Get current time as Unix timestamp in seconds.
/// Alias for `unixSeconds()` for API compatibility.
pub fn nowSeconds() i64 {
    return platform_time.nowSeconds();
}

/// Get current time as Unix timestamp in milliseconds.
///
/// Uses monotonic time since application start, converted to approximate
/// Unix time in milliseconds. Suitable for timing and timeout calculations.
pub fn unixMs() i64 {
    return platform_time.unixMs();
}

/// Get current time as Unix timestamp in milliseconds.
/// Alias for `unixMs()` for API compatibility.
pub fn nowMs() i64 {
    return platform_time.nowMs();
}

/// Get current monotonic time in nanoseconds.
///
/// Returns a high-precision monotonic timestamp suitable for measuring
/// elapsed time. Not affected by system clock adjustments.
pub fn nowNanoseconds() i64 {
    return platform_time.nowNanoseconds();
}

/// Get current time in milliseconds.
/// Alias for `nowMs()` for legacy compatibility.
pub fn nowMilliseconds() i64 {
    return platform_time.nowMilliseconds();
}

/// Get current time as Unix timestamp in milliseconds.
/// Alias for `unixMs()`.
pub const unixMilliseconds = unixMs;

/// Sleep the current thread for the specified number of milliseconds.
///
/// Uses the platform's native sleep mechanism. On WASM/freestanding
/// targets, this is a no-op.
pub fn sleepMs(ms: u64) void {
    platform_time.sleepMs(ms);
}

/// Sleep the current thread for the specified number of nanoseconds.
///
/// Uses the platform's native sleep mechanism. Precision depends on
/// the operating system's scheduler granularity.
pub fn sleepNs(ns: u64) void {
    platform_time.sleepNs(ns);
}

// ============================================================================
// Math Utilities
// ============================================================================

/// Mathematical utility functions for statistics and numerical operations.
///
/// Includes basic statistics (mean, variance, standard deviation),
/// percentile calculations, and interpolation functions.
pub const math = struct {
    /// Result of min/max calculation containing both values.
    pub const MinMax = struct {
        min: f64,
        max: f64,
    };

    /// Clamp a value to be within the specified range [min_value, max_value].
    ///
    /// Returns min_value if value < min_value, max_value if value > max_value,
    /// otherwise returns value unchanged. Works with any comparable type.
    pub fn clamp(value: anytype, min_value: @TypeOf(value), max_value: @TypeOf(value)) @TypeOf(value) {
        return std.math.clamp(value, min_value, max_value);
    }

    /// Linear interpolation between two values.
    ///
    /// Returns a + (b - a) * t, where t=0 gives a and t=1 gives b.
    /// Values of t outside [0,1] extrapolate beyond a and b.
    pub fn lerp(a: f64, b: f64, t: f64) f64 {
        return a + (b - a) * t;
    }

    /// Calculate the arithmetic mean (average) of a slice of values.
    ///
    /// Returns 0 for an empty slice.
    pub fn mean(values: []const f64) f64 {
        if (values.len == 0) return 0;
        var total: f64 = 0;
        for (values) |value| {
            total += value;
        }
        return total / @as(f64, @floatFromInt(values.len));
    }

    /// Calculate the sum of all values in a slice.
    pub fn sum(values: []const f64) f64 {
        var total: f64 = 0;
        for (values) |value| {
            total += value;
        }
        return total;
    }

    /// Calculate the population variance of a slice of values.
    ///
    /// Variance measures how spread out values are from the mean.
    /// Returns 0 for an empty slice.
    pub fn variance(values: []const f64) f64 {
        if (values.len == 0) return 0;
        const avg = mean(values);
        var acc: f64 = 0;
        for (values) |value| {
            const diff = value - avg;
            acc += diff * diff;
        }
        return acc / @as(f64, @floatFromInt(values.len));
    }

    /// Calculate the population standard deviation of a slice of values.
    ///
    /// Standard deviation is the square root of variance, giving a measure
    /// of spread in the same units as the original values.
    pub fn stddev(values: []const f64) f64 {
        return std.math.sqrt(variance(values));
    }

    /// Find the minimum and maximum values in a slice.
    ///
    /// Returns null for an empty slice.
    pub fn minMax(values: []const f64) ?MinMax {
        if (values.len == 0) return null;
        var min_value = values[0];
        var max_value = values[0];
        for (values[1..]) |value| {
            if (value < min_value) min_value = value;
            if (value > max_value) max_value = value;
        }
        return .{ .min = min_value, .max = max_value };
    }

    /// Calculate the median of an already-sorted slice of values.
    ///
    /// For even-length slices, returns the average of the two middle values.
    /// Returns 0 for an empty slice.
    pub fn medianSorted(values: []const f64) f64 {
        if (values.len == 0) return 0;
        const mid = values.len / 2;
        if (values.len % 2 == 1) return values[mid];
        return (values[mid - 1] + values[mid]) / 2.0;
    }

    /// Calculate the median of an unsorted slice of values.
    ///
    /// Creates a sorted copy internally, so requires an allocator.
    /// Returns 0 for an empty slice.
    pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
        if (values.len == 0) return 0;
        const copy = try allocator.dupe(f64, values);
        defer allocator.free(copy);
        std.sort.heap(f64, copy, {}, comptime std.sort.asc(f64));
        return medianSorted(copy);
    }

    /// Calculate a percentile value from an already-sorted slice.
    ///
    /// The percentile_value should be in [0.0, 1.0], where 0.5 is the median
    /// and 0.95 is the 95th percentile. Uses linear interpolation for
    /// fractional positions.
    pub fn percentileSorted(values: []const f64, percentile_value: f64) f64 {
        if (values.len == 0) return 0;
        const clamped = clamp(percentile_value, 0.0, 1.0);
        const position = clamped * @as(f64, @floatFromInt(values.len - 1));
        const lower_index: usize = @intFromFloat(@floor(position));
        const upper_index: usize = @intFromFloat(@ceil(position));
        if (lower_index == upper_index) return values[lower_index];
        const weight = position - @as(f64, @floatFromInt(lower_index));
        return lerp(values[lower_index], values[upper_index], weight);
    }

    /// Calculate a percentile value from an unsorted slice.
    ///
    /// Creates a sorted copy internally, so requires an allocator.
    /// The percentile_value should be in [0.0, 1.0].
    ///
    /// Example:
    /// ```zig
    /// const p95 = try math.percentile(allocator, response_times, 0.95);
    /// ```
    pub fn percentile(
        allocator: std.mem.Allocator,
        values: []const f64,
        percentile_value: f64,
    ) !f64 {
        if (values.len == 0) return 0;
        const copy = try allocator.dupe(f64, values);
        defer allocator.free(copy);
        std.sort.heap(f64, copy, {}, comptime std.sort.asc(f64));
        return percentileSorted(copy, percentile_value);
    }
};

// ============================================================================
// String Utilities
// ============================================================================

/// String manipulation utilities for parsing and transforming text.
///
/// All functions work with ASCII strings. For Unicode handling,
/// use std.unicode directly.
pub const string = struct {
    /// Result of splitting a string once on a delimiter.
    pub const SplitPair = struct {
        /// The part before the delimiter.
        head: []const u8,
        /// The part after the delimiter.
        tail: []const u8,
    };

    /// Trim leading and trailing whitespace from a string.
    ///
    /// Removes spaces, tabs, carriage returns, and newlines.
    /// Returns a slice into the original string (no allocation).
    pub fn trimWhitespace(input: []const u8) []const u8 {
        return std.mem.trim(u8, input, " \t\r\n");
    }

    /// Split a string on the first occurrence of a delimiter.
    ///
    /// Returns null if the delimiter is not found.
    /// The delimiter itself is not included in either part.
    ///
    /// Example:
    /// ```zig
    /// const result = string.splitOnce("key=value", '=');
    /// // result.head == "key", result.tail == "value"
    /// ```
    pub fn splitOnce(input: []const u8, delimiter: u8) ?SplitPair {
        const pair = std.mem.splitOnce(u8, input, delimiter);
        return if (pair) |p| SplitPair{ .head = p.head, .tail = p.tail } else null;
    }

    /// Parse a string as a boolean value.
    ///
    /// Recognizes: "true", "1" (case-insensitive) as true.
    /// Recognizes: "false", "0" (case-insensitive) as false.
    /// Returns null for any other input.
    pub fn parseBool(input: []const u8) ?bool {
        if (std.ascii.eqlIgnoreCase(input, "true") or std.mem.eql(u8, input, "1")) {
            return true;
        }
        if (std.ascii.eqlIgnoreCase(input, "false") or std.mem.eql(u8, input, "0")) {
            return false;
        }
        return null;
    }

    /// Convert a string to lowercase ASCII (allocating).
    ///
    /// Returns a newly allocated string that must be freed by the caller.
    /// Non-ASCII characters are copied unchanged.
    pub fn toLowerAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const copy = try allocator.alloc(u8, input.len);
        for (input, 0..) |char, i| {
            copy[i] = std.ascii.toLower(char);
        }
        return copy;
    }

    /// Convert a string to uppercase ASCII (allocating).
    ///
    /// Returns a newly allocated string that must be freed by the caller.
    /// Non-ASCII characters are copied unchanged.
    pub fn toUpperAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const copy = try allocator.alloc(u8, input.len);
        for (input, 0..) |char, i| {
            copy[i] = std.ascii.toUpper(char);
        }
        return copy;
    }

    /// Convert a string to lowercase ASCII in-place.
    ///
    /// Modifies the buffer directly and returns it for chaining.
    /// Non-ASCII characters are unchanged.
    pub fn lowerStringMut(buf: []u8) []u8 {
        for (buf, 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        return buf;
    }

    /// Case-insensitive string equality comparison.
    ///
    /// Compares ASCII characters ignoring case. Non-ASCII characters
    /// must match exactly. Returns false if lengths differ.
    pub inline fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return false;
        for (a, b) |ac, bc| {
            if (std.ascii.toLower(ac) != std.ascii.toLower(bc)) {
                return false;
            }
        }
        return true;
    }

    /// Compute a case-insensitive hash of a string.
    ///
    /// Uses Wyhash with lowercase conversion. Useful for case-insensitive
    /// hash maps.
    pub fn hashIgnoreCase(s: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (s) |c| {
            hasher.update(&[_]u8{std.ascii.toLower(c)});
        }
        return hasher.final();
    }
};

// ---------------------------------------------------------------------------
// Unit Tests for time utilities
// ---------------------------------------------------------------------------

test "unixMs monotonic" {
    const start = unixMs();
    sleepMs(5);
    const later = unixMs();
    try std.testing.expect(later > start);
}

test "sleepMs respects duration" {
    const before = unixMs();
    const wait: u64 = 20;
    sleepMs(wait);
    const after = unixMs();
    try std.testing.expect((after - before) >= @as(i64, @intCast(wait)));
}

// ============================================================================
// Lifecycle Management
// ============================================================================

/// Utilities for managing feature and module lifecycles.
///
/// Provides compile-time feature gating and runtime initialization tracking.
/// Used by feature modules to ensure proper init/deinit ordering.
pub const lifecycle = struct {
    /// Errors returned when attempting to use a disabled feature.
    /// Each error corresponds to a build-time feature flag.
    pub const FeatureDisabledError = error{
        AiDisabled,
        GpuDisabled,
        WebDisabled,
        DatabaseDisabled,
        NetworkDisabled,
        ProfilingDisabled,
        MonitoringDisabled,
        ExploreDisabled,
        LlmDisabled,
        TestDisabled,
    };

    /// Identifiers for build-time features.
    /// Used to generate appropriate error types for disabled features.
    pub const FeatureId = enum {
        ai_disabled,
        gpu_disabled,
        web_disabled,
        database_disabled,
        network_disabled,
        profiling_disabled,
        monitoring_disabled,
        explore_disabled,
        llm_disabled,
        test_disabled,

        /// Convert this feature ID to its corresponding disabled error.
        pub fn toError(self: FeatureId) FeatureDisabledError {
            return switch (self) {
                .ai_disabled => error.AiDisabled,
                .gpu_disabled => error.GpuDisabled,
                .web_disabled => error.WebDisabled,
                .database_disabled => error.DatabaseDisabled,
                .network_disabled => error.NetworkDisabled,
                .profiling_disabled => error.ProfilingDisabled,
                .monitoring_disabled => error.MonitoringDisabled,
                .explore_disabled => error.ExploreDisabled,
                .llm_disabled => error.LlmDisabled,
                .test_disabled => error.TestDisabled,
            };
        }
    };

    /// Compile-time feature lifecycle management.
    ///
    /// Creates a type that tracks initialization state and returns
    /// appropriate errors when a feature is disabled at compile time.
    ///
    /// Example:
    /// ```zig
    /// const GpuLifecycle = FeatureLifecycle(build_options.enable_gpu, .gpu_disabled);
    /// var gpu_lifecycle = GpuLifecycle{};
    /// try gpu_lifecycle.init(allocator); // Returns GpuDisabled if disabled
    /// ```
    pub fn FeatureLifecycle(comptime enabled: bool, comptime feature_id: FeatureId) type {
        return struct {
            initialized: bool = false,
            const Self = @This();

            /// Initialize the feature. Returns feature-disabled error if not enabled.
            pub fn init(self: *Self, _: std.mem.Allocator) !void {
                if (!comptime enabled) return feature_id.toError();
                self.initialized = true;
            }

            /// Deinitialize the feature. Safe to call even if not initialized.
            pub fn deinit(self: *Self) void {
                self.initialized = false;
            }

            /// Check if this feature is enabled at compile time.
            pub fn isEnabled(_: *const Self) bool {
                return enabled;
            }

            /// Check if this feature has been initialized at runtime.
            pub fn isInitialized(self: *const Self) bool {
                return self.initialized;
            }
        };
    }
};

/// Errors that can occur during module lifecycle operations.
pub const LifecycleError = error{
    /// Module was already initialized; cannot initialize twice.
    AlreadyInitialized,
    /// Operation requires initialization, but module was not initialized.
    NotInitialized,
    /// Initialization callback returned an error.
    InitFailed,
};

/// Simple module lifecycle management with init/deinit callbacks.
///
/// Tracks initialization state and prevents double-initialization.
/// Use for modules that need to perform setup/teardown operations.
///
/// Example:
/// ```zig
/// var lifecycle = SimpleModuleLifecycle{};
///
/// pub fn init() !void {
///     try lifecycle.init(doSetup);
/// }
///
/// pub fn deinit() void {
///     lifecycle.deinit(doTeardown);
/// }
/// ```
pub const SimpleModuleLifecycle = struct {
    initialized: bool = false,

    /// Initialize the module, calling the provided init function.
    ///
    /// Returns `AlreadyInitialized` if already initialized.
    /// Returns `InitFailed` if the init function returns an error.
    pub fn init(self: *SimpleModuleLifecycle, initFn: ?*const fn () anyerror!void) LifecycleError!void {
        if (self.initialized) return LifecycleError.AlreadyInitialized;
        if (initFn) |f| {
            f() catch return LifecycleError.InitFailed;
        }
        self.initialized = true;
    }

    /// Deinitialize the module, calling the provided deinit function.
    ///
    /// Safe to call even if not initialized (will be a no-op).
    pub fn deinit(self: *SimpleModuleLifecycle, deinitFn: ?*const fn () void) void {
        if (!self.initialized) return;
        if (deinitFn) |f| f();
        self.initialized = false;
    }

    /// Check if the module has been initialized.
    pub fn isInitialized(self: *const SimpleModuleLifecycle) bool {
        return self.initialized;
    }
};

// ============================================================================
// Retry Utilities
// ============================================================================

/// Retry logic utilities for handling transient failures.
///
/// Provides exponential backoff and other retry strategies commonly
/// used for network requests and external service calls.
pub const retry = struct {
    /// Exponential backoff calculator for retry delays.
    ///
    /// Starts at an initial delay and multiplies by a factor on each
    /// retry, up to a maximum delay. Useful for:
    /// - API rate limiting
    /// - Database connection retries
    /// - Network request failures
    ///
    /// Example:
    /// ```zig
    /// var backoff = ExponentialBackoff.init(100, 30000, 2.0);
    /// while (retry_count < max_retries) {
    ///     const result = try_operation() catch {
    ///         backoff.wait(); // Sleep with exponential delay
    ///         retry_count += 1;
    ///         continue;
    ///     };
    ///     break;
    /// }
    /// ```
    pub const ExponentialBackoff = struct {
        /// Current delay in milliseconds.
        current_ms: u64,
        /// Maximum delay cap in milliseconds.
        max_ms: u64,
        /// Multiplier applied after each wait (e.g., 2.0 for doubling).
        multiplier: f32,

        /// Create a new exponential backoff calculator.
        ///
        /// - `initial_ms`: Starting delay in milliseconds
        /// - `max_ms`: Maximum delay cap
        /// - `multiplier`: Factor to multiply delay by (e.g., 2.0)
        pub fn init(initial_ms: u64, max_ms: u64, multiplier: f32) ExponentialBackoff {
            return .{
                .current_ms = initial_ms,
                .max_ms = max_ms,
                .multiplier = multiplier,
            };
        }

        /// Sleep for the current backoff duration, then increase the delay.
        ///
        /// The delay is multiplied by the multiplier after sleeping,
        /// but capped at max_ms.
        pub fn wait(self: *ExponentialBackoff) void {
            sleepMs(self.current_ms);
            self.current_ms = @intFromFloat(@min(@as(f32, @floatFromInt(self.max_ms)), @as(f32, @floatFromInt(self.current_ms)) * self.multiplier));
        }
    };
};

// ============================================================================
// Binary Utilities
// ============================================================================

/// Binary serialization utilities for wire protocols and file formats.
///
/// Provides cursor-based reading and writing of binary data with
/// automatic endianness handling.
pub const binary = @import("utils/binary.zig");

/// Writer for serializing data to binary format.
/// Supports little-endian and big-endian output.
pub const SerializationWriter = binary.SerializationWriter;

/// Cursor for reading serialized binary data.
/// Tracks position and provides typed reads.
pub const SerializationCursor = binary.SerializationCursor;

// ============================================================================
// Sub-module Imports
// ============================================================================

/// Cryptographic utilities: hashing, random bytes, secure comparison.
/// Wraps std.crypto with convenient higher-level APIs.
pub const crypto = @import("utils/crypto/mod.zig");

/// Encoding/decoding utilities: Base64, hex, URL encoding.
/// Both allocating and buffer-based APIs available.
pub const encoding = @import("utils/encoding/mod.zig");

/// Filesystem utilities compatible with Zig 0.16's I/O model.
/// File reading, writing, directory operations.
pub const fs = @import("utils/fs/mod.zig");

/// HTTP client utilities for making web requests.
/// Includes connection pooling and timeout handling.
pub const http = @import("utils/http/mod.zig");

/// Async HTTP client for non-blocking requests.
/// Built on Zig's async/await infrastructure.
pub const async_http = @import("utils/http/async_http.zig");

/// JSON parsing and serialization.
/// Streaming parser and DOM-based APIs.
pub const json = @import("utils/json/mod.zig");

/// Memory management utilities: pools, arenas, tracking.
/// Specialized allocators for different use cases.
pub const memory = @import("utils/memory/mod.zig");

/// Network utilities: address parsing, DNS, socket helpers.
/// Platform-independent networking primitives.
pub const net = @import("utils/net/mod.zig");

/// Configuration file loading and parsing.
/// Supports JSON, TOML, and environment variable overrides.
pub const config = @import("utils/config.zig");

// v2 utility modules
/// Foundational primitives: Math, String, Time, Atomic, Platform, RingBuffer (v2).
pub const v2_primitives = @import("utils/v2_primitives.zig");
/// Structured error handling with categories, severity, and accumulation (v2).
pub const structured_error = @import("utils/structured_error.zig");
/// SwissMap: high-performance open-addressing hash map with H2 control bytes (v2).
pub const swiss_map = @import("utils/swiss_map.zig");
/// ABIX binary serialization: compact wire format with comptime struct support (v2).
pub const abix_serialize = @import("utils/abix_serialize.zig");
/// Hierarchical span-based profiler with Chrome Trace export (v2).
pub const profiler = @import("utils/profiler.zig");
/// Statistical benchmark suite with Chauvenet outlier filtering (v2).
pub const benchmark = @import("utils/benchmark.zig");

/// HTTP retry logic with backoff strategies.
/// Configurable retry policies for web requests.
pub const http_retry = @import("utils/retry.zig");

// Note: Top-level shared modules (logging, os, platform, plugins, simd, time)
// are exported via shared/mod.zig. Access them as abi.shared.<module>.

// ============================================================================
// Tests
// ============================================================================

test "Math helpers" {
    const vals = [_]f64{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(f64, 3.0), math.mean(&vals));
}

test "String helpers" {
    try std.testing.expect(string.eqlIgnoreCase("HELLO", "hello"));
}

test "Time helpers" {
    const s = unixSeconds();
    // On first call, elapsed time from app start might be 0
    try std.testing.expect(s >= 0);
}
