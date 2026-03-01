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
// String Utilities
// ============================================================================

pub const string = struct {
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

// Foundational utility modules
/// Foundational primitives: Math, String, Time, Atomic, Platform, RingBuffer.
pub const primitives = @import("utils/primitives.zig");
/// Structured error handling with categories, severity, and accumulation.
pub const structured_error = @import("utils/structured_error.zig");
/// SwissMap: high-performance open-addressing hash map with H2 control bytes.
pub const swiss_map = @import("utils/swiss_map.zig");
/// ABIX binary serialization: compact wire format with comptime struct support.
pub const abix_serialize = @import("utils/abix_serialize.zig");
/// Hierarchical span-based profiler with Chrome Trace export.
pub const profiler = @import("utils/profiler.zig");
/// Statistical benchmark suite with Chauvenet outlier filtering.
pub const benchmark = @import("utils/benchmark.zig");

/// HTTP retry logic with backoff strategies.
/// Configurable retry policies for web requests.
pub const http_retry = @import("utils/retry.zig");
/// Shared metric primitives (Counter, Gauge, FloatGauge, Histogram).
/// Used by observability, GPU, and AI modules for consistent metrics.
pub const metric_types = @import("utils/metric_types.zig");

// Note: Top-level shared modules (logging, os, platform, plugins, simd, time)
// are exported via shared/mod.zig. Access them as abi.services.shared.<module>.

// ============================================================================
// Tests
// ============================================================================

test "Time helpers" {
    const s = unixSeconds();
    // On first call, elapsed time from app start might be 0
    try std.testing.expect(s >= 0);
}
