//! Shared Utilities Module
//!
//! Provides fundamental building blocks and generic helpers used throughout
//! the ABI framework. Organized by functional sub-modules.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Time Utilities
// ============================================================================

/// Get current time in unix seconds.
pub fn unixSeconds() i64 {
    return std.time.timestamp();
}

/// Get current time in unix milliseconds.
pub fn unixMs() i64 {
    return std.time.milliTimestamp();
}

/// Sleep for a specified number of milliseconds.
pub fn sleepMs(ms: u64) void {
    std.time.sleep(ms * std.time.ns_per_ms);
}

/// Get current time in nanoseconds.
pub fn nowNanoseconds() i64 {
    return std.time.nanoTimestamp();
}

// ============================================================================
// Math Utilities
// ============================================================================

pub const math = struct {
    pub const MinMax = struct {
        min: f64,
        max: f64,
    };

    pub fn clamp(value: anytype, min_value: @TypeOf(value), max_value: @TypeOf(value)) @TypeOf(value) {
        return std.math.clamp(value, min_value, max_value);
    }

    pub fn lerp(a: f64, b: f64, t: f64) f64 {
        return a + (b - a) * t;
    }

    pub fn mean(values: []const f64) f64 {
        if (values.len == 0) return 0;
        var total: f64 = 0;
        for (values) |value| {
            total += value;
        }
        return total / @as(f64, @floatFromInt(values.len));
    }

    pub fn sum(values: []const f64) f64 {
        var total: f64 = 0;
        for (values) |value| {
            total += value;
        }
        return total;
    }

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

    pub fn stddev(values: []const f64) f64 {
        return std.math.sqrt(variance(values));
    }

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

    pub fn medianSorted(values: []const f64) f64 {
        if (values.len == 0) return 0;
        const mid = values.len / 2;
        if (values.len % 2 == 1) return values[mid];
        return (values[mid - 1] + values[mid]) / 2.0;
    }

    pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
        if (values.len == 0) return 0;
        const copy = try allocator.dupe(f64, values);
        defer allocator.free(copy);
        std.sort.heap(f64, copy, {}, comptime std.sort.asc(f64));
        return medianSorted(copy);
    }

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

pub const string = struct {
    pub const SplitPair = struct {
        head: []const u8,
        tail: []const u8,
    };

    pub fn trimWhitespace(input: []const u8) []const u8 {
        return std.mem.trim(u8, input, " \t\r\n");
    }

    pub fn splitOnce(input: []const u8, delimiter: u8) ?SplitPair {
        const pair = std.mem.splitOnce(u8, input, delimiter);
        return if (pair) |p| SplitPair{ .head = p.head, .tail = p.tail } else null;
    }

    pub fn parseBool(input: []const u8) ?bool {
        if (std.ascii.eqlIgnoreCase(input, "true") or std.mem.eql(u8, input, "1")) {
            return true;
        }
        if (std.ascii.eqlIgnoreCase(input, "false") or std.mem.eql(u8, input, "0")) {
            return false;
        }
        return null;
    }

    pub fn toLowerAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const copy = try allocator.alloc(u8, input.len);
        for (input, 0..) |char, i| {
            copy[i] = std.ascii.toLower(char);
        }
        return copy;
    }

    pub fn lowerStringMut(buf: []u8) []u8 {
        for (buf, 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        return buf;
    }

    pub inline fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return false;
        for (a, b) |ac, bc| {
            if (std.ascii.toLower(ac) != std.ascii.toLower(bc)) {
                return false;
            }
        }
        return true;
    }

    pub fn hashIgnoreCase(s: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (s) |c| {
            hasher.update(&[_]u8{std.ascii.toLower(c)});
        }
        return hasher.final();
    }
};

// ============================================================================
// Lifecycle Management
// ============================================================================

pub const lifecycle = struct {
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

    pub fn FeatureLifecycle(comptime enabled: bool, comptime feature_id: FeatureId) type {
        return struct {
            initialized: bool = false,
            const Self = @This();
            pub fn init(self: *Self, _: std.mem.Allocator) !void {
                if (!comptime enabled) return feature_id.toError();
                self.initialized = true;
            }
            pub fn deinit(self: *Self) void {
                self.initialized = false;
            }
            pub fn isEnabled(_: *const Self) bool {
                return enabled;
            }
            pub fn isInitialized(self: *const Self) bool {
                return self.initialized;
            }
        };
    }
};

pub const LifecycleError = error{
    AlreadyInitialized,
    NotInitialized,
    InitFailed,
};

/// Simple module lifecycle management with init callback.
pub const SimpleModuleLifecycle = struct {
    initialized: bool = false,

    pub fn init(self: *SimpleModuleLifecycle, initFn: ?*const fn () anyerror!void) LifecycleError!void {
        if (self.initialized) return LifecycleError.AlreadyInitialized;
        if (initFn) |f| {
            f() catch return LifecycleError.InitFailed;
        }
        self.initialized = true;
    }

    pub fn deinit(self: *SimpleModuleLifecycle, deinitFn: ?*const fn () void) void {
        if (!self.initialized) return;
        if (deinitFn) |f| f();
        self.initialized = false;
    }

    pub fn isInitialized(self: *const SimpleModuleLifecycle) bool {
        return self.initialized;
    }
};

// ============================================================================
// Retry Utilities
// ============================================================================

pub const retry = struct {
    pub const ExponentialBackoff = struct {
        current_ms: u64,
        max_ms: u64,
        multiplier: f32,

        pub fn init(initial_ms: u64, max_ms: u64, multiplier: f32) ExponentialBackoff {
            return .{
                .current_ms = initial_ms,
                .max_ms = max_ms,
                .multiplier = multiplier,
            };
        }

        pub fn wait(self: *ExponentialBackoff) void {
            sleepMs(self.current_ms);
            self.current_ms = @intFromFloat(@min(@as(f32, @floatFromInt(self.max_ms)), @as(f32, @floatFromInt(self.current_ms)) * self.multiplier));
        }
    };
};

// ============================================================================
// Binary Utilities
// ============================================================================

pub const binary = struct {
    pub fn readInt(comptime T: type, bytes: []const u8, endian: std.builtin.Endian) T {
        return std.mem.readInt(T, bytes[0..@sizeOf(T)], endian);
    }

    pub fn writeInt(comptime T: type, buf: []u8, value: T, endian: std.builtin.Endian) void {
        std.mem.writeInt(T, buf[0..@sizeOf(T)], value, endian);
    }
};

// ============================================================================
// Sub-module Imports (for legacy or complex modules)
// ============================================================================

pub const crypto = @import("utils/crypto/mod.zig");
pub const encoding = @import("utils/encoding/mod.zig");
pub const fs = @import("utils/fs/mod.zig");
pub const http = @import("utils/http/mod.zig");
pub const json = @import("utils/json/mod.zig");
pub const memory = @import("utils/memory/mod.zig");
pub const net = @import("utils/net/mod.zig");
pub const config = @import("utils/config.zig");

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
    try std.testing.expect(s > 0);
}
