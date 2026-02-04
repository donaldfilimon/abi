//! Common test helpers and utilities.
//!
//! This module provides shared setup/teardown logic and assertion
//! helpers used across the test suite.

const std = @import("std");
const abi = @import("abi");
const time = abi.shared.time;

// ============================================================================
// Time Utilities
// ============================================================================

/// Sleep for a specified number of milliseconds.
/// On WASM, this is a no-op (can't block in WASM).
/// Re-exports from shared/time.zig for test convenience.
pub const sleepMs = time.sleepMs;

/// Sleep for a specified number of nanoseconds.
pub const sleepNs = time.sleepNs;

/// Test allocator with leak detection.
/// Wraps GeneralPurposeAllocator with automatic leak checking on deinit.
pub const TestAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{
        .stack_trace_frames = 10,
    }),

    pub fn init() TestAllocator {
        return .{ .gpa = .{} };
    }

    pub fn allocator(self: *TestAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    /// Deinitializes and checks for memory leaks.
    /// Panics if leaks are detected.
    pub fn deinit(self: *TestAllocator) void {
        const check = self.gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in test");
        }
    }
};

/// Skip test if GPU hardware is not available.
/// Use at the start of GPU-dependent tests.
pub fn skipIfNoGpu() error{SkipZigTest}!void {
    if (!hasGpuSupport()) {
        return error.SkipZigTest;
    }
}

fn hasGpuSupport() bool {
    const builtin = @import("builtin");
    // Skip GPU tests on WASM and freestanding
    return builtin.os.tag != .freestanding and builtin.cpu.arch != .wasm32;
}

/// Skip test if timer is not available on this platform.
pub fn skipIfNoTimer() error{SkipZigTest}!void {
    _ = std.time.Timer.start() catch return error.SkipZigTest;
}

// ============================================================================
// Vector Test Utilities
// ============================================================================

/// Generate a random vector for testing.
/// Fills the provided buffer with random values in the range [-1, 1].
pub fn generateRandomVector(rng: *std.Random.DefaultPrng, buffer: []f32) void {
    for (buffer) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }
}

/// Generate a random vector with allocation.
/// Returns a newly allocated slice of random values in [-1, 1].
pub fn generateRandomVectorAlloc(
    allocator: std.mem.Allocator,
    rng: *std.Random.DefaultPrng,
    dims: usize,
) ![]f32 {
    const vec = try allocator.alloc(f32, dims);
    generateRandomVector(rng, vec);
    return vec;
}

/// Normalize a vector to unit length.
pub fn normalizeVector(vec: []f32) void {
    var sum: f32 = 0;
    for (vec) |v| {
        sum += v * v;
    }
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }
}

test "generateRandomVector produces valid values" {
    var rng = std.Random.DefaultPrng.init(42);
    var buffer: [128]f32 = undefined;
    generateRandomVector(&rng, &buffer);

    for (buffer) |v| {
        try std.testing.expect(v >= -1.0 and v <= 1.0);
    }
}

test "normalizeVector produces unit vector" {
    var vec = [_]f32{ 3.0, 4.0 };
    normalizeVector(&vec);

    var sum_sq: f32 = 0;
    for (vec) |v| {
        sum_sq += v * v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_sq, 0.0001);
}

test "TestAllocator detects leaks" {
    // This test verifies the allocator works - actual leak detection
    // would panic, so we just verify basic allocation/free works
    var ta = TestAllocator.init();
    defer ta.deinit();

    const alloc = ta.allocator();
    const slice = try alloc.alloc(u8, 100);
    alloc.free(slice);
}
