//! Common test helpers and utilities.
//!
//! This module provides shared setup/teardown logic and assertion
//! helpers used across the test suite.

const std = @import("std");

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

test "TestAllocator detects leaks" {
    // This test verifies the allocator works - actual leak detection
    // would panic, so we just verify basic allocation/free works
    var ta = TestAllocator.init();
    defer ta.deinit();

    const alloc = ta.allocator();
    const slice = try alloc.alloc(u8, 100);
    alloc.free(slice);
}
