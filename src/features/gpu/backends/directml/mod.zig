//! DirectML Backend for Windows GPU Acceleration
//!
//! Provides GPU compute via Microsoft's DirectML API on Windows.
//! Falls back gracefully on non-Windows platforms.
//!
//! DirectML is a high-performance, hardware-accelerated DirectX 12 library
//! for machine learning operators. It provides GPU acceleration for common
//! ML workloads across all DirectX 12-capable hardware.

const std = @import("std");
const builtin = @import("builtin");

pub const loader = @import("loader.zig");
pub const operators = @import("operators.zig");

pub fn isAvailable() bool {
    if (builtin.os.tag != .windows) return false;
    return loader.canLoadDirectML();
}

test "DirectML availability" {
    const available = isAvailable();
    if (builtin.os.tag != .windows) {
        try std.testing.expect(!available);
    }
}

test "DirectML submodule imports" {
    // Verify submodules are importable (compile-time check)
    _ = loader.DirectMlDevice;
    _ = operators.DmlMatMul;
    _ = operators.DmlConvolution;
    _ = operators.DmlActivation;
}

test {
    std.testing.refAllDecls(@This());
}
