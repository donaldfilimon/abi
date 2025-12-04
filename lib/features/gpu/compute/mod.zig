//! GPU Compute Module
//!
//! GPU-accelerated compute operations and kernels

const std = @import("std");

// GPU compute components
pub const kernels = @import("kernels.zig");
pub const gpu_backend_manager = @import("gpu_backend_manager.zig");
pub const gpu_ai_acceleration = @import("gpu_ai_acceleration.zig");

test {
    std.testing.refAllDecls(@This());
}
