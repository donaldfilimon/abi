//! Accelerator Module
//!
//! Hardware acceleration interfaces and implementations for AI/ML workloads

const std = @import("std");

// Core accelerator interface
pub const accelerator = @import("accelerator.zig");

// Re-export key types for convenience
pub const Accelerator = accelerator.Accelerator;
pub const DeviceMemory = accelerator.DeviceMemory;
pub const Tensor = accelerator.Tensor;
pub const TensorOps = accelerator.TensorOps;

/// Create the best available accelerator for the current system
pub fn createBestAccelerator(allocator: std.mem.Allocator) !*Accelerator {
    return accelerator.createBestAccelerator(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
