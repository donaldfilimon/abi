//! CUDA (Compute Unified Device Architecture) Code Generator
//!
//! Provides the pre-instantiated generator for CUDA C++ output and
//! vision-specific kernel generators.

const std = @import("std");
const generic = @import("generic.zig");
const vision = @import("vision_kernels.zig");

/// Pre-instantiated CUDA generator using standard CUDA configuration.
pub const Generator = generic.CudaGenerator;

/// Re-export for backward compatibility.
pub const CudaGenerator = generic.CudaGenerator;

/// Centralized vision operations for CUDA.
pub const VisionKernels = vision.VisionKernels;

/// Compatibility aliases for the rest of the DSL system
pub const Language = .cuda;

test "CudaGenerator availability" {
    const allocator = std.testing.allocator;
    var g = Generator.init(allocator, .{});
    defer g.deinit();

    try std.testing.expect(g.backend_config.language == .cuda);
}
