//! WGSL (WebGPU Shading Language) Code Generator
//!
//! Provides the pre-instantiated generator for WGSL output and
//! vision-specific kernel generators.

const std = @import("std");
const generic = @import("generic.zig");
const vision = @import("vision_kernels.zig");

/// Pre-instantiated WGSL generator using standard WGSL configuration.
pub const Generator = generic.WgslGenerator;

/// Re-export for backward compatibility.
pub const WgslGenerator = generic.WgslGenerator;

/// Centralized vision operations for WGSL.
pub const VisionKernels = vision.VisionKernels;

// Compatibility aliases for the rest of the DSL system
pub const Language = .wgsl;

test "WgslGenerator availability" {
    const allocator = std.testing.allocator;
    var g = Generator.init(allocator);
    defer g.deinit();

    try std.testing.expect(g.backend_config.language == .wgsl);
}
