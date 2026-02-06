//! GLSL (OpenGL Shading Language) Code Generator
//!
//! Provides the pre-instantiated generator for GLSL output and
//! vision-specific kernel generators.

const std = @import("std");
const generic = @import("generic.zig");
const vision = @import("vision_kernels.zig");

/// Pre-instantiated GLSL generator using standard GLSL configuration.
pub const Generator = generic.GlslGenerator;

/// Re-export for backward compatibility.
pub const GlslGenerator = generic.GlslGenerator;

/// Centralized vision operations for GLSL.
pub const VisionKernels = vision.VisionKernels;

/// Compatibility aliases for the rest of the DSL system
pub const Language = .glsl;

test "GlslGenerator availability" {
    const allocator = std.testing.allocator;
    var g = Generator.init(allocator);
    defer g.deinit();

    try std.testing.expect(g.backend_config.language == .glsl);
}
