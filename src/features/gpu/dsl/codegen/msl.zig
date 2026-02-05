//! MSL (Metal Shading Language) Code Generator
//!
//! Provides the pre-instantiated generator for MSL output and
//! vision-specific kernel generators.

const std = @import("std");
const generic = @import("generic.zig");
const vision = @import("vision_kernels.zig");

/// Pre-instantiated MSL generator using standard MSL configuration.
pub const Generator = generic.MslGenerator;

/// Re-export for backward compatibility.
pub const MslGenerator = generic.MslGenerator;

/// Centralized vision operations for MSL.
pub const VisionKernels = vision.VisionKernels;

/// Compatibility aliases for the rest of the DSL system
pub const Language = .msl;

test "MslGenerator availability" {
    const allocator = std.testing.allocator;
    var g = Generator.init(allocator, .{});
    defer g.deinit();

    try std.testing.expect(g.backend_config.language == .msl);
}
