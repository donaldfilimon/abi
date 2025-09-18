//! GPU Demo Module
//!
//! GPU demonstration and example code

const std = @import("std");

// GPU demo components
pub const gpu_demo = @import("gpu_demo.zig");
pub const enhanced_gpu_demo = @import("enhanced_gpu_demo.zig");
pub const advanced_gpu_demo = @import("advanced_gpu_demo.zig");

// Shader resources
pub const test_shader = @embedFile("test_shader.glsl");

test {
    std.testing.refAllDecls(@This());
}
