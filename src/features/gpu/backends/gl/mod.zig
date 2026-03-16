//! OpenGL backend implementation for GPU compute.
//! Exposes common GL types, shader loader, runtime context, profile
//! detection, and backend interface for the GPU subsystem.

const std = @import("std");

pub const common = @import("common.zig");
pub const loader = @import("loader.zig");
pub const runtime = @import("runtime.zig");
pub const profile = @import("profile.zig");
pub const backend = @import("backend.zig");

test {
    std.testing.refAllDecls(@This());
}
