//! Omni-Compute Module
//!
//! Provides the distributed mesh networking, multi-GPU orchestration,
//! and tensor sharing protocols.

pub const mesh = @import("mesh.zig");

const std = @import("std");

pub const ComputeError = error{
    MeshUnavailable,
    NodeUnreachable,
    TaskFailed,
    OutOfMemory,
};

pub const Error = ComputeError;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = true };
    }

    pub fn deinit(self: *Context) void {
        self.initialized = false;
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
