//! Omni-Compute Module
//!
//! Provides the distributed mesh networking, multi-GPU orchestration,
//! and tensor sharing protocols.

pub const mesh = @import("mesh.zig");
pub const types = @import("types.zig");

const std = @import("std");

pub const ComputeError = types.ComputeError;
pub const Error = types.Error;

var compute_initialized = std.atomic.Value(bool).init(false);

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        compute_initialized.store(true, .release);
        return .{ .allocator = allocator, .initialized = true };
    }

    pub fn deinit(self: *Context) void {
        self.initialized = false;
        compute_initialized.store(false, .release);
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return compute_initialized.load(.acquire);
}

test {
    std.testing.refAllDecls(@This());
}

test "Context init and deinit" {
    var ctx = Context.init(std.testing.allocator);
    try std.testing.expect(ctx.initialized);
    try std.testing.expect(isInitialized());
    ctx.deinit();
    try std.testing.expect(!ctx.initialized);
    try std.testing.expect(!isInitialized());
}

test "isEnabled returns true" {
    try std.testing.expect(isEnabled());
}

test "mesh module accessible" {
    // Verify mesh sub-module re-export compiles
    _ = mesh;
}
