//! Desktop Integration
//!
//! Provides native UI extensions and integrations for the host OS.

pub const macos_menu = @import("macos_menu.zig");

const std = @import("std");

pub const DesktopError = error{
    PlatformUnsupported,
    IntegrationFailed,
    OutOfMemory,
};

pub const Error = DesktopError;

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

test "Context init and deinit" {
    var ctx = Context.init(std.testing.allocator);
    try std.testing.expect(ctx.initialized);
    ctx.deinit();
    try std.testing.expect(!ctx.initialized);
}

test "isEnabled returns true" {
    try std.testing.expect(isEnabled());
}
