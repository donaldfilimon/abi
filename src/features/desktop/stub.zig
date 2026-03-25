//! Desktop stub — disabled at compile time.

const std = @import("std");
const stub_helpers = @import("../../core/stub_helpers.zig");
pub const types = @import("types.zig");

pub const macos_menu = struct {};

pub const DesktopError = types.DesktopError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

const _stub = stub_helpers.StubFeatureNoConfig(DesktopError);
pub const isEnabled = _stub.isEnabled;
pub const isInitialized = _stub.isInitialized;

test {
    std.testing.refAllDecls(@This());
}
