//! Desktop stub — disabled at compile time.

const std = @import("std");

pub const macos_menu = struct {};

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
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
