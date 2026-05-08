//! Desktop Integration
//!
//! Provides native UI extensions and integrations for the host OS.

pub const macos_menu = @import("macos_menu.zig");
pub const types = @import("types.zig");

const std = @import("std");
const builtin = @import("builtin");

pub const DesktopError = types.DesktopError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,
    mac_menu: ?macos_menu.MacMenu = null,

    pub fn init(allocator: std.mem.Allocator) Context {
        var menu: ?macos_menu.MacMenu = null;
        if (builtin.os.tag == .macos) {
            menu = macos_menu.MacMenu.init(allocator);
        }
        return .{
            .allocator = allocator,
            .initialized = true,
            .mac_menu = menu,
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.mac_menu) |*menu| {
            menu.deinit();
            self.mac_menu = null;
        }
        self.initialized = false;
    }

    /// Spawns the native macOS Status Item (Menu Bar Icon).
    pub fn showMenu(self: *Context, title: [:0]const u8) !void {
        if (!self.initialized) return error.IntegrationFailed;
        if (self.mac_menu) |*menu| {
            menu.spawn(title) catch return error.IntegrationFailed;
        } else {
            return error.PlatformUnsupported;
        }
    }

    /// Spawns a transparent overlay.
    pub fn showOverlay(self: *Context) !void {
        if (!self.initialized) return error.IntegrationFailed;
        if (self.mac_menu) |*menu| {
            menu.spawnOverlay() catch return error.IntegrationFailed;
        } else {
            return error.PlatformUnsupported;
        }
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return true; // Module is available when feature flag is enabled
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
