//! Cross-platform OS Control and Permissions Layer
//!
//! Exposes tools for the AI agent to interact with the host OS,
//! simulating keyboard/mouse events, and reading screen contents safely.

const std = @import("std");

/// The permission level granted to the OS agent.
pub const PermissionLevel = enum {
    read_only,
    ask_before_action,
    full_control,
};

pub const OSControlManager = struct {
    allocator: std.mem.Allocator,
    permission_level: PermissionLevel,

    pub fn init(allocator: std.mem.Allocator, permission_level: PermissionLevel) OSControlManager {
        return .{
            .allocator = allocator,
            .permission_level = permission_level,
        };
    }

    pub fn deinit(self: *OSControlManager) void {
        _ = self;
    }

    /// Ask user for permission via CLI/TUI hook.
    fn checkPermission(self: *OSControlManager, action_desc: []const u8) !bool {
        switch (self.permission_level) {
            .read_only => return false,
            .full_control => return true,
            .ask_before_action => {
                // In a real CLI, this hooks into the confirmation prompt.
                // TODO: wire into actual CLI/TUI confirmation dialog.
                std.log.info("[Security] Agent wants to: {s}. Allow? (y/N)", .{action_desc});
                // Stubbing permission to true for this prototype
                return true;
            },
        }
    }

    /// Take a screenshot.
    pub fn captureScreen(self: *OSControlManager) ![]const u8 {
        // Stub implementation
        return try self.allocator.dupe(u8, "[binary_image_data_stub]");
    }

    /// Simulate a keypress.
    pub fn typeKeys(self: *OSControlManager, keys: []const u8) !void {
        if (!try self.checkPermission(keys)) return error.PermissionDenied;
        // Stub implementation
        std.log.info("Agent typed: {s}", .{keys});
    }

    /// Simulate a mouse click.
    pub fn clickMouse(self: *OSControlManager, x: u32, y: u32) !void {
        var buf: [64]u8 = [_]u8{0} ** 64;
        const msg = try std.fmt.bufPrint(&buf, "Click mouse at ({d}, {d})", .{ x, y });
        if (!try self.checkPermission(msg)) return error.PermissionDenied;
        // Stub implementation
        std.log.info("Agent clicked at ({d}, {d})", .{ x, y });
    }
};

test {
    std.testing.refAllDecls(@This());
}
