//! TUI menu item definitions and lookup helpers.

const std = @import("std");
const types = @import("types.zig");
const launcher_catalog = @import("launcher_catalog.zig");

const MenuItem = types.MenuItem;

/// Return the full list of menu items with metadata.
pub fn menuItemsExtended() []const MenuItem {
    return launcher_catalog.menuItems();
}

/// Find a menu item by its shortcut number (1-9).
pub fn findByShortcut(items: []const MenuItem, num: u8) ?usize {
    for (items, 0..) |item, i| {
        if (item.shortcut) |s| {
            if (s == num) return i;
        }
    }
    return null;
}

/// Get the display name for a command id.
pub fn commandName(command_id: []const u8) []const u8 {
    return launcher_catalog.commandName(command_id);
}

test "training monitor launcher command ref" {
    const cmd = launcher_catalog.findCommandById("train-monitor") orelse return error.TestExpectedCommand;
    try std.testing.expectEqualStrings("train", cmd.command);
    try std.testing.expectEqual(@as(usize, 1), cmd.args.len);
    try std.testing.expectEqualStrings("monitor", cmd.args[0]);
}

test {
    std.testing.refAllDecls(@This());
}
