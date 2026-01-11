//! Output formatting utilities for CLI commands.

const std = @import("std");

/// Convert boolean to "yes" or "no" label.
pub fn boolLabel(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

/// Print an optional u32 value, or "n/a" if null.
pub fn printOptionalU32(value: ?u32) void {
    if (value) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("n/a", .{});
    }
}

test "boolLabel helper function" {
    try std.testing.expectEqualStrings("yes", boolLabel(true));
    try std.testing.expectEqualStrings("no", boolLabel(false));
}
