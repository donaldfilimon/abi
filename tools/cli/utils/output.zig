//! Output formatting utilities for CLI commands.

const std = @import("std");

/// ANSI color codes for terminal output.
pub const Color = struct {
    pub const reset = "\x1b[0m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const cyan = "\x1b[36m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
};

/// Print an error message with red formatting.
pub fn printError(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(Color.red ++ "error: " ++ Color.reset ++ fmt ++ "\n", args);
}

/// Print a warning message with yellow formatting.
pub fn printWarning(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(Color.yellow ++ "warning: " ++ Color.reset ++ fmt ++ "\n", args);
}

/// Print an info message with cyan formatting.
pub fn printInfo(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(Color.cyan ++ "info: " ++ Color.reset ++ fmt ++ "\n", args);
}

/// Print a success message with green formatting.
pub fn printSuccess(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(Color.green ++ "success: " ++ Color.reset ++ fmt ++ "\n", args);
}

/// Print a plain message.
pub fn print(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

/// Print a plain message with newline.
pub fn println(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt ++ "\n", args);
}

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

/// Print a header with decorative formatting.
pub fn printHeader(title: []const u8) void {
    std.debug.print("\n" ++ Color.bold ++ Color.cyan ++ "=== " ++ "{s}" ++ " ===" ++ Color.reset ++ "\n", .{title});
}

/// Print a key-value pair.
pub fn printKeyValue(key: []const u8, value: []const u8) void {
    std.debug.print("  " ++ Color.bold ++ "{s}:" ++ Color.reset ++ " {s}\n", .{ key, value });
}

/// Print a bullet list with a title.
pub fn printBulletList(title: []const u8, items: []const []const u8) void {
    std.debug.print(Color.bold ++ "{s}:" ++ Color.reset ++ "\n", .{title});
    for (items) |item| {
        std.debug.print("  " ++ Color.cyan ++ "*" ++ Color.reset ++ " {s}\n", .{item});
    }
}

/// Re-export color constants for direct access (e.g., utils.output.color.green).
pub const color = Color;

test "boolLabel helper function" {
    try std.testing.expectEqualStrings("yes", boolLabel(true));
    try std.testing.expectEqualStrings("no", boolLabel(false));
}
