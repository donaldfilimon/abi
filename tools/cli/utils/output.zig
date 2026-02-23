//! Output formatting utilities for CLI commands.
//!
//! Respects the NO_COLOR convention (https://no-color.org/) and the --no-color flag.
//! When color is disabled, all color constants become empty strings.

const std = @import("std");

/// Whether color output is enabled. Defaults to true, disabled by
/// the NO_COLOR environment variable or the --no-color flag.
/// Checked lazily on first access since std.c.getenv is runtime-only.
var color_enabled: bool = true;
var color_initialized: bool = false;

/// Programmatically disable color output (e.g., from --no-color flag).
pub fn disableColor() void {
    color_enabled = false;
}

/// Programmatically enable color output.
pub fn enableColor() void {
    color_enabled = true;
}

/// Returns whether color output is currently enabled.
/// On first call, checks the NO_COLOR environment variable.
pub fn isColorEnabled() bool {
    if (!color_initialized) {
        color_initialized = true;
        if (std.c.getenv("NO_COLOR")) |_| {
            color_enabled = false;
        }
    }
    return color_enabled;
}

/// ANSI color codes for terminal output.
/// When NO_COLOR is set or --no-color is passed, all codes return empty strings.
pub const Color = struct {
    pub const reset = "\x1b[0m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const cyan = "\x1b[36m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";

    /// Get a color code, respecting NO_COLOR.
    pub fn get(code: []const u8) []const u8 {
        return if (isColorEnabled()) code else "";
    }
};

/// Print an error message with red formatting.
pub fn printError(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}error:{s} ", .{ Color.get(Color.red), Color.get(Color.reset) });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a warning message with yellow formatting.
pub fn printWarning(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}warning:{s} ", .{ Color.get(Color.yellow), Color.get(Color.reset) });
    std.debug.print(fmt ++ "\n", args);
}

/// Print an info message with cyan formatting.
pub fn printInfo(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}info:{s} ", .{ Color.get(Color.cyan), Color.get(Color.reset) });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a success message with green formatting.
pub fn printSuccess(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}success:{s} ", .{ Color.get(Color.green), Color.get(Color.reset) });
    std.debug.print(fmt ++ "\n", args);
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
    std.debug.print("\n{s}{s}=== {s} ==={s}\n", .{
        Color.get(Color.bold),
        Color.get(Color.cyan),
        title,
        Color.get(Color.reset),
    });
}

/// Print a formatted header without pre-allocating the title string.
pub fn printHeaderFmt(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}{s}=== ", .{ Color.get(Color.bold), Color.get(Color.cyan) });
    std.debug.print(fmt, args);
    std.debug.print(" ==={s}\n", .{Color.get(Color.reset)});
}

/// Print a key-value pair.
pub fn printKeyValue(key: []const u8, value: []const u8) void {
    std.debug.print("  {s}{s}:{s} {s}\n", .{
        Color.get(Color.bold),
        key,
        Color.get(Color.reset),
        value,
    });
}

/// Print a formatted key-value pair without pre-allocating the value string.
pub fn printKeyValueFmt(key: []const u8, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("  {s}{s}:{s} ", .{ Color.get(Color.bold), key, Color.get(Color.reset) });
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}

/// Print a bullet list with a title.
pub fn printBulletList(title: []const u8, items: []const []const u8) void {
    std.debug.print("{s}{s}:{s}\n", .{ Color.get(Color.bold), title, Color.get(Color.reset) });
    for (items) |item| {
        std.debug.print("  {s}*{s} {s}\n", .{ Color.get(Color.cyan), Color.get(Color.reset), item });
    }
}

/// Print a separator line of a given width.
pub fn printSeparator(width: usize) void {
    var i: usize = 0;
    while (i < width) : (i += 1) {
        std.debug.print("-", .{});
    }
    std.debug.print("\n", .{});
}

/// Print a status line with icon: [ok] or [--] with appropriate coloring.
pub fn printStatusLine(label: []const u8, enabled: bool) void {
    const icon_color = if (enabled) Color.get(Color.green) else Color.get(Color.dim);
    const marker = if (enabled) "[ok]" else "[--]";
    const reset = Color.get(Color.reset);
    std.debug.print("  {s}{s}{s} {s}\n", .{ icon_color, marker, reset, label });
}

/// Print a status line with a formatted label (for enum/tag names).
pub fn printStatusLineFmt(comptime fmt: []const u8, args: anytype, enabled: bool) void {
    const icon_color = if (enabled) Color.get(Color.green) else Color.get(Color.dim);
    const marker = if (enabled) "[ok]" else "[--]";
    const reset = Color.get(Color.reset);
    std.debug.print("  {s}{s}{s} ", .{ icon_color, marker, reset });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a count summary (e.g., "12/24 features active").
pub fn printCountSummary(count: usize, total: usize, label: []const u8) void {
    std.debug.print("\n  {d}/{d} {s}\n", .{ count, total, label });
}

/// Re-export color constants for direct access (e.g., utils.output.color.green).
pub const color = Color;

test "boolLabel helper function" {
    try std.testing.expectEqualStrings("yes", boolLabel(true));
    try std.testing.expectEqualStrings("no", boolLabel(false));
}
