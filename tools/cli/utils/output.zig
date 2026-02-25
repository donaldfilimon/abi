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
/// All accessors automatically respect NO_COLOR / --no-color.
pub const Color = struct {
    // Raw escape sequences (private — use the public functions below).
    const esc_reset = "\x1b[0m";
    const esc_red = "\x1b[31m";
    const esc_green = "\x1b[32m";
    const esc_yellow = "\x1b[33m";
    const esc_blue = "\x1b[34m";
    const esc_cyan = "\x1b[36m";
    const esc_bold = "\x1b[1m";
    const esc_dim = "\x1b[2m";

    pub fn reset() []const u8 {
        return gate(esc_reset);
    }
    pub fn red() []const u8 {
        return gate(esc_red);
    }
    pub fn green() []const u8 {
        return gate(esc_green);
    }
    pub fn yellow() []const u8 {
        return gate(esc_yellow);
    }
    pub fn blue() []const u8 {
        return gate(esc_blue);
    }
    pub fn cyan() []const u8 {
        return gate(esc_cyan);
    }
    pub fn bold() []const u8 {
        return gate(esc_bold);
    }
    pub fn dim() []const u8 {
        return gate(esc_dim);
    }

    /// Get a color code, respecting NO_COLOR. Kept for any external callers.
    pub fn get(code: []const u8) []const u8 {
        return gate(code);
    }

    fn gate(code: []const u8) []const u8 {
        return if (isColorEnabled()) code else "";
    }
};

/// Print an error message with red formatting.
pub fn printError(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}error:{s} ", .{ Color.red(), Color.reset() });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a warning message with yellow formatting.
pub fn printWarning(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}warning:{s} ", .{ Color.yellow(), Color.reset() });
    std.debug.print(fmt ++ "\n", args);
}

/// Print an info message with cyan formatting.
pub fn printInfo(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}info:{s} ", .{ Color.cyan(), Color.reset() });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a success message with green formatting.
pub fn printSuccess(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}success:{s} ", .{ Color.green(), Color.reset() });
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
        Color.bold(),
        Color.cyan(),
        title,
        Color.reset(),
    });
}

/// Print a formatted header without pre-allocating the title string.
pub fn printHeaderFmt(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}{s}=== ", .{ Color.bold(), Color.cyan() });
    std.debug.print(fmt, args);
    std.debug.print(" ==={s}\n", .{Color.reset()});
}

/// Print a key-value pair.
pub fn printKeyValue(key: []const u8, value: []const u8) void {
    std.debug.print("  {s}{s}:{s} {s}\n", .{
        Color.bold(),
        key,
        Color.reset(),
        value,
    });
}

/// Print a formatted key-value pair without pre-allocating the value string.
pub fn printKeyValueFmt(key: []const u8, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("  {s}{s}:{s} ", .{ Color.bold(), key, Color.reset() });
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}

/// Print a bullet list with a title.
pub fn printBulletList(title: []const u8, items: []const []const u8) void {
    std.debug.print("{s}{s}:{s}\n", .{ Color.bold(), title, Color.reset() });
    for (items) |item| {
        std.debug.print("  {s}*{s} {s}\n", .{ Color.cyan(), Color.reset(), item });
    }
}

/// Print a separator line of a given width (max 256 chars).
pub fn printSeparator(width: usize) void {
    const max_width = 256;
    const dashes: [max_width]u8 = .{'-'} ** max_width;
    const w = @min(width, max_width);
    std.debug.print("{s}\n", .{dashes[0..w]});
}

/// Print a status line with icon: [ok] or [--] with appropriate coloring.
pub fn printStatusLine(label: []const u8, enabled: bool) void {
    const icon_color = if (enabled) Color.green() else Color.dim();
    const marker = if (enabled) "[ok]" else "[--]";
    std.debug.print("  {s}{s}{s} {s}\n", .{ icon_color, marker, Color.reset(), label });
}

/// Print a status line with a formatted label (for enum/tag names).
pub fn printStatusLineFmt(comptime fmt: []const u8, args: anytype, enabled: bool) void {
    const icon_color = if (enabled) Color.green() else Color.dim();
    const marker = if (enabled) "[ok]" else "[--]";
    std.debug.print("  {s}{s}{s} ", .{ icon_color, marker, Color.reset() });
    std.debug.print(fmt ++ "\n", args);
}

/// Print a table with bold headers and auto-sized column widths.
/// Supports up to 16 columns; extra columns are silently truncated.
pub fn printTable(headers: []const []const u8, rows: []const []const []const u8) void {
    if (headers.len == 0) return;
    const max_cols: usize = @min(headers.len, 16);

    // Compute column widths (max of header and all cell lengths).
    var widths: [16]usize = .{0} ** 16;
    for (headers[0..max_cols], 0..) |h, c| widths[c] = h.len;
    for (rows) |row| {
        const n = @min(row.len, max_cols);
        for (row[0..n], 0..) |cell, c| {
            if (cell.len > widths[c]) widths[c] = cell.len;
        }
    }

    // Header row.
    for (headers[0..max_cols], 0..) |h, c| {
        std.debug.print("{s}{s}{s}", .{ Color.bold(), h, Color.reset() });
        const pad = widths[c] + 2 - h.len;
        for (0..pad) |_| std.debug.print(" ", .{});
    }
    std.debug.print("\n", .{});

    // Separator.
    for (0..max_cols) |c| {
        for (0..widths[c]) |_| std.debug.print("-", .{});
        std.debug.print("  ", .{});
    }
    std.debug.print("\n", .{});

    // Data rows.
    for (rows) |row| {
        const n = @min(row.len, max_cols);
        for (row[0..n], 0..) |cell, c| {
            std.debug.print("{s}", .{cell});
            const pad = widths[c] + 2 - cell.len;
            for (0..pad) |_| std.debug.print(" ", .{});
        }
        std.debug.print("\n", .{});
    }
}

/// Print a progress bar: `[=====>    ] 55% label`
/// Uses cyan while in-progress, green at 100%.
pub fn printProgress(label: []const u8, current: usize, total: usize) void {
    const pct: usize = if (total > 0) (current * 100) / total else 0;
    const filled: usize = (pct * 40) / 100;
    const bar_color = if (pct >= 100) Color.green() else Color.cyan();
    std.debug.print("{s}[", .{bar_color});
    var i: usize = 0;
    while (i < 40) : (i += 1) {
        if (i < filled) {
            std.debug.print("=", .{});
        } else if (i == filled and filled < 40) {
            std.debug.print(">", .{});
        } else {
            std.debug.print(" ", .{});
        }
    }
    std.debug.print("]{s} {d}% {s}\n", .{ Color.reset(), pct, label });
}

/// Print a count summary (e.g., "12/24 features active").
pub fn printCountSummary(count: usize, total: usize, label: []const u8) void {
    std.debug.print("\n  {d}/{d} {s}\n", .{ count, total, label });
}

/// Re-export Color for direct access (e.g., utils.output.color.green()).
pub const color = Color;

test "boolLabel helper function" {
    try std.testing.expectEqualStrings("yes", boolLabel(true));
    try std.testing.expectEqualStrings("no", boolLabel(false));
}

test "Color functions return escape codes when enabled" {
    // Force color on (override any NO_COLOR in test env).
    enableColor();
    color_initialized = true;

    try std.testing.expectEqualStrings("\x1b[31m", Color.red());
    try std.testing.expectEqualStrings("\x1b[32m", Color.green());
    try std.testing.expectEqualStrings("\x1b[33m", Color.yellow());
    try std.testing.expectEqualStrings("\x1b[34m", Color.blue());
    try std.testing.expectEqualStrings("\x1b[36m", Color.cyan());
    try std.testing.expectEqualStrings("\x1b[1m", Color.bold());
    try std.testing.expectEqualStrings("\x1b[2m", Color.dim());
    try std.testing.expectEqualStrings("\x1b[0m", Color.reset());
}

test "Color functions return empty strings when disabled" {
    disableColor();
    color_initialized = true;

    try std.testing.expectEqualStrings("", Color.red());
    try std.testing.expectEqualStrings("", Color.green());
    try std.testing.expectEqualStrings("", Color.yellow());
    try std.testing.expectEqualStrings("", Color.blue());
    try std.testing.expectEqualStrings("", Color.cyan());
    try std.testing.expectEqualStrings("", Color.bold());
    try std.testing.expectEqualStrings("", Color.dim());
    try std.testing.expectEqualStrings("", Color.reset());

    // Restore for other tests.
    enableColor();
}

test "Color.get respects color state" {
    enableColor();
    color_initialized = true;
    try std.testing.expectEqualStrings("\x1b[31m", Color.get("\x1b[31m"));

    disableColor();
    try std.testing.expectEqualStrings("", Color.get("\x1b[31m"));

    enableColor();
}

test "disableColor and enableColor toggle state" {
    color_initialized = true;
    enableColor();
    try std.testing.expect(isColorEnabled());

    disableColor();
    try std.testing.expect(!isColorEnabled());

    enableColor();
    try std.testing.expect(isColorEnabled());
}

test "printSeparator smoke test at boundary widths" {
    // These should not panic or crash. We can't capture stderr output
    // but we verify the buffer logic handles edge cases.
    printSeparator(0);
    printSeparator(1);
    printSeparator(80);
    printSeparator(256);
    printSeparator(300); // clamped to 256
}

test "printTable smoke test" {
    // Empty — should not crash.
    printTable(&.{}, &.{});
    // Single column.
    printTable(&.{"Name"}, &.{ &.{"alpha"}, &.{"beta"} });
    // Multiple columns with ragged rows.
    printTable(
        &.{ "Name", "Size", "Type" },
        &.{
            &.{ "model-a", "1.2 GB", "GGUF" },
            &.{ "model-b", "3.4 GB", "GGUF" },
            &.{"short-row"},
        },
    );
}

test "printProgress smoke test" {
    printProgress("Loading", 0, 100);
    printProgress("Loading", 50, 100);
    printProgress("Loading", 100, 100);
    printProgress("Loading", 0, 0); // zero total — should not divide by zero
    printProgress("Loading", 150, 100); // over 100% — clamped
}

test "print functions do not crash (smoke)" {
    // Smoke-test all print variants — verifies format strings are valid
    // and functions don't panic. Output goes to stderr (not captured).
    printError("test error {s}", .{"msg"});
    printWarning("test warning {d}", .{42});
    printInfo("test info {s}", .{"data"});
    printSuccess("test success {s}", .{"ok"});
    print("plain {s}", .{"text"});
    println("line {d}", .{1});
    printOptionalU32(42);
    printOptionalU32(null);
    printHeader("Test");
    printHeaderFmt("Header {d}", .{1});
    printKeyValue("key", "value");
    printKeyValueFmt("key", "{d}", .{99});
    printBulletList("list", &.{ "a", "b" });
    printStatusLine("feature", true);
    printStatusLine("feature", false);
    printStatusLineFmt("{s}", .{"feature"}, true);
    printStatusLineFmt("{s}", .{"feature"}, false);
    printCountSummary(5, 10, "items");
}

test {
    std.testing.refAllDecls(@This());
}
