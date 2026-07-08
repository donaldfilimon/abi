const std = @import("std");
const builtin = @import("builtin");
const sanitize = @import("sanitize.zig");
const terminal = @import("terminal.zig");
const types = @import("types.zig");

pub const repl = @import("repl.zig");
pub const ReplLoop = repl.ReplLoop;
pub const ReplState = repl.ReplState;
pub const ReplConfig = repl.ReplConfig;

pub const Status = types.Status;
pub const Item = types.Item;
pub const State = types.State;
pub const ScreenState = types.ScreenState;
pub const PaneKind = types.PaneKind;
pub const DiagPane = types.DiagPane;
pub const DashboardPaneMeta = types.DashboardPaneMeta;
pub const DashboardState = types.DashboardState;
pub const DASHBOARD_PANES = types.DASHBOARD_PANES;
pub const DASHBOARD_PANE_COUNT = types.DASHBOARD_PANE_COUNT;

pub const sanitizeControlBytes = sanitize.sanitizeControlBytes;

pub const InteractiveTerminal = terminal.InteractiveTerminal;
pub const ScreenSession = terminal.ScreenSession;
pub const stdinFd = terminal.stdinFd;
pub const isQuitKey = terminal.isQuitKey;
pub const isRefreshKey = terminal.isRefreshKey;
pub const initScreen = terminal.initScreen;
pub const initScreenWriter = terminal.initScreenWriter;
pub const clearScreen = terminal.clearScreen;
pub const homeScreen = terminal.homeScreen;
pub const homeScreenWriter = terminal.homeScreenWriter;
pub const clearToEnd = terminal.clearToEnd;
pub const clearToEndWriter = terminal.clearToEndWriter;
pub const clearScreenWriter = terminal.clearScreenWriter;
pub const render = terminal.render;
pub const renderWriter = terminal.renderWriter;
pub const deinitScreen = terminal.deinitScreen;
pub const deinitScreenWriter = terminal.deinitScreenWriter;

const dashboard = @import("dashboard.zig");
pub const DiagnosticRenderOptions = dashboard.DiagnosticRenderOptions;
pub const statusText = dashboard.statusText;
pub const dashboardPaneIndexForKey = dashboard.dashboardPaneIndexForKey;
pub const dashboardPaneName = dashboard.dashboardPaneName;
pub const dashboardPaneIndexForToken = dashboard.dashboardPaneIndexForToken;
pub const nextDashboardPane = dashboard.nextDashboardPane;
pub const renderDashboard = dashboard.renderDashboard;
pub const renderDiagnostics = dashboard.renderDiagnostics;
pub const renderDiagnosticsWithOptions = dashboard.renderDiagnosticsWithOptions;
pub const writeDashboard = dashboard.writeDashboard;
pub const writeDiagnostics = dashboard.writeDiagnostics;

test "tui sanitizeControlBytes neutralizes C1 controls and preserves UTF-8" {
    const allocator = std.testing.allocator;

    // Lone 0x9B (raw CSI on a non-UTF-8 terminal): not a valid UTF-8 start byte,
    // replaced 1:1 with '.'. No 0x9B survives; length preserved.
    {
        const clean = try sanitizeControlBytes(allocator, "x\x9by");
        defer allocator.free(clean);
        try std.testing.expectEqualStrings("x.y", clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x9b) == null);
        try std.testing.expectEqual(@as(usize, 3), clean.len);
    }

    // Encoded C1 "\xc2\x9b" decodes to U+009B (CSI): both bytes collapse to "..".
    // Neither the 0x9b payload nor the 0xc2 lead byte survives; length preserved.
    {
        const clean = try sanitizeControlBytes(allocator, "a\xc2\x9bb");
        defer allocator.free(clean);
        try std.testing.expectEqualStrings("a..b", clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x9b) == null);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0xc2) == null);
        try std.testing.expectEqual(@as(usize, 4), clean.len);
    }

    // Valid multibyte UTF-8 (accented "aéb" and a box-drawing "─") is preserved
    // byte-for-byte.
    {
        const accented = "a\xc3\xa9b"; // a é b
        const clean = try sanitizeControlBytes(allocator, accented);
        defer allocator.free(clean);
        try std.testing.expectEqualStrings(accented, clean);

        const box = "\xe2\x94\x80"; // ─ (U+2500)
        const clean_box = try sanitizeControlBytes(allocator, box);
        defer allocator.free(clean_box);
        try std.testing.expectEqualStrings(box, clean_box);
    }

    // ESC (0x1b) and NUL (0x00) are still stripped; output length == input length.
    {
        const dirty = "\x1b[2J\x00ok";
        const clean = try sanitizeControlBytes(allocator, dirty);
        defer allocator.free(clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x1b) == null);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x00) == null);
        try std.testing.expectEqual(dirty.len, clean.len);
    }
}

test "writer render functions are testable" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }

        pub fn print(self: *@This(), comptime fmt: []const u8, args: anytype) !void {
            try self.buffer.print(self.allocator, fmt, args);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };

    try renderWriter(&writer, .{ .width = 80, .height = 24 });
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "80x24") != null);
}

test {
    std.testing.refAllDecls(@This());
}

test "InteractiveTerminal struct layout" {
    const term = if (builtin.os.tag == .windows) InteractiveTerminal{
        .fd = 0,
        .is_tty = false,
    } else InteractiveTerminal{
        .fd = 0,
        .original = undefined,
        .is_tty = false,
    };
    try std.testing.expect(!term.is_tty);
    try std.testing.expectEqual(@as(std.posix.fd_t, 0), term.fd);
}

test "quit and refresh key detection" {
    try std.testing.expect(isQuitKey('q'));
    try std.testing.expect(isQuitKey('Q'));
    try std.testing.expect(isQuitKey(0x1b));
    try std.testing.expect(!isQuitKey('r'));
    try std.testing.expect(isRefreshKey('r'));
    try std.testing.expect(isRefreshKey('R'));
    try std.testing.expect(!isRefreshKey('q'));
}
