//! `/pane` split view for the interactive REPL: chat/completion output in a
//! left column, `git diff --stat` (or the `/open` file-context summary when
//! the working tree is clean) in a right column.
//!
//! This is an honest terminal split view — one composed text block printed
//! after each completed turn — not a windowing system, and not a live
//! token-by-token redraw (still out of scope). Layout reuses the split
//! machinery from `dashboard_render.zig` (`AGENT_PANE_WIDTH`, `SPLIT_SEP`,
//! and the ANSI/UTF-8 fit helpers) rather than inventing a new engine.
//!
//! Pure composition (`paneWidths`, `paneToggleDecision`, `buildRightPane`,
//! `composeSplit`) is separated from the IO glue (`terminalCols`,
//! `collectDiffStat`, `renderPaneView`) so the layout math is unit-testable.

const std = @import("std");
const builtin = @import("builtin");
const dashboard_render = @import("dashboard_render.zig");
const repl_types = @import("repl_types.zig");
const sanitize = @import("sanitize.zig");
const cmds = @import("repl_commands.zig");

/// Minimum terminal width for the split view. Below this, `/pane` prints a
/// one-line notice and the REPL stays unsplit.
pub const MIN_SPLIT_COLS: usize = 80;

/// Right (diff/file) column width — same as the dashboard's agent pane.
pub const RIGHT_PANE_COLS: usize = dashboard_render.AGENT_PANE_WIDTH;

pub const PaneWidths = struct {
    left: usize,
    right: usize,
};

/// Compute pane column widths for a terminal of `cols` visible columns.
/// The right pane is fixed at `RIGHT_PANE_COLS`; the left pane takes the
/// remainder after the dashboard `SPLIT_SEP` gap. `cols` below
/// `MIN_SPLIT_COLS` is clamped up so the result is always well-formed.
pub fn paneWidths(cols: usize) PaneWidths {
    const total = @max(cols, MIN_SPLIT_COLS);
    return .{
        .left = total - RIGHT_PANE_COLS - dashboard_render.SPLIT_SEP.len,
        .right = RIGHT_PANE_COLS,
    };
}

pub const ToggleDecision = struct {
    enabled: bool,
    /// Terminal width captured at toggle time (0 when disabled). The split
    /// uses this snapshot; a later resize takes effect on the next `/pane`.
    cols: usize = 0,
    notice: []const u8,
};

/// Decide the `/pane` toggle outcome from the current mode and the measured
/// terminal width (`null` when the width cannot be determined, e.g. non-tty).
/// Pure so the narrow/unknown/enable branches are unit-testable.
pub fn paneToggleDecision(currently_on: bool, cols: ?usize) ToggleDecision {
    if (currently_on) {
        return .{ .enabled = false, .notice = "pane: split view off" };
    }
    const measured = cols orelse {
        return .{ .enabled = false, .notice = "pane: terminal width unknown (not a tty?); staying unsplit" };
    };
    if (measured < MIN_SPLIT_COLS) {
        return .{ .enabled = false, .notice = "pane: terminal too narrow (< 80 cols); staying unsplit" };
    }
    return .{
        .enabled = true,
        .cols = measured,
        .notice = "pane: split view on (chat left, git diff --stat right)",
    };
}

/// Build the right-pane text: `git diff --stat` output when non-empty,
/// otherwise the `/open` file-context summary, otherwise a placeholder.
/// Pure: takes the already-collected diff text. Caller owns the result.
pub fn buildRightPane(
    allocator: std.mem.Allocator,
    diff_stat: []const u8,
    open_path: []const u8,
    open_content_len: usize,
) ![]u8 {
    const trimmed = std.mem.trim(u8, diff_stat, " \t\r\n");
    if (trimmed.len > 0) {
        return std.fmt.allocPrint(allocator, "[diff --stat]\n{s}", .{trimmed});
    }
    if (open_path.len > 0) {
        return std.fmt.allocPrint(allocator, "[file: {s}]\n({d} bytes in context)", .{ open_path, open_content_len });
    }
    return allocator.dupe(u8, "(working tree clean; no open file)");
}

/// Compose the split block: left and right texts interleaved line-by-line,
/// each side padded/truncated to its pane width with the dashboard's
/// visible-column fit helper (ANSI/UTF-8 safe), separated by `SPLIT_SEP`.
/// Pure: no IO. Caller owns the returned slice.
pub fn composeSplit(
    allocator: std.mem.Allocator,
    left_text: []const u8,
    right_text: []const u8,
    widths: PaneWidths,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var left_iter = std.mem.splitScalar(u8, std.mem.trimEnd(u8, left_text, "\n"), '\n');
    var right_iter = std.mem.splitScalar(u8, std.mem.trimEnd(u8, right_text, "\n"), '\n');

    var left_done = false;
    var right_done = false;
    while (true) {
        const left_line = left_iter.next() orelse blk: {
            left_done = true;
            break :blk "";
        };
        const right_line = right_iter.next() orelse blk: {
            right_done = true;
            break :blk "";
        };
        if (left_done and right_done) break;

        try dashboard_render.appendFittedVisible(&out, allocator, left_line, widths.left);
        try out.appendSlice(allocator, dashboard_render.SPLIT_SEP);
        try dashboard_render.appendFittedVisible(&out, allocator, right_line, widths.right);
        try out.append(allocator, '\n');
    }

    return out.toOwnedSlice(allocator);
}

/// Measure the terminal width in columns via TIOCGWINSZ on the stdout device.
/// Returns null when the width cannot be determined (non-tty, Windows, WASI).
pub fn terminalCols(io: std.Io) ?usize {
    switch (builtin.os.tag) {
        .windows, .wasi => return null,
        else => {},
    }
    var ws: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
    const result = io.operate(.{ .device_io_control = .{
        .file = std.Io.File.stdout(),
        .code = std.posix.T.IOCGWINSZ,
        .arg = &ws,
    } }) catch return null;
    if (result.device_io_control < 0) return null;
    if (ws.col == 0) return null;
    return ws.col;
}

/// Run `git diff --stat` and return its stdout (possibly empty on a clean
/// tree). Caller owns the result. Spawn failures propagate so the caller can
/// report them (never silently swallowed).
pub fn collectDiffStat(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    var child = try std.process.spawn(io, .{
        .argv = cmds.diffArgv(true),
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    });
    defer child.kill(io);

    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.Io.File.readStreaming(child.stdout.?, io, &.{&buf}) catch break;
        if (n == 0) break;
        try output.appendSlice(allocator, buf[0..n]);
    }
    _ = try child.wait(io);
    return output.toOwnedSlice(allocator);
}

/// Print one split block for a completed turn: sanitized chat output on the
/// left, fresh `git diff --stat` (or the open-file summary) on the right.
pub fn renderPaneView(
    allocator: std.mem.Allocator,
    state: *const repl_types.ReplState,
    chat_text: []const u8,
    io: std.Io,
) !void {
    const widths = paneWidths(state.pane_cols);

    const diff_stat = collectDiffStat(allocator, io) catch |err| blk: {
        std.debug.print("pane: git diff unavailable: {s}\n", .{@errorName(err)});
        break :blk try allocator.dupe(u8, "");
    };
    defer allocator.free(diff_stat);

    const right = try buildRightPane(allocator, diff_stat, state.open_path, state.open_content.len);
    defer allocator.free(right);

    // The chat text is model output routed around the streaming sanitizer in
    // pane mode, so neutralize control bytes here before it reaches the terminal.
    const safe_chat = try sanitize.sanitizeControlBytes(allocator, chat_text);
    defer allocator.free(safe_chat);
    const left = try std.fmt.allocPrint(allocator, "[chat]\n{s}", .{safe_chat});
    defer allocator.free(left);

    const composed = try composeSplit(allocator, left, right, widths);
    defer allocator.free(composed);
    std.debug.print("{s}", .{composed});
}

// ── Tests ─────────────────────────────────────────────────────────────────

test "paneWidths fixes the right pane and gives the left the remainder" {
    const w = paneWidths(120);
    try std.testing.expectEqual(RIGHT_PANE_COLS, w.right);
    try std.testing.expectEqual(@as(usize, 120 - RIGHT_PANE_COLS - dashboard_render.SPLIT_SEP.len), w.left);
}

test "paneWidths clamps below the minimum so layout stays well-formed" {
    const w = paneWidths(10);
    try std.testing.expectEqual(@as(usize, MIN_SPLIT_COLS - RIGHT_PANE_COLS - dashboard_render.SPLIT_SEP.len), w.left);
    try std.testing.expectEqual(RIGHT_PANE_COLS, w.right);
}

test "paneToggleDecision turns off when currently on regardless of width" {
    const d = paneToggleDecision(true, 200);
    try std.testing.expect(!d.enabled);
    try std.testing.expect(std.mem.indexOf(u8, d.notice, "off") != null);
}

test "paneToggleDecision stays unsplit when width is unknown" {
    const d = paneToggleDecision(false, null);
    try std.testing.expect(!d.enabled);
    try std.testing.expect(std.mem.indexOf(u8, d.notice, "unknown") != null);
}

test "paneToggleDecision stays unsplit below 80 columns" {
    const d = paneToggleDecision(false, MIN_SPLIT_COLS - 1);
    try std.testing.expect(!d.enabled);
    try std.testing.expect(std.mem.indexOf(u8, d.notice, "narrow") != null);
}

test "paneToggleDecision enables at 80 columns and captures the width" {
    const d = paneToggleDecision(false, MIN_SPLIT_COLS);
    try std.testing.expect(d.enabled);
    try std.testing.expectEqual(MIN_SPLIT_COLS, d.cols);
}

test "buildRightPane prefers the diff stat when present" {
    const right = try buildRightPane(std.testing.allocator, " 1 file changed\n", "open.zig", 42);
    defer std.testing.allocator.free(right);
    try std.testing.expect(std.mem.indexOf(u8, right, "[diff --stat]") != null);
    try std.testing.expect(std.mem.indexOf(u8, right, "1 file changed") != null);
    try std.testing.expect(std.mem.indexOf(u8, right, "open.zig") == null);
}

test "buildRightPane falls back to the open-file summary on a clean tree" {
    const right = try buildRightPane(std.testing.allocator, "  \n", "src/main.zig", 128);
    defer std.testing.allocator.free(right);
    try std.testing.expect(std.mem.indexOf(u8, right, "[file: src/main.zig]") != null);
    try std.testing.expect(std.mem.indexOf(u8, right, "128 bytes") != null);
}

test "buildRightPane reports a placeholder when there is nothing to show" {
    const right = try buildRightPane(std.testing.allocator, "", "", 0);
    defer std.testing.allocator.free(right);
    try std.testing.expectEqualStrings("(working tree clean; no open file)", right);
}

test "composeSplit interleaves lines with padded visible columns" {
    const widths = PaneWidths{ .left = 10, .right = 8 };
    const composed = try composeSplit(std.testing.allocator, "aa\nbb", "rr", widths);
    defer std.testing.allocator.free(composed);

    var rows: usize = 0;
    var iter = std.mem.splitScalar(u8, std.mem.trimEnd(u8, composed, "\n"), '\n');
    while (iter.next()) |row| {
        rows += 1;
        try std.testing.expectEqual(
            widths.left + dashboard_render.SPLIT_SEP.len + widths.right,
            dashboard_render.ansiVisibleWidth(row),
        );
    }
    try std.testing.expectEqual(@as(usize, 2), rows);
    try std.testing.expect(std.mem.indexOf(u8, composed, "aa") != null);
    try std.testing.expect(std.mem.indexOf(u8, composed, "rr") != null);
    try std.testing.expect(std.mem.indexOf(u8, composed, "bb") != null);
}

test "composeSplit pads the shorter side to the longer side's row count" {
    const widths = PaneWidths{ .left = 6, .right = 6 };
    const composed = try composeSplit(std.testing.allocator, "one", "r1\nr2\nr3", widths);
    defer std.testing.allocator.free(composed);
    try std.testing.expectEqual(@as(usize, 3), dashboard_render.countLines(composed));
}

test "composeSplit truncates overlong lines without splitting ANSI escapes" {
    const widths = PaneWidths{ .left = 5, .right = 5 };
    const composed = try composeSplit(std.testing.allocator, "\x1b[32mgreen text overflowing\x1b[0m", "r", widths);
    defer std.testing.allocator.free(composed);
    var iter = std.mem.splitScalar(u8, std.mem.trimEnd(u8, composed, "\n"), '\n');
    const row = iter.next().?;
    try std.testing.expectEqual(widths.left + dashboard_render.SPLIT_SEP.len + widths.right, dashboard_render.ansiVisibleWidth(row));
    try std.testing.expect(std.mem.indexOf(u8, row, "\x1b[32m") != null);
}

test {
    std.testing.refAllDecls(@This());
}
