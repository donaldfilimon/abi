const std = @import("std");
const types = @import("types.zig");
const widgets = @import("dashboard_widgets.zig");
const panes = @import("dashboard_panes.zig");

const appendFitted = widgets.appendFitted;
const appendRule = widgets.appendRule;
const appendRepeated = widgets.appendRepeated;
const appendBorder = widgets.appendBorder;
const appendRow = widgets.appendRow;
const appendPanelHeader = widgets.appendPanelHeader;
const appendPanelFooter = widgets.appendPanelFooter;
const DIAG_WIDTH = widgets.DIAG_WIDTH;

const appendPaneBody = panes.appendPaneBody;
const paneColor = panes.paneColor;

const DASHBOARD_PANES = types.DASHBOARD_PANES;
const DASHBOARD_PANE_COUNT = types.DASHBOARD_PANE_COUNT;

/// Visible columns for the left diagnostics column (border corners + DIAG_WIDTH).
pub const LEFT_PANE_COLS: usize = DIAG_WIDTH + 2;
/// Visible columns for the right agent-output column.
pub const AGENT_PANE_WIDTH: usize = 40;
/// Horizontal gap between the diagnostics column and the agent-status column.
/// Boxes already carry their own `│` borders; a plain gap avoids a triple-pipe seam.
pub const SPLIT_SEP: []const u8 = "  ";

pub const DiagnosticRenderOptions = struct {
    color: bool = true,
    refresh_interval_ms: u64 = 1000,
    compact: bool = false,
};

fn boolText(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

/// Operational health band for the dashboard header / JSON snapshot.
/// `cpu` is the normal Metal-linked CPU-SIMD path (accelerated=false is common);
/// `degraded` is reserved for future explicit failure signals — not "GPU idle".
pub fn dashboardHealth(ds: types.DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (ds.gpu_accelerated and ds.gpu_linked) return "nominal";
    return "cpu";
}

/// Visible terminal columns in `s`, treating CSI/ANSI sequences as width 0 and
/// each UTF-8 codepoint as width 1 (sufficient for ASCII + box-drawing).
pub fn ansiVisibleWidth(s: []const u8) usize {
    var width: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == 0x1b) {
            i += 1;
            if (i < s.len and s[i] == '[') {
                i += 1;
                while (i < s.len and (s[i] < 0x40 or s[i] > 0x7e)) : (i += 1) {}
                if (i < s.len) i += 1;
            }
            continue;
        }
        const seq_len = std.unicode.utf8ByteSequenceLength(s[i]) catch {
            i += 1;
            width += 1;
            continue;
        };
        if (i + seq_len > s.len) break;
        i += seq_len;
        width += 1;
    }
    return width;
}

/// Append `raw` padded or truncated to exactly `cols` visible columns without
/// slicing mid-UTF-8 sequence or mid-CSI escape.
pub fn appendFittedVisible(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, raw: []const u8, cols: usize) !void {
    var width: usize = 0;
    var i: usize = 0;
    while (i < raw.len and width < cols) {
        if (raw[i] == 0x1b) {
            const esc_start = i;
            i += 1;
            if (i < raw.len and raw[i] == '[') {
                i += 1;
                while (i < raw.len and (raw[i] < 0x40 or raw[i] > 0x7e)) : (i += 1) {}
                if (i < raw.len) i += 1;
            }
            try out.appendSlice(allocator, raw[esc_start..i]);
            continue;
        }
        const seq_len = std.unicode.utf8ByteSequenceLength(raw[i]) catch {
            try out.append(allocator, raw[i]);
            i += 1;
            width += 1;
            continue;
        };
        if (i + seq_len > raw.len) break;
        try out.appendSlice(allocator, raw[i..][0..seq_len]);
        i += seq_len;
        width += 1;
    }
    if (width < cols) try appendRepeated(out, allocator, ' ', cols - width);
}

pub fn countLines(buf: []const u8) usize {
    if (buf.len == 0) return 0;
    var n: usize = 1;
    for (buf) |b| {
        if (b == '\n') n += 1;
    }
    // Trailing newline means a final empty line from splitScalar; callers that
    // build with terminal '\n' should strip it first. Match splitScalar behavior
    // used by the interleave loop: empty trailing piece after final \n.
    if (buf[buf.len - 1] == '\n') n -= 1;
    return n;
}

pub fn selectedPaneIndex(selected: usize) usize {
    if (selected < DASHBOARD_PANE_COUNT) return selected;
    return 0;
}

pub fn appendStyle(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, enabled: bool, code: []const u8) !void {
    if (enabled) try out.appendSlice(allocator, code);
}

pub fn appendRefreshInterval(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, refresh_interval_ms: u64) !void {
    if (refresh_interval_ms > 0 and refresh_interval_ms % 1000 == 0) {
        try out.print(allocator, "{d}s", .{refresh_interval_ms / 1000});
    } else {
        try out.print(allocator, "{d}ms", .{refresh_interval_ms});
    }
}

/// Honest status digest for the dashboard's right-hand "Agent Output" pane.
/// This is not live REPL traffic — it summarizes the current snapshot so the
/// split view is never blank. Callers own the returned slice.
pub fn formatAgentStatusDigest(allocator: std.mem.Allocator, ds: types.DashboardState) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "Snapshot status\n");
    try out.print(allocator, "health: {s}\n", .{dashboardHealth(ds)});
    try out.print(allocator, "GPU: {s}\n", .{ds.gpu_backend});
    try out.print(allocator, "  accelerated: {s}\n", .{boolText(ds.gpu_accelerated)});
    try out.print(allocator, "  native linked: {s}\n", .{boolText(ds.gpu_linked)});
    try out.print(allocator, "plugins: {d}\n", .{ds.plugin_count});
    try out.print(allocator, "wdbx kv/vectors: {d}/{d}\n", .{ ds.wdbx_entries, ds.wdbx_vectors });
    try out.print(allocator, "scheduler done/fail: {d}/{d}\n", .{ ds.scheduler_completed, ds.scheduler_failed });
    try out.print(allocator, "memory current: {d}\n", .{ds.memory_current});
    try out.appendSlice(allocator, "\n(dashboard status log;\n agent REPL: abi agent tui)");

    return try out.toOwnedSlice(allocator);
}

pub fn appendDashboardHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, options: DiagnosticRenderOptions) !void {
    try appendStyle(out, allocator, options.color, "\x1b[1;36m");
    try appendBorder(out, allocator, "╔", "", "╗");
    try appendRow(out, allocator, "ABI Diagnostics Dashboard", "operational snapshot");
    try appendRow(out, allocator, "health", dashboardHealth(ds));
    try appendBorder(out, allocator, "╚", "", "╝");
    try appendStyle(out, allocator, options.color, "\x1b[0m");
}

pub fn appendDashboardFooter(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, options: DiagnosticRenderOptions) !void {
    try out.append(allocator, '\n');
    try appendStyle(out, allocator, options.color, "\x1b[2m");
    try out.appendSlice(allocator, "[q/Esc] Quit  [r] Refresh  [h/l] Select");
    if (DASHBOARD_PANES.len > 0) {
        const first = DASHBOARD_PANES[0].hotkey;
        const last = DASHBOARD_PANES[DASHBOARD_PANES.len - 1].hotkey;
        if (first == last) {
            try out.print(allocator, "  [{c}] Pane", .{first});
        } else {
            try out.print(allocator, "  [{c}-{c}] Panes", .{ first, last });
        }
    }
    try out.appendSlice(allocator, "  live snapshot every ");
    try appendRefreshInterval(out, allocator, options.refresh_interval_ms);
    try appendStyle(out, allocator, options.color, "\x1b[0m");
    try out.append(allocator, '\n');
}

/// Footer for split-pane layout, includes focus indicator and scroll hints.
pub fn appendDashboardFooterSplit(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, options: DiagnosticRenderOptions, focused: types.FocusedPane) !void {
    try out.append(allocator, '\n');
    try appendStyle(out, allocator, options.color, "\x1b[2m");

    const focus_indicator = if (focused == .left) " [L]" else " [R]";
    try out.print(allocator, "[q] Quit  [Tab] Focus{s}", .{focus_indicator});

    if (focused == .right) {
        try out.appendSlice(allocator, "  [j/k] Scroll");
    } else {
        try out.appendSlice(allocator, "  [r] Refresh  [1-5] Panes  [h/l] Prev/Next");
    }

    try out.appendSlice(allocator, "  every ");
    try appendRefreshInterval(out, allocator, options.refresh_interval_ms);
    try appendStyle(out, allocator, options.color, "\x1b[0m");
    try out.append(allocator, '\n');
}

pub fn appendDiagnosticsPane(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, pane: types.DashboardPaneMeta, idx: usize, selected: usize, options: DiagnosticRenderOptions) !void {
    const highlight = "\x1b[7m";
    const no_highlight = "\x1b[27m";

    if (idx == selected) try appendStyle(out, allocator, options.color, highlight);
    try appendStyle(out, allocator, options.color, paneColor(pane.kind));
    try appendPanelHeader(out, allocator, pane.title);
    try appendPaneBody(out, allocator, ds, pane.kind);
    try appendPanelFooter(out, allocator);
    try appendStyle(out, allocator, options.color, "\x1b[0m");
    if (idx == selected) try appendStyle(out, allocator, options.color, no_highlight);
}

/// Render the agent output pane (right side of split layout).
/// `total_rows` is the visible line budget (including borders) so the pane can
/// stretch to match the left diagnostics column height.
pub fn renderAgentOutputPane(
    out: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    ds: types.DashboardState,
    pane_width: usize,
    total_rows: usize,
    options: DiagnosticRenderOptions,
) !void {
    const is_focused = ds.focused_pane == .right;
    const highlight = if (is_focused) "\x1b[7m" else "";
    const no_highlight = if (is_focused) "\x1b[27m" else "";
    const color_code = paneColor(.agent_output);

    const rows = if (total_rows < 3) @as(usize, 3) else total_rows;
    const body_rows = rows - 2;
    const inner_width = if (pane_width > 2) pane_width - 2 else 0;

    try appendStyle(out, allocator, options.color, highlight);
    try appendStyle(out, allocator, options.color, color_code);

    // Top border (pane_width rule cells + corners)
    try out.appendSlice(allocator, "┌");
    const title = " Agent Output ";
    try out.appendSlice(allocator, title);
    const used = title.len;
    if (used < pane_width) try appendRule(out, allocator, pane_width - used);
    try out.appendSlice(allocator, "┐\n");

    try appendStyle(out, allocator, options.color, "\x1b[0m");
    if (is_focused) try appendStyle(out, allocator, options.color, no_highlight);

    var written: usize = 0;
    if (ds.agent_output_buffer.len == 0) {
        try out.appendSlice(allocator, "│ ");
        try appendFitted(out, allocator, "(no status digest)", inner_width);
        try out.appendSlice(allocator, " │\n");
        written = 1;
    } else {
        var line_iter = std.mem.splitScalar(u8, ds.agent_output_buffer, '\n');
        var idx: usize = 0;
        while (line_iter.next()) |line| : (idx += 1) {
            if (idx < ds.agent_output_scroll) continue;
            if (written >= body_rows) break;
            try out.appendSlice(allocator, "│ ");
            try appendFitted(out, allocator, line, inner_width);
            try out.appendSlice(allocator, " │\n");
            written += 1;
        }
    }

    while (written < body_rows) : (written += 1) {
        try out.appendSlice(allocator, "│ ");
        try appendRepeated(out, allocator, ' ', inner_width);
        try out.appendSlice(allocator, " │\n");
    }

    // Bottom border
    try out.appendSlice(allocator, "└");
    try appendRule(out, allocator, pane_width);
    try out.appendSlice(allocator, "┘\n");
}

pub fn renderDiagnosticsWithOptions(allocator: std.mem.Allocator, ds: types.DashboardState, options: DiagnosticRenderOptions) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try appendDashboardHeader(&out, allocator, ds, options);

    const selected = selectedPaneIndex(ds.selected_pane);
    if (options.compact) {
        try appendDiagnosticsPane(&out, allocator, ds, DASHBOARD_PANES[selected], selected, selected, options);
    } else {
        for (DASHBOARD_PANES, 0..) |pane, idx| {
            try appendDiagnosticsPane(&out, allocator, ds, pane, idx, selected, options);
        }
    }

    try appendDashboardFooter(&out, allocator, options);

    return try out.toOwnedSlice(allocator);
}

/// Render the diagnostics dashboard in split-pane layout with custom options.
pub fn renderDiagnosticsSplitWithOptions(allocator: std.mem.Allocator, ds: types.DashboardState, options: DiagnosticRenderOptions) ![]u8 {
    const sep: []const u8 = SPLIT_SEP;

    // 1. Render left pane content (diagnostics header + panes, no footer)
    var left_buf = std.ArrayListUnmanaged(u8).empty;
    defer left_buf.deinit(allocator);

    try appendDashboardHeader(&left_buf, allocator, ds, options);

    const selected = selectedPaneIndex(ds.selected_pane);
    if (options.compact) {
        try appendDiagnosticsPane(&left_buf, allocator, ds, DASHBOARD_PANES[selected], selected, selected, options);
    } else {
        for (DASHBOARD_PANES, 0..) |pane, idx| {
            try appendDiagnosticsPane(&left_buf, allocator, ds, pane, idx, selected, options);
        }
    }

    const left_rows = countLines(left_buf.items);

    // 2. Render right pane stretched to the left column height
    var right_buf = std.ArrayListUnmanaged(u8).empty;
    defer right_buf.deinit(allocator);

    try renderAgentOutputPane(&right_buf, allocator, ds, AGENT_PANE_WIDTH, left_rows, options);

    // 3. Interleave left and right lines using visible-column padding so
    // UTF-8 box-drawing and ANSI escapes are not byte-sliced mid-sequence.
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var left_iter = std.mem.splitScalar(u8, left_buf.items, '\n');
    var right_iter = std.mem.splitScalar(u8, right_buf.items, '\n');

    while (true) {
        const left_line = left_iter.next() orelse break;
        const right_line = right_iter.next() orelse "";

        try appendFittedVisible(&out, allocator, left_line, LEFT_PANE_COLS);
        try out.appendSlice(allocator, sep);
        try appendFittedVisible(&out, allocator, right_line, AGENT_PANE_WIDTH + 2);
        try out.append(allocator, '\n');
    }

    // 4. Split-mode footer
    try appendDashboardFooterSplit(&out, allocator, options, ds.focused_pane);

    return try out.toOwnedSlice(allocator);
}

test "ansiVisibleWidth ignores CSI and counts UTF-8 codepoints" {
    try std.testing.expectEqual(@as(usize, 5), ansiVisibleWidth("hello"));
    try std.testing.expectEqual(@as(usize, 5), ansiVisibleWidth("\x1b[1;31mhello\x1b[0m"));
    try std.testing.expectEqual(@as(usize, 1), ansiVisibleWidth("─"));
    try std.testing.expectEqual(@as(usize, 0), ansiVisibleWidth(""));
}

test "countLines matches splitScalar trailing-newline semantics" {
    try std.testing.expectEqual(@as(usize, 0), countLines(""));
    try std.testing.expectEqual(@as(usize, 1), countLines("a"));
    try std.testing.expectEqual(@as(usize, 2), countLines("a\nb"));
    try std.testing.expectEqual(@as(usize, 2), countLines("a\nb\n"));
    try std.testing.expectEqual(@as(usize, 1), countLines("a\n"));
}

test "appendFittedVisible pads and preserves ANSI escapes" {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(std.testing.allocator);

    try appendFittedVisible(&out, std.testing.allocator, "\x1b[1mhi\x1b[0m", 5);
    try std.testing.expectEqual(@as(usize, 5), ansiVisibleWidth(out.items));
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\x1b[1m") != null);
    try std.testing.expect(std.mem.endsWith(u8, out.items, "   ") or ansiVisibleWidth(out.items) == 5);
}

test "appendRefreshInterval formats seconds when divisible" {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(std.testing.allocator);

    try appendRefreshInterval(&out, std.testing.allocator, 2000);
    try std.testing.expectEqualStrings("2s", out.items);

    out.clearRetainingCapacity();
    try appendRefreshInterval(&out, std.testing.allocator, 250);
    try std.testing.expectEqualStrings("250ms", out.items);
}

test "selectedPaneIndex clamps out-of-range to zero" {
    try std.testing.expectEqual(@as(usize, 0), selectedPaneIndex(0));
    try std.testing.expectEqual(@as(usize, 3), selectedPaneIndex(3));
    try std.testing.expectEqual(@as(usize, 0), selectedPaneIndex(DASHBOARD_PANE_COUNT + 9));
}

test {
    std.testing.refAllDecls(@This());
}
