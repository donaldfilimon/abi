const std = @import("std");
const sanitize = @import("sanitize.zig");
const types = @import("types.zig");

const sanitizeControlBytes = sanitize.sanitizeControlBytes;

const DASHBOARD_PANES = types.DASHBOARD_PANES;
const DASHBOARD_PANE_COUNT = types.DASHBOARD_PANE_COUNT;

const DIAG_WIDTH: usize = 68;
const LABEL_WIDTH: usize = 25;
const VALUE_WIDTH: usize = 40;
const MAX_PLUGIN_ROWS: usize = 6;
/// Visible columns for the left diagnostics column (border corners + DIAG_WIDTH).
const LEFT_PANE_COLS: usize = DIAG_WIDTH + 2;
/// Visible columns for the right agent-output column.
const AGENT_PANE_WIDTH: usize = 40;
/// Horizontal gap between the diagnostics column and the agent-status column.
/// Boxes already carry their own `│` borders; a plain gap avoids a triple-pipe seam.
const SPLIT_SEP: []const u8 = "  ";

pub const DiagnosticRenderOptions = struct {
    color: bool = true,
    refresh_interval_ms: u64 = 1000,
    compact: bool = false,
};

pub fn statusText(status: types.Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}

/// Operational health band for the dashboard header / JSON snapshot.
/// `cpu` is the normal Metal-linked CPU-SIMD path (accelerated=false is common);
/// `degraded` is reserved for future explicit failure signals — not "GPU idle".
pub fn dashboardHealth(ds: types.DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (ds.gpu_accelerated and ds.gpu_linked) return "nominal";
    return "cpu";
}

pub fn dashboardPaneIndexForKey(key: u8) ?usize {
    for (DASHBOARD_PANES, 0..) |pane, idx| {
        if (pane.hotkey == key) return idx;
    }
    return null;
}

pub fn dashboardPaneName(kind: types.PaneKind) []const u8 {
    return switch (kind) {
        .system => "system",
        .plugins => "plugins",
        .storage => "storage",
        .scheduler => "scheduler",
        .memory => "memory",
        .agent_output => "agent_output",
    };
}

pub fn dashboardPaneIndexForToken(token: []const u8) ?usize {
    if (token.len == 1) {
        if (dashboardPaneIndexForKey(token[0])) |idx| return idx;
    }
    for (DASHBOARD_PANES, 0..) |pane, idx| {
        if (std.ascii.eqlIgnoreCase(token, dashboardPaneName(pane.kind))) return idx;
        if (pane.kind == .storage and std.ascii.eqlIgnoreCase(token, "wdbx")) return idx;
    }
    return null;
}

pub fn nextDashboardPane(current: usize, key: u8) ?usize {
    if (dashboardPaneIndexForKey(key)) |idx| return idx;
    if (key == 'l' or key == 'L' or key == '>') return (current + 1) % DASHBOARD_PANE_COUNT;
    if (key == 'h' or key == 'H' or key == '<') return (current + DASHBOARD_PANE_COUNT - 1) % DASHBOARD_PANE_COUNT;
    return null;
}

fn boolText(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

fn appendRepeated(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, byte: u8, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.append(allocator, byte);
}

fn appendRule(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.appendSlice(allocator, "─");
}

fn appendFitted(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, raw: []const u8, width: usize) !void {
    const safe = try sanitizeControlBytes(allocator, raw);
    defer allocator.free(safe);

    if (safe.len <= width) {
        try out.appendSlice(allocator, safe);
        try appendRepeated(out, allocator, ' ', width - safe.len);
        return;
    }

    if (width == 0) return;
    if (width == 1) {
        try out.append(allocator, '~');
        return;
    }

    const end = utf8PrefixLen(safe, width - 1);
    try out.appendSlice(allocator, safe[0..end]);
    try appendRepeated(out, allocator, ' ', (width - 1) - end);
    try out.append(allocator, '~');
}

fn utf8PrefixLen(input: []const u8, max_len: usize) usize {
    var end = @min(input.len, max_len);
    while (end > 0 and !std.unicode.utf8ValidateSlice(input[0..end])) : (end -= 1) {}
    return end;
}

/// Visible terminal columns in `s`, treating CSI/ANSI sequences as width 0 and
/// each UTF-8 codepoint as width 1 (sufficient for ASCII + box-drawing).
fn ansiVisibleWidth(s: []const u8) usize {
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
fn appendFittedVisible(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, raw: []const u8, cols: usize) !void {
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

fn countLines(buf: []const u8) usize {
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

fn appendBorder(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, left: []const u8, title: []const u8, right: []const u8) !void {
    try out.appendSlice(allocator, left);
    if (title.len > 0) {
        try out.appendSlice(allocator, " ");
        try appendFitted(out, allocator, title, @min(title.len, DIAG_WIDTH - 4));
        try out.appendSlice(allocator, " ");
        const used = @min(title.len, DIAG_WIDTH - 4) + 2;
        if (used < DIAG_WIDTH) try appendRule(out, allocator, DIAG_WIDTH - used);
    } else {
        try appendRule(out, allocator, DIAG_WIDTH);
    }
    try out.appendSlice(allocator, right);
    try out.append(allocator, '\n');
}

fn appendRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: []const u8) !void {
    try out.appendSlice(allocator, "│ ");
    try appendFitted(out, allocator, label, LABEL_WIDTH);
    try out.appendSlice(allocator, " ");
    try appendFitted(out, allocator, value, VALUE_WIDTH);
    try out.appendSlice(allocator, " │\n");
}

fn appendMetricRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: usize) !void {
    var buf: [32]u8 = undefined;
    const rendered = try std.fmt.bufPrint(&buf, "{d}", .{value});
    try appendRow(out, allocator, label, rendered);
}

fn appendPanelHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, title: []const u8) !void {
    try appendBorder(out, allocator, "┌", title, "┐");
}

fn appendPanelFooter(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator) !void {
    try appendBorder(out, allocator, "└", "", "┘");
}

fn appendPluginRows(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, plugin_names: []const []const u8) !void {
    const shown = @min(plugin_names.len, MAX_PLUGIN_ROWS);
    var i: usize = 0;
    while (i < shown) : (i += 1) {
        try appendRow(out, allocator, "plugin", plugin_names[i]);
    }
    if (plugin_names.len > shown) {
        var buf: [48]u8 = undefined;
        const more = try std.fmt.bufPrint(&buf, "+{d} more registered", .{plugin_names.len - shown});
        try appendRow(out, allocator, "plugin", more);
    }
}

fn paneColor(kind: types.PaneKind) []const u8 {
    return switch (kind) {
        .system => "\x1b[1;33m",
        .plugins => "\x1b[1;32m",
        .storage => "\x1b[1;35m",
        .scheduler => "\x1b[1;34m",
        .memory => "\x1b[1;31m",
        .agent_output => "\x1b[1;36m",
    };
}

fn selectedPaneIndex(selected: usize) usize {
    if (selected < DASHBOARD_PANE_COUNT) return selected;
    return 0;
}

fn appendPaneBody(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, kind: types.PaneKind) !void {
    switch (kind) {
        .system => {
            try appendRow(out, allocator, "GPU backend", ds.gpu_backend);
            try appendRow(out, allocator, "accelerated", boolText(ds.gpu_accelerated));
            try appendRow(out, allocator, "native linked", boolText(ds.gpu_linked));
        },
        .plugins => {
            try appendMetricRow(out, allocator, "Registered", ds.plugin_count);
            try appendPluginRows(out, allocator, ds.plugin_names);
        },
        .storage => {
            try appendRow(out, allocator, "scope", "ephemeral CLI probe");
            try appendMetricRow(out, allocator, "Block chain", ds.wdbx_blocks);
            try appendMetricRow(out, allocator, "Vectors", ds.wdbx_vectors);
            try appendMetricRow(out, allocator, "KV Entries", ds.wdbx_entries);
            try appendMetricRow(out, allocator, "Spatial 3D", ds.wdbx_spatial_records);
        },
        .scheduler => {
            try appendRow(out, allocator, "source", ds.scheduler_source);
            try appendMetricRow(out, allocator, "Running", ds.scheduler_running);
            try appendMetricRow(out, allocator, "Pending", ds.scheduler_pending);
            try appendMetricRow(out, allocator, "Completed", ds.scheduler_completed);
            try appendMetricRow(out, allocator, "Failed", ds.scheduler_failed);
        },
        .memory => {
            try appendRow(out, allocator, "source", ds.memory_source);
            try appendMetricRow(out, allocator, "Peak bytes", ds.memory_peak);
            try appendMetricRow(out, allocator, "Current bytes", ds.memory_current);
            try appendMetricRow(out, allocator, "Leaked bytes", ds.memory_leaked);
        },
        .agent_output => {
            try appendRow(out, allocator, "Agent Output", "see right pane");
        },
    }
}

fn appendStyle(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, enabled: bool, code: []const u8) !void {
    if (enabled) try out.appendSlice(allocator, code);
}

fn appendDiagnosticsPane(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, pane: types.DashboardPaneMeta, idx: usize, selected: usize, options: DiagnosticRenderOptions) !void {
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
fn renderAgentOutputPane(
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

/// Footer for split-pane layout, includes focus indicator and scroll hints.
fn appendDashboardFooterSplit(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, options: DiagnosticRenderOptions, focused: types.FocusedPane) !void {
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

fn appendRefreshInterval(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, refresh_interval_ms: u64) !void {
    if (refresh_interval_ms > 0 and refresh_interval_ms % 1000 == 0) {
        try out.print(allocator, "{d}s", .{refresh_interval_ms / 1000});
    } else {
        try out.print(allocator, "{d}ms", .{refresh_interval_ms});
    }
}

fn appendDashboardFooter(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, options: DiagnosticRenderOptions) !void {
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

fn appendDashboardHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, options: DiagnosticRenderOptions) !void {
    try appendStyle(out, allocator, options.color, "\x1b[1;36m");
    try appendBorder(out, allocator, "╔", "", "╗");
    try appendRow(out, allocator, "ABI Diagnostics Dashboard", "operational snapshot");
    try appendRow(out, allocator, "health", dashboardHealth(ds));
    try appendBorder(out, allocator, "╚", "", "╝");
    try appendStyle(out, allocator, options.color, "\x1b[0m");
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: types.State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const title = try sanitizeControlBytes(allocator, state.title);
    defer allocator.free(title);

    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "| {s:<28} |\n", .{title});
    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "status: {s}\n", .{statusText(state.status)});
    for (state.items) |item| {
        const label = try sanitizeControlBytes(allocator, item.label);
        defer allocator.free(label);
        const value = try sanitizeControlBytes(allocator, item.value);
        defer allocator.free(value);
        try out.print(allocator, "- {s}: {s}\n", .{ label, value });
    }
    try out.print(allocator, "\nCommands: abi help | abi agent train all | abi agent os dry-run <cmd>\n", .{});

    return try out.toOwnedSlice(allocator);
}

pub fn renderDiagnostics(allocator: std.mem.Allocator, ds: types.DashboardState) ![]u8 {
    return renderDiagnosticsWithOptions(allocator, ds, .{});
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

/// Render the diagnostics dashboard in split-pane layout: diagnostics (left)
/// and agent output log (right) side by side.
pub fn renderDiagnosticsSplit(allocator: std.mem.Allocator, ds: types.DashboardState) ![]u8 {
    return renderDiagnosticsSplitWithOptions(allocator, ds, .{});
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

pub fn writeDashboard(writer: anytype, allocator: std.mem.Allocator, state: types.State) !void {
    const rendered = try renderDashboard(allocator, state);
    defer allocator.free(rendered);
    try writer.writeAll(rendered);
}

pub fn writeDiagnostics(writer: anytype, allocator: std.mem.Allocator, ds: types.DashboardState) !void {
    const rendered = try renderDiagnostics(allocator, ds);
    defer allocator.free(rendered);
    try writer.writeAll(rendered);
}

test "dashboard requires a title" {
    try std.testing.expectError(error.InvalidTuiState, renderDashboard(std.testing.allocator, .{ .title = "" }));
}

test "dashboard renders status and items" {
    const rendered = try renderDashboard(std.testing.allocator, .{
        .title = "ABI",
        .status = .warning,
        .items = &.{.{ .label = "AI", .value = "safe" }},
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "status: warning") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "- AI: safe") != null);
}

test "diagnostics dashboard renders all panes" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .plugin_count = 2,
        .plugin_names = &.{ "core", "wdbx" },
        .wdbx_blocks = 5,
        .wdbx_vectors = 10,
        .wdbx_entries = 3,
        .wdbx_spatial_records = 4,
        .scheduler_source = "test snapshot",
        .scheduler_running = 1,
        .scheduler_pending = 2,
        .scheduler_completed = 7,
        .scheduler_failed = 1,
        .memory_source = "MemoryTracker",
        .memory_peak = 4096,
        .memory_current = 2048,
        .memory_leaked = 0,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "operational snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "health") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "metal") != null);
    for (DASHBOARD_PANES) |pane| {
        try std.testing.expect(std.mem.indexOf(u8, rendered, pane.title) != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spatial 3D") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Failed") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "test snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "MemoryTracker") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Peak") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Leaked") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Refresh") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "[1-5] Panes") != null);
}

test "diagnostics pane metadata drives navigation" {
    try std.testing.expectEqual(@as(usize, 5), DASHBOARD_PANE_COUNT);
    for (DASHBOARD_PANES, 0..) |pane, idx| {
        try std.testing.expectEqual(idx, dashboardPaneIndexForKey(pane.hotkey).?);
        try std.testing.expectEqual(idx, dashboardPaneIndexForToken(dashboardPaneName(pane.kind)).?);
    }
    try std.testing.expectEqual(@as(usize, 2), dashboardPaneIndexForToken("wdbx").?);
    try std.testing.expectEqual(@as(usize, 4), dashboardPaneIndexForToken("5").?);
    try std.testing.expect(dashboardPaneIndexForToken("missing") == null);
    try std.testing.expectEqual(@as(?usize, 1), nextDashboardPane(0, 'l'));
    try std.testing.expectEqual(@as(?usize, 0), nextDashboardPane(DASHBOARD_PANE_COUNT - 1, 'L'));
    try std.testing.expectEqual(@as(?usize, DASHBOARD_PANE_COUNT - 1), nextDashboardPane(0, 'h'));
    try std.testing.expect(nextDashboardPane(0, '9') == null);
}

test "diagnostics dashboard summarizes attention state and bounds plugin rows" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "cpu",
        .gpu_accelerated = false,
        .gpu_linked = false,
        .plugin_count = 8,
        .plugin_names = &.{ "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7" },
        .scheduler_failed = 1,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "attention") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "+2 more registered") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "p7") == null);
}

test "diagnostics dashboard health distinguishes nominal cpu and attention" {
    const nominal = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
    });
    defer std.testing.allocator.free(nominal);
    try std.testing.expect(std.mem.indexOf(u8, nominal, "nominal") != null);

    const cpu = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "cpu",
        .gpu_accelerated = false,
        .gpu_linked = false,
    });
    defer std.testing.allocator.free(cpu);
    try std.testing.expect(std.mem.indexOf(u8, cpu, "cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, cpu, "degraded") == null);

    const attention = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .scheduler_failed = 1,
    });
    defer std.testing.allocator.free(attention);
    try std.testing.expect(std.mem.indexOf(u8, attention, "attention") != null);
}

test "diagnostics dashboard keeps truncated UTF-8 valid" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .plugin_count = 1,
        .plugin_names = &.{"éééééééééééééééééééééééé────────────────"},
        .memory_source = "éééééééééééééééééééééééé────────────────",
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.unicode.utf8ValidateSlice(rendered));
}

test "diagnostics dashboard can highlight memory pane" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .memory_source = "MemoryTracker",
        .selected_pane = DASHBOARD_PANE_COUNT - 1,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "\x1b[7m\x1b[1;31m") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Memory") != null);
}

test "diagnostics dashboard defaults invalid selected pane to first pane" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .selected_pane = DASHBOARD_PANE_COUNT + 20,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "\x1b[7m\x1b[1;33m") != null);
}

test "diagnostics dashboard can render without ANSI style escapes" {
    const rendered = try renderDiagnosticsWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .selected_pane = DASHBOARD_PANE_COUNT - 1,
    }, .{ .color = false });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Memory") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "\x1b[") == null);
}

test "diagnostics dashboard compact mode renders only selected pane" {
    const rendered = try renderDiagnosticsWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .scheduler_source = "test-scheduler",
        .selected_pane = 3,
    }, .{ .color = false, .compact = true });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Scheduler") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "test-scheduler") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "System") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "WDBX Storage") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Memory") == null);
}

test "split-pane dashboard renders both sides with placeholder agent output" {
    const rendered = try renderDiagnosticsSplitWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
    }, .{ .color = false });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Agent Output") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "no status digest") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, SPLIT_SEP) != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "[Tab] Focus [L]") != null);
    // Left diagnostics must keep full GPU/plugin labels (no 36-byte shredding).
    try std.testing.expect(std.mem.indexOf(u8, rendered, "GPU backend") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "operational snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Registered") != null);
    try std.testing.expect(std.unicode.utf8ValidateSlice(rendered));
}

test "formatAgentStatusDigest summarizes dashboard snapshot" {
    const digest = try formatAgentStatusDigest(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = false,
        .gpu_linked = false,
        .plugin_count = 16,
        .wdbx_entries = 3,
        .wdbx_vectors = 7,
        .scheduler_completed = 2,
        .scheduler_failed = 0,
        .memory_current = 128,
    });
    defer std.testing.allocator.free(digest);

    try std.testing.expect(std.mem.indexOf(u8, digest, "health: cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, digest, "GPU: metal") != null);
    try std.testing.expect(std.mem.indexOf(u8, digest, "plugins: 16") != null);
    try std.testing.expect(std.mem.indexOf(u8, digest, "abi agent tui") != null);
}

test "split-pane dashboard preserves UTF-8 box drawing under color" {
    const rendered = try renderDiagnosticsSplitWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .plugin_count = 1,
        .plugin_names = &.{"core"},
        .agent_output_buffer = "Snapshot status\nhealth: nominal",
    }, .{ .color = true });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.unicode.utf8ValidateSlice(rendered));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "\u{fffd}") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "GPU backend") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Snapshot status") != null);
}

test "split-pane dashboard with right focus shows correct indicator" {
    const rendered = try renderDiagnosticsSplitWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .focused_pane = .right,
    }, .{ .color = false });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "[Tab] Focus [R]") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "[j/k] Scroll") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Scheduler") != null);
}

test "split-pane dashboard renders agent output buffer content when present" {
    const rendered = try renderDiagnosticsSplitWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .agent_output_buffer = "hello world\nline two",
    }, .{ .color = false });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "hello world") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "line two") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "no status digest") == null);
}

test "split-pane dashboard respects agent_output_scroll offset" {
    const rendered = try renderDiagnosticsSplitWithOptions(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .agent_output_buffer = "line1\nline2\nline3",
        .agent_output_scroll = 1,
    }, .{ .color = false });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "line1") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "line2") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "line3") != null);
}

test {
    std.testing.refAllDecls(@This());
}
