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

fn dashboardHealth(ds: types.DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (!ds.gpu_accelerated or !ds.gpu_linked) return "degraded";
    return "nominal";
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

test "diagnostics dashboard health distinguishes nominal degraded and attention" {
    const nominal = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
    });
    defer std.testing.allocator.free(nominal);
    try std.testing.expect(std.mem.indexOf(u8, nominal, "nominal") != null);

    const degraded = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "cpu",
        .gpu_accelerated = false,
        .gpu_linked = false,
    });
    defer std.testing.allocator.free(degraded);
    try std.testing.expect(std.mem.indexOf(u8, degraded, "degraded") != null);

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

test {
    std.testing.refAllDecls(@This());
}
