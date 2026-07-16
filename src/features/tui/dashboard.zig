const std = @import("std");
const sanitize = @import("sanitize.zig");
const types = @import("types.zig");
const render = @import("dashboard_render.zig");

const sanitizeControlBytes = sanitize.sanitizeControlBytes;

const DASHBOARD_PANES = types.DASHBOARD_PANES;
const DASHBOARD_PANE_COUNT = types.DASHBOARD_PANE_COUNT;

pub const DiagnosticRenderOptions = render.DiagnosticRenderOptions;
pub const SPLIT_SEP = render.SPLIT_SEP;
pub const dashboardHealth = render.dashboardHealth;
pub const formatAgentStatusDigest = render.formatAgentStatusDigest;
pub const renderDiagnosticsWithOptions = render.renderDiagnosticsWithOptions;
pub const renderDiagnosticsSplitWithOptions = render.renderDiagnosticsSplitWithOptions;

pub fn statusText(status: types.Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
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

pub fn renderDiagnosticsSplit(allocator: std.mem.Allocator, ds: types.DashboardState) ![]u8 {
    return renderDiagnosticsSplitWithOptions(allocator, ds, .{});
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

test "diagnostics storage pane discloses ephemeral CLI probe scope" {
    const rendered = try renderDiagnosticsWithOptions(std.testing.allocator, .{
        .gpu_backend = "cpu",
        .selected_pane = 2,
    }, .{ .color = false, .compact = true });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "ephemeral CLI probe") != null);
}

test {
    std.testing.refAllDecls(@This());
}
