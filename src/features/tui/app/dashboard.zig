//! Interactive framework dashboard.
//!
//! Displays build configuration, feature flags, and GPU backend
//! status in a live terminal UI. Data sourced from build_options.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const types = @import("../types.zig");
const terminal_mod = @import("../terminal.zig");
const render_mod = @import("../render.zig");
const ansi = @import("../ansi.zig");
const layout_mod = @import("../layout.zig");
const widgets = @import("../widgets.zig");
const events_mod = @import("../events.zig");

const Style = types.Style;
const Key = types.Key;
const Rect = types.Rect;
const Screen = render_mod.Screen;
const Terminal = terminal_mod.Terminal;

// SIGWINCH handling for terminal resize detection
var sigwinch_received: bool = false;

fn installSigwinchHandler() void {
    if (comptime builtin.os.tag == .windows) return;
    const act: std.posix.Sigaction = .{
        .handler = .{ .handler = sigwinchHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.WINCH, &act, null);
}

fn sigwinchHandler(_: std.posix.SIG) callconv(.c) void {
    sigwinch_received = true;
}

const green_style = Style{ .fg = .green, .bold = true };
const red_style = Style{ .fg = .red };
const header_style = Style{ .fg = .cyan, .bold = true };
const status_style = Style{ .fg = .black, .bg = .white };
const dim_style = Style{ .fg = .bright_black };

/// Feature flag entry for display.
const FlagEntry = struct {
    name: []const u8,
    enabled: bool,
};

pub const Panel = enum {
    features,
    gpu,
};

pub const AppState = struct {
    focused_panel: Panel = .features,
};

/// Result of computing the dashboard layout regions.
pub const DashboardLayout = struct {
    header: Rect,
    body: Rect,
    status: Rect,
    features_panel: Rect,
    gpu_panel: Rect,
};

/// Actions the dashboard can take in response to input.
pub const DashboardAction = enum {
    none,
    quit,
};

/// Compute the layout regions for the dashboard from a full-screen rect.
pub fn computeLayout(full: Rect) DashboardLayout {
    // Split: header (3) + body + status bar (1)
    const vsplit = full.splitHorizontal(3);
    const header_area = vsplit.top;
    const body_and_status = vsplit.bottom;
    const bsplit = body_and_status.splitHorizontal(body_and_status.height -| 1);
    const body_area = bsplit.top;
    const status_area = bsplit.bottom;

    // Body: split into features (left) and GPU/AI (right)
    const body_split = body_area.splitVertical(body_area.width / 2);

    return .{
        .header = header_area,
        .body = body_area,
        .status = status_area,
        .features_panel = body_split.left,
        .gpu_panel = body_split.right,
    };
}

/// Process a key event and return the resulting action.
pub fn handleKey(state: *AppState, key: Key) DashboardAction {
    switch (key) {
        .char => |c| {
            if (c == 'q' or c == 'Q') return .quit;
        },
        .ctrl => |c| {
            if (c == 'c') return .quit;
        },
        .tab => {
            state.focused_panel = if (state.focused_panel == .features) .gpu else .features;
        },
        .escape => return .quit,
        else => {},
    }
    return .none;
}

/// Hit test the features and GPU panels, returning which panel (if any)
/// contains the given coordinates.
pub fn hitTestPanel(dl: DashboardLayout, x: u16, y: u16) ?Panel {
    if (dl.features_panel.contains(x, y)) return .features;
    if (dl.gpu_panel.contains(x, y)) return .gpu;
    return null;
}

/// Check whether any cell in the slice has a visible (non-space) character.
pub fn hasVisibleCell(cells: []const types.Cell) bool {
    for (cells) |cell| {
        if (cell.char != ' ') return true;
    }
    return false;
}

/// Get all feature flags from build_options (33 distinct flags; catalog features
/// without own flags — embeddings, agents, profiles, constitution — inherit feat_ai).
fn getFeatureFlags() [33]FlagEntry {
    return .{
        // Core features
        .{ .name = "ai", .enabled = build_options.feat_ai },
        .{ .name = "gpu", .enabled = build_options.feat_gpu },
        .{ .name = "database", .enabled = build_options.feat_database },
        .{ .name = "network", .enabled = build_options.feat_network },
        .{ .name = "observability", .enabled = build_options.feat_observability },
        .{ .name = "web", .enabled = build_options.feat_web },
        .{ .name = "pages", .enabled = build_options.feat_pages },
        .{ .name = "search", .enabled = build_options.feat_search },
        .{ .name = "cache", .enabled = build_options.feat_cache },
        .{ .name = "auth", .enabled = build_options.feat_auth },
        .{ .name = "analytics", .enabled = build_options.feat_analytics },
        .{ .name = "cloud", .enabled = build_options.feat_cloud },
        .{ .name = "messaging", .enabled = build_options.feat_messaging },
        .{ .name = "storage", .enabled = build_options.feat_storage },
        .{ .name = "gateway", .enabled = build_options.feat_gateway },
        .{ .name = "compute", .enabled = build_options.feat_compute },
        .{ .name = "documents", .enabled = build_options.feat_documents },
        .{ .name = "desktop", .enabled = build_options.feat_desktop },
        .{ .name = "mobile", .enabled = build_options.feat_mobile },
        .{ .name = "benchmarks", .enabled = build_options.feat_benchmarks },
        // AI sub-features
        .{ .name = "llm", .enabled = build_options.feat_llm },
        .{ .name = "training", .enabled = build_options.feat_training },
        .{ .name = "vision", .enabled = build_options.feat_vision },
        .{ .name = "explore", .enabled = build_options.feat_explore },
        .{ .name = "reasoning", .enabled = build_options.feat_reasoning },
        // Protocols
        .{ .name = "lsp", .enabled = build_options.feat_lsp },
        .{ .name = "mcp", .enabled = build_options.feat_mcp },
        .{ .name = "acp", .enabled = build_options.feat_acp },
        .{ .name = "ha", .enabled = build_options.feat_ha },
        // Standalone modules
        .{ .name = "connectors", .enabled = build_options.feat_connectors },
        .{ .name = "tasks", .enabled = build_options.feat_tasks },
        .{ .name = "inference", .enabled = build_options.feat_inference },
        .{ .name = "tui", .enabled = build_options.feat_tui },
    };
}

fn getGpuFlags() [10]FlagEntry {
    return .{
        .{ .name = "metal", .enabled = build_options.gpu_metal },
        .{ .name = "cuda", .enabled = build_options.gpu_cuda },
        .{ .name = "vulkan", .enabled = build_options.gpu_vulkan },
        .{ .name = "webgpu", .enabled = build_options.gpu_webgpu },
        .{ .name = "opengl", .enabled = build_options.gpu_opengl },
        .{ .name = "opengles", .enabled = build_options.gpu_opengles },
        .{ .name = "webgl2", .enabled = build_options.gpu_webgl2 },
        .{ .name = "stdgpu", .enabled = build_options.gpu_stdgpu },
        .{ .name = "fpga", .enabled = build_options.gpu_fpga },
        .{ .name = "tpu", .enabled = build_options.gpu_tpu },
    };
}

/// Render the dashboard to a screen buffer.
pub fn renderDashboard(screen: *Screen, state: *const AppState) void {
    screen.clear();
    const dl = computeLayout(screen.rect());

    // Header
    widgets.renderPanel(screen, dl.header, " ABI Dashboard ", header_style);
    if (dl.header.height >= 2) {
        const version_text = "v" ++ build_options.package_version ++ " | Zig 0.16 | Multi-Profile AI + WDBX";
        widgets.renderText(screen, .{
            .x = dl.header.x + 2,
            .y = dl.header.y + 1,
            .width = dl.header.width -| 4,
            .height = 1,
        }, version_text, dim_style);
    }

    // Feature flags panel
    renderFeaturePanel(screen, dl.features_panel, state.focused_panel == .features);

    // GPU + AI panel
    renderGpuPanel(screen, dl.gpu_panel, state.focused_panel == .gpu);

    // Status bar
    widgets.renderStatusBar(screen, dl.status, " q:quit  tab:focus", "ABI Framework ", status_style);
}

/// Render a list of flag entries starting at a given row within an inner rect.
/// Returns the number of rows rendered.
fn renderFlags(screen: *Screen, inner: Rect, flags: []const FlagEntry, start_row: u16) u16 {
    var rendered: u16 = 0;
    for (flags, 0..) |flag, i| {
        const row = start_row + @as(u16, @intCast(i));
        if (row >= inner.height) break;
        const indicator: []const u8 = if (flag.enabled) "[+]" else "[-]";
        const style = if (flag.enabled) green_style else red_style;
        widgets.renderText(screen, .{ .x = inner.x, .y = inner.y + row, .width = 4, .height = 1 }, indicator, style);
        widgets.renderText(screen, .{ .x = inner.x + 4, .y = inner.y + row, .width = inner.width -| 4, .height = 1 }, flag.name, style);
        rendered += 1;
    }
    return rendered;
}

fn innerRect(area: Rect) Rect {
    return .{
        .x = area.x + 2,
        .y = area.y + 1,
        .width = area.width -| 4,
        .height = area.height -| 2,
    };
}

fn renderFeaturePanel(screen: *Screen, area: Rect, focused: bool) void {
    const style = if (focused) header_style else Style{ .fg = .white };
    widgets.renderPanel(screen, area, " Features ", style);
    const flags = getFeatureFlags();
    _ = renderFlags(screen, innerRect(area), &flags, 0);
}

fn renderGpuPanel(screen: *Screen, area: Rect, focused: bool) void {
    const style = if (focused) header_style else Style{ .fg = .white };
    widgets.renderPanel(screen, area, " GPU Backends ", style);
    const inner = innerRect(area);
    const gpu_flags = getGpuFlags();
    const rows_used = renderFlags(screen, inner, &gpu_flags, 0);

    // AI sub-features section
    const ai_row = rows_used + 1;
    if (ai_row >= inner.height) return;

    widgets.renderText(screen, .{
        .x = inner.x,
        .y = inner.y + ai_row,
        .width = inner.width,
        .height = 1,
    }, "AI Sub-features:", style);

    const ai_flags = [_]FlagEntry{
        .{ .name = "llm", .enabled = build_options.feat_llm },
        .{ .name = "training", .enabled = build_options.feat_training },
        .{ .name = "vision", .enabled = build_options.feat_vision },
        .{ .name = "reasoning", .enabled = build_options.feat_reasoning },
    };
    _ = renderFlags(screen, inner, &ai_flags, ai_row + 1);
}

/// Write the contents of a buffer to a POSIX file descriptor.
fn flushToFd(fd: std.posix.fd_t, data: []const u8) void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = std.c.write(fd, data[offset..].ptr, data.len - offset);
        if (n > 0) {
            offset += @intCast(n);
        } else break;
    }
}

/// Flush a screen to a file descriptor via a buffer.
fn flushScreenToFd(screen: *Screen, fd: std.posix.fd_t) !void {
    var buf: [16384]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    screen.flush(&writer) catch {
        // Buffer may be too small; flush what we have
        flushToFd(fd, buf[0..writer.end]);
        return;
    };
    flushToFd(fd, buf[0..writer.end]);
}

/// Run the interactive dashboard.
pub fn run(allocator: std.mem.Allocator) !void {
    var term = Terminal.init() catch {
        // Fallback: print static dashboard to stderr
        std.debug.print("TUI dashboard requires an interactive terminal.\n", .{});
        std.debug.print("Use 'abi doctor' for non-interactive diagnostics.\n", .{});
        return;
    };
    defer term.deinit();

    const size = try term.getSize();
    var screen = try Screen.init(allocator, size.width, size.height);
    defer screen.deinit();

    try term.enableRawMode();
    defer term.disableRawMode();

    const stdout_fd = std.posix.STDOUT_FILENO;

    // Hide cursor and clear screen
    {
        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try ansi.hideCursor(&writer);
        try ansi.clearScreen(&writer);
        flushToFd(stdout_fd, buf[0..writer.end]);
    }

    var app_state = AppState{};

    // Install SIGWINCH handler for terminal resize detection
    installSigwinchHandler();

    // Initial render
    renderDashboard(&screen, &app_state);
    try flushScreenToFd(&screen, stdout_fd);

    var event_reader = events_mod.EventReader.init();

    // Event loop — readEvent uses VMIN=0/VTIME=1 (100ms non-blocking poll)
    while (true) {
        // Handle terminal resize
        if (sigwinch_received) {
            sigwinch_received = false;
            const new_size = try term.getSize();
            screen.deinit();
            screen = try Screen.init(allocator, new_size.width, new_size.height);
        }

        if (event_reader.readEvent() catch null) |event| {
            const action = handleKey(&app_state, event.key);
            if (action == .quit) break;
        }

        renderDashboard(&screen, &app_state);
        try flushScreenToFd(&screen, stdout_fd);
    }

    // Cleanup: clear screen and show cursor
    {
        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try ansi.clearScreen(&writer);
        try ansi.moveCursor(&writer, 0, 0);
        try ansi.showCursor(&writer);
        flushToFd(stdout_fd, buf[0..writer.end]);
    }
}

// =============================================================================
// Tests
// =============================================================================

test "getFeatureFlags returns 33 entries" {
    const flags = getFeatureFlags();
    try std.testing.expectEqual(@as(usize, 33), flags.len);
}

test "getGpuFlags returns 10 entries" {
    const flags = getGpuFlags();
    try std.testing.expectEqual(@as(usize, 10), flags.len);
}

test "renderDashboard does not crash" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();
    const app_state = AppState{};
    renderDashboard(&screen, &app_state);
    // Verify some content was written
    try std.testing.expect(screen.back[0].char != ' ' or screen.back[1].char != ' ');
}

test "computeLayout partitions 80x24 screen" {
    const full = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const dl = computeLayout(full);

    // Header is 3 rows
    try std.testing.expectEqual(@as(u16, 3), dl.header.height);
    try std.testing.expectEqual(@as(u16, 0), dl.header.y);

    // Status bar is 1 row at the bottom
    try std.testing.expectEqual(@as(u16, 1), dl.status.height);
    try std.testing.expectEqual(@as(u16, 23), dl.status.y);

    // Body fills the gap
    try std.testing.expectEqual(@as(u16, 20), dl.body.height);
    try std.testing.expectEqual(@as(u16, 3), dl.body.y);

    // Panels split the body in half
    try std.testing.expectEqual(@as(u16, 40), dl.features_panel.width);
    try std.testing.expectEqual(@as(u16, 40), dl.gpu_panel.width);
    try std.testing.expectEqual(@as(u16, 0), dl.features_panel.x);
    try std.testing.expectEqual(@as(u16, 40), dl.gpu_panel.x);
}

test "computeLayout compact screen 40x10" {
    const full = Rect{ .x = 0, .y = 0, .width = 40, .height = 10 };
    const dl = computeLayout(full);

    try std.testing.expectEqual(@as(u16, 3), dl.header.height);
    try std.testing.expectEqual(@as(u16, 1), dl.status.height);
    // Body = 10 - 3 - 1 = 6
    try std.testing.expectEqual(@as(u16, 6), dl.body.height);
    // Panels split 40 / 2 = 20 each
    try std.testing.expectEqual(@as(u16, 20), dl.features_panel.width);
    try std.testing.expectEqual(@as(u16, 20), dl.gpu_panel.width);
}

test "computeLayout minimal 20x5 screen" {
    const full = Rect{ .x = 0, .y = 0, .width = 20, .height = 5 };
    const dl = computeLayout(full);

    try std.testing.expectEqual(@as(u16, 3), dl.header.height);
    try std.testing.expectEqual(@as(u16, 1), dl.status.height);
    try std.testing.expectEqual(@as(u16, 1), dl.body.height);
    try std.testing.expectEqual(@as(u16, 10), dl.features_panel.width);
    try std.testing.expectEqual(@as(u16, 10), dl.gpu_panel.width);
}

test "handleKey quit on q" {
    var state = AppState{};
    const action = handleKey(&state, Key{ .char = 'q' });
    try std.testing.expectEqual(DashboardAction.quit, action);
}

test "handleKey quit on Q" {
    var state = AppState{};
    const action = handleKey(&state, Key{ .char = 'Q' });
    try std.testing.expectEqual(DashboardAction.quit, action);
}

test "handleKey quit on escape" {
    var state = AppState{};
    const action = handleKey(&state, .escape);
    try std.testing.expectEqual(DashboardAction.quit, action);
}

test "handleKey quit on ctrl-c" {
    var state = AppState{};
    const action = handleKey(&state, Key{ .ctrl = 'c' });
    try std.testing.expectEqual(DashboardAction.quit, action);
}

test "handleKey tab toggles focus" {
    var state = AppState{};
    try std.testing.expectEqual(Panel.features, state.focused_panel);

    const a1 = handleKey(&state, .tab);
    try std.testing.expectEqual(DashboardAction.none, a1);
    try std.testing.expectEqual(Panel.gpu, state.focused_panel);

    const a2 = handleKey(&state, .tab);
    try std.testing.expectEqual(DashboardAction.none, a2);
    try std.testing.expectEqual(Panel.features, state.focused_panel);
}

test "handleKey unknown key returns none" {
    var state = AppState{};
    const action = handleKey(&state, Key{ .char = 'x' });
    try std.testing.expectEqual(DashboardAction.none, action);
}

test "hitTestPanel returns features for left half" {
    const full = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const dl = computeLayout(full);
    // Inside features panel
    try std.testing.expectEqual(Panel.features, hitTestPanel(dl, 5, 10).?);
    // Inside GPU panel
    try std.testing.expectEqual(Panel.gpu, hitTestPanel(dl, 50, 10).?);
    // Header area — neither panel
    try std.testing.expect(hitTestPanel(dl, 5, 1) == null);
    // Status bar — neither panel
    try std.testing.expect(hitTestPanel(dl, 5, 23) == null);
}

test "hasVisibleCell detects non-blank" {
    const blank = [_]types.Cell{.{}} ** 5;
    try std.testing.expect(!hasVisibleCell(&blank));

    var cells = [_]types.Cell{.{}} ** 5;
    cells[2] = .{ .char = 'A', .style = .{} };
    try std.testing.expect(hasVisibleCell(&cells));
}

test "renderDashboard smoke 40x10" {
    var screen = try Screen.init(std.testing.allocator, 40, 10);
    defer screen.deinit();
    const app_state = AppState{};
    renderDashboard(&screen, &app_state);
    try std.testing.expect(hasVisibleCell(screen.back));
}

test "renderDashboard smoke 120x40" {
    var screen = try Screen.init(std.testing.allocator, 120, 40);
    defer screen.deinit();
    const app_state = AppState{};
    renderDashboard(&screen, &app_state);
    try std.testing.expect(hasVisibleCell(screen.back));
}

test "renderDashboard with GPU focus" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();
    const app_state = AppState{ .focused_panel = .gpu };
    renderDashboard(&screen, &app_state);
    try std.testing.expect(hasVisibleCell(screen.back));
}

test {
    std.testing.refAllDecls(@This());
}
