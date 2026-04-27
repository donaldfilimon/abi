const std = @import("std");
const builtin = @import("builtin");
const types = @import("../../types.zig");
const terminal_mod = @import("../../terminal.zig");
const render_mod = @import("../../render.zig");
const ansi = @import("../../ansi.zig");
const events_mod = @import("../../events.zig");

const state_mod = @import("state.zig");
const layout_util = @import("layout.zig");
const widget_util = @import("widgets.zig");
const view_overview = @import("view_overview.zig");
const view_features = @import("view_features.zig");
const view_runtime = @import("view_runtime.zig");

pub const View = state_mod.View;
pub const FocusRegion = state_mod.FocusRegion;
pub const LayoutMode = state_mod.LayoutMode;
pub const AppState = state_mod.AppState;
pub const DashboardLayout = layout_util.DashboardLayout;
pub const DashboardAction = state_mod.DashboardAction;

const Screen = render_mod.Screen;
const Terminal = terminal_mod.Terminal;

var sigwinch_received: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

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
    sigwinch_received.store(true, .release);
}

fn renderSummary(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode != .compact) {
        widget_util.drawHorizontalDivider(screen, area, area.height - 1, widget_util.muted_style);
    }

    const divider_rows: u16 = if (mode == .compact) 0 else 1;
    const content: Rect = .{
        .x = area.x,
        .y = area.y,
        .width = area.width,
        .height = area.height -| divider_rows,
    };

    switch (state.current_view) {
        .overview => view_overview.renderOverviewSummary(screen, content, state, mode, screen_width, screen_height),
        .features => view_features.renderFeaturesSummary(screen, content, state),
        .runtime => view_runtime.renderRuntimeSummary(screen, content, state),
    }
}

fn renderDetail(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    switch (state.current_view) {
        .overview => view_overview.renderOverviewDetail(screen, area, state, mode, screen_width, screen_height),
        .features => view_features.renderFeaturesDetail(screen, area, state, mode),
        .runtime => view_runtime.renderRuntimeDetail(screen, area, state, mode),
    }
}

fn renderMinimal(screen: *Screen, area: Rect) void {
    if (area.width == 0 or area.height == 0) return;

    screen.clear();
    widget_util.renderCenteredText(screen, area, 0, "ABI", widget_util.title_style);
    if (area.height > 1) widget_util.renderCenteredText(screen, area, 1, "dashboard", widget_util.body_style);
    if (area.height > 2) widget_util.renderCenteredText(screen, area, 2, "grow terminal", widget_util.muted_style);
    if (area.height > 3) widget_util.renderCenteredText(screen, area, 3, "? help  q quit", widget_util.muted_style);
}

pub fn renderDashboard(screen: *Screen, state: *const AppState) void {
    screen.clear();

    var render_state = state.*;
    state_mod.clampState(&render_state);

    const dl = layout_util.computeLayout(screen.rect());
    if (dl.mode == .minimal) {
        renderMinimal(screen, dl.full);
        if (render_state.help_visible) widget_util.renderHelpOverlay(screen, dl.overlay, &render_state);
        return;
    }

    widget_util.renderHeader(screen, dl.header, &render_state, dl.mode);
    widget_util.renderNav(screen, dl.nav, &render_state, dl.mode);
    renderSummary(screen, dl.summary, &render_state, dl.mode, screen.width, screen.height);
    renderDetail(screen, dl.detail, &render_state, dl.mode, screen.width, screen.height);
    widget_util.renderFooter(screen, dl.footer, &render_state, dl.mode);

    if (render_state.help_visible) {
        widget_util.renderHelpOverlay(screen, dl.overlay, &render_state);
    }
}

fn flushToFd(fd: std.posix.fd_t, data: []const u8) void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = std.c.write(fd, data[offset..].ptr, data.len - offset);
        if (n > 0) {
            offset += @intCast(n);
        } else break;
    }
}

fn flushScreenToFd(screen: *Screen, fd: std.posix.fd_t) !void {
    var buf: [16384]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    screen.flush(&writer) catch {
        flushToFd(fd, buf[0..writer.end]);
        return;
    };
    flushToFd(fd, buf[0..writer.end]);
}

const Rect = types.Rect;

pub fn run(allocator: std.mem.Allocator) !void {
    var term = Terminal.init() catch {
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

    {
        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try ansi.hideCursor(&writer);
        try ansi.clearScreen(&writer);
        flushToFd(stdout_fd, buf[0..writer.end]);
    }

    var app_state = AppState{};
    installSigwinchHandler();

    renderDashboard(&screen, &app_state);
    try flushScreenToFd(&screen, stdout_fd);

    var event_reader = events_mod.EventReader.init();
    while (true) {
        if (sigwinch_received.swap(false, .acquire)) {
            const new_size = try term.getSize();
            screen.deinit();
            screen = try Screen.init(allocator, new_size.width, new_size.height);
        }

        if (event_reader.readEvent() catch null) |event| {
            const action = state_mod.handleKey(&app_state, event.key);
            if (action == .quit) break;
        }

        renderDashboard(&screen, &app_state);
        try flushScreenToFd(&screen, stdout_fd);
    }

    {
        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try ansi.clearScreen(&writer);
        try ansi.moveCursor(&writer, 0, 0);
        try ansi.showCursor(&writer);
        flushToFd(stdout_fd, buf[0..writer.end]);
    }
}
