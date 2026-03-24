//! Interactive framework dashboard.
//!
//! Displays build configuration, feature flags, and GPU backend
//! status in a live terminal UI. Data sourced from build_options.

const std = @import("std");
const build_options = @import("build_options");
const types = @import("../types.zig");
const terminal_mod = @import("../terminal.zig");
const render_mod = @import("../render.zig");
const ansi = @import("../ansi.zig");
const layout_mod = @import("../layout.zig");
const widgets = @import("../widgets.zig");
const events_mod = @import("../events.zig");

const Style = types.Style;
const Color = types.Color;
const Rect = types.Rect;
const Screen = render_mod.Screen;
const Terminal = terminal_mod.Terminal;

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

/// Get all feature flags from build_options.
fn getFeatureFlags() [20]FlagEntry {
    return .{
        .{ .name = "ai", .enabled = build_options.feat_ai },
        .{ .name = "gpu", .enabled = build_options.feat_gpu },
        .{ .name = "database", .enabled = build_options.feat_database },
        .{ .name = "network", .enabled = build_options.feat_network },
        .{ .name = "web", .enabled = build_options.feat_web },
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
        .{ .name = "lsp", .enabled = build_options.feat_lsp },
        .{ .name = "mcp", .enabled = build_options.feat_mcp },
    };
}

fn getGpuFlags() [4]FlagEntry {
    return .{
        .{ .name = "metal", .enabled = build_options.gpu_metal },
        .{ .name = "cuda", .enabled = build_options.gpu_cuda },
        .{ .name = "vulkan", .enabled = build_options.gpu_vulkan },
        .{ .name = "stdgpu", .enabled = build_options.gpu_stdgpu },
    };
}

/// Render the dashboard to a screen buffer.
pub fn renderDashboard(screen: *Screen) void {
    screen.clear();
    const full = screen.rect();

    // Split: header (3) + body + status bar (1)
    const vsplit = full.splitHorizontal(3);
    const header_area = vsplit.top;
    const body_and_status = vsplit.bottom;
    const bsplit = body_and_status.splitHorizontal(body_and_status.height -| 1);
    const body_area = bsplit.top;
    const status_area = bsplit.bottom;

    // Header
    widgets.renderPanel(screen, header_area, " ABI Dashboard ", header_style);
    if (header_area.height >= 2) {
        const version_text = "v" ++ build_options.package_version ++ " | Zig 0.16 | Multi-Persona AI + WDBX";
        widgets.renderText(screen, .{
            .x = header_area.x + 2,
            .y = header_area.y + 1,
            .width = header_area.width -| 4,
            .height = 1,
        }, version_text, dim_style);
    }

    // Body: split into features (left) and GPU/AI (right)
    const body_split = body_area.splitVertical(body_area.width / 2);
    const left_area = body_split.left;
    const right_area = body_split.right;

    // Feature flags panel
    renderFeaturePanel(screen, left_area);

    // GPU + AI panel
    renderGpuPanel(screen, right_area);

    // Status bar
    widgets.renderStatusBar(screen, status_area, " q:quit  r:refresh", "ABI Framework ", status_style);
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

fn renderFeaturePanel(screen: *Screen, area: Rect) void {
    widgets.renderPanel(screen, area, " Features ", header_style);
    const flags = getFeatureFlags();
    _ = renderFlags(screen, innerRect(area), &flags, 0);
}

fn renderGpuPanel(screen: *Screen, area: Rect) void {
    widgets.renderPanel(screen, area, " GPU Backends ", header_style);
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
    }, "AI Sub-features:", header_style);

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

    // Show cursor on exit
    defer {
        var buf: [64]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        ansi.showCursor(&writer) catch {};
        flushToFd(stdout_fd, buf[0..writer.end]);
    }

    // Initial render
    renderDashboard(&screen);
    try flushScreenToFd(&screen, stdout_fd);

    // Simple polling loop — read one byte at a time
    const stdin_fd = std.posix.STDIN_FILENO;
    while (true) {
        var buf: [8]u8 = undefined;
        const bytes_read = std.posix.read(stdin_fd, &buf) catch break;
        if (bytes_read == 0) continue;

        const key = events_mod.EventReader.parseKey(buf[0]);
        switch (key) {
            .char => |c| {
                if (c == 'q' or c == 'Q') break;
            },
            .ctrl => |c| {
                if (c == 'c') break; // Ctrl+C
            },
            .escape => break,
            else => {},
        }

        // Re-render on any key
        renderDashboard(&screen);
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

test "getFeatureFlags returns 20 entries" {
    const flags = getFeatureFlags();
    try std.testing.expectEqual(@as(usize, 20), flags.len);
}

test "getGpuFlags returns 4 entries" {
    const flags = getGpuFlags();
    try std.testing.expectEqual(@as(usize, 4), flags.len);
}

test "renderDashboard does not crash" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();
    renderDashboard(&screen);
    // Verify some content was written
    try std.testing.expect(screen.back[0].char != ' ' or screen.back[1].char != ' ');
}

test {
    std.testing.refAllDecls(@This());
}
