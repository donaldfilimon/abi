//! Interactive framework dashboard.
//!
//! Presents a developer-focused diagnostics shell for the ABI framework with
//! catalog-driven feature data, runtime status, and resize-aware layouts.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const feature_catalog = @import("../../../core/feature_catalog.zig");
const platform = @import("../../../platform/mod.zig");
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

var sigwinch_received: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

const title_style = Style{ .fg = .cyan, .bold = true };
const accent_style = Style{ .fg = .cyan };
const body_style = Style{ .fg = .white };
const muted_style = Style{ .fg = .bright_black };
const ok_style = Style{ .fg = .green, .bold = true };
const off_style = Style{ .fg = .red };
const row_highlight_style = Style{ .bg = .bright_black };
const nav_selected_style = Style{ .fg = .black, .bg = .cyan, .bold = true };
const status_style = Style{ .fg = .black, .bg = .white };
const help_panel_style = Style{ .fg = .cyan, .bold = true };

const NavItem = struct {
    view: View,
    label: []const u8,
    compact_label: []const u8,
    blurb: []const u8,
};

const nav_items = [_]NavItem{
    .{
        .view = .overview,
        .label = "Overview",
        .compact_label = "OVR",
        .blurb = "Version, platform, shell status",
    },
    .{
        .view = .features,
        .label = "Features",
        .compact_label = "FEAT",
        .blurb = "Catalog, compile flags, hierarchy",
    },
    .{
        .view = .runtime,
        .label = "Runtime",
        .compact_label = "RUN",
        .blurb = "GPU, protocols, services",
    },
};

const RuntimeEntry = struct {
    group: []const u8,
    name: []const u8,
    enabled: bool,
    detail: []const u8,
};

const runtime_entries = [_]RuntimeEntry{
    .{ .group = "GPU", .name = "metal", .enabled = build_options.gpu_metal, .detail = "Apple Metal backend" },
    .{ .group = "GPU", .name = "cuda", .enabled = build_options.gpu_cuda, .detail = "NVIDIA CUDA backend" },
    .{ .group = "GPU", .name = "vulkan", .enabled = build_options.gpu_vulkan, .detail = "Cross-platform Vulkan backend" },
    .{ .group = "GPU", .name = "webgpu", .enabled = build_options.gpu_webgpu, .detail = "WebGPU runtime bridge" },
    .{ .group = "GPU", .name = "opengl", .enabled = build_options.gpu_opengl, .detail = "Desktop OpenGL backend" },
    .{ .group = "GPU", .name = "opengles", .enabled = build_options.gpu_opengles, .detail = "OpenGL ES backend" },
    .{ .group = "GPU", .name = "webgl2", .enabled = build_options.gpu_webgl2, .detail = "WebGL2 browser backend" },
    .{ .group = "GPU", .name = "stdgpu", .enabled = build_options.gpu_stdgpu, .detail = "stdgpu experimental backend" },
    .{ .group = "GPU", .name = "fpga", .enabled = build_options.gpu_fpga, .detail = "FPGA accelerator path" },
    .{ .group = "GPU", .name = "tpu", .enabled = build_options.gpu_tpu, .detail = "TPU accelerator path" },
    .{ .group = "SERV", .name = "connectors", .enabled = build_options.feat_connectors, .detail = "LLM provider and external service adapters" },
    .{ .group = "SERV", .name = "tasks", .enabled = build_options.feat_tasks, .detail = "Task management and async job queues" },
    .{ .group = "SERV", .name = "inference", .enabled = build_options.feat_inference, .detail = "Inference engines, schedulers, and samplers" },
    .{ .group = "SERV", .name = "lsp", .enabled = build_options.feat_lsp, .detail = "Language Server Protocol surface" },
    .{ .group = "SERV", .name = "mcp", .enabled = build_options.feat_mcp, .detail = "Model Context Protocol surface" },
    .{ .group = "SERV", .name = "acp", .enabled = build_options.feat_acp, .detail = "Agent Communication Protocol surface" },
    .{ .group = "SERV", .name = "ha", .enabled = build_options.feat_ha, .detail = "High availability and replication surface" },
    .{ .group = "SERV", .name = "tui", .enabled = build_options.feat_tui, .detail = "Terminal UI feature gate" },
};

pub const View = enum {
    overview,
    features,
    runtime,
};

pub const FocusRegion = enum {
    nav,
    detail,
};

pub const LayoutMode = enum {
    wide,
    medium,
    compact,
    minimal,
};

pub const AppState = struct {
    current_view: View = .overview,
    focused_region: FocusRegion = .nav,
    nav_index: usize = 0,
    selected_row: usize = 0,
    detail_scroll: usize = 0,
    help_visible: bool = false,
};

pub const DashboardLayout = struct {
    mode: LayoutMode,
    full: Rect,
    header: Rect,
    nav: Rect,
    summary: Rect,
    detail: Rect,
    footer: Rect,
    overlay: Rect,
};

pub const DashboardAction = enum {
    none,
    quit,
};

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

fn emptyRect() Rect {
    return .{};
}

fn insetRect(area: Rect, pad_x: u16, pad_y: u16) Rect {
    return .{
        .x = area.x + @min(pad_x, area.width),
        .y = area.y + @min(pad_y, area.height),
        .width = area.width -| (@min(pad_x, area.width) * 2),
        .height = area.height -| (@min(pad_y, area.height) * 2),
    };
}

fn centeredRect(area: Rect, width: u16, height: u16) Rect {
    const w = @min(width, area.width);
    const h = @min(height, area.height);
    return .{
        .x = area.x + (area.width -| w) / 2,
        .y = area.y + (area.height -| h) / 2,
        .width = w,
        .height = h,
    };
}

fn rowRect(area: Rect, row: u16) Rect {
    if (row >= area.height) return emptyRect();
    return .{
        .x = area.x,
        .y = area.y + row,
        .width = area.width,
        .height = 1,
    };
}

fn classifyLayout(full: Rect) LayoutMode {
    if (full.width < 18 or full.height < 7) return .minimal;
    if (full.width < 52 or full.height < 14) return .compact;
    if (full.width < 96 or full.height < 24) return .medium;
    return .wide;
}

fn viewLabel(view: View) []const u8 {
    return switch (view) {
        .overview => "Overview",
        .features => "Features",
        .runtime => "Runtime",
    };
}

fn modeLabel(mode: LayoutMode) []const u8 {
    return switch (mode) {
        .wide => "wide",
        .medium => "medium",
        .compact => "compact",
        .minimal => "minimal",
    };
}

fn focusLabel(region: FocusRegion) []const u8 {
    return switch (region) {
        .nav => "nav",
        .detail => "detail",
    };
}

fn boolLabel(enabled: bool) []const u8 {
    return if (enabled) "enabled" else "disabled";
}

fn boolIndicator(enabled: bool) []const u8 {
    return if (enabled) "[+]" else "[-]";
}

fn styleForState(enabled: bool, highlighted: bool) Style {
    var style = if (enabled) ok_style else off_style;
    if (highlighted) style.bg = .bright_black;
    return style;
}

fn textStyle(highlighted: bool) Style {
    var style = body_style;
    if (highlighted) style.bg = .bright_black;
    return style;
}

fn navIndexForView(view: View) usize {
    inline for (nav_items, 0..) |item, idx| {
        if (item.view == view) return idx;
    }
    return 0;
}

fn selectedItemCount(view: View) usize {
    return switch (view) {
        .overview => 0,
        .features => feature_catalog.feature_count,
        .runtime => runtime_entries.len,
    };
}

fn enabledFeatureCount() usize {
    const count = comptime blk: {
        var enabled: usize = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) enabled += 1;
        }
        break :blk enabled;
    };
    return count;
}

fn enabledRuntimeCount(group: []const u8) usize {
    var count: usize = 0;
    for (runtime_entries) |entry| {
        if (std.mem.eql(u8, entry.group, group) and entry.enabled) count += 1;
    }
    return count;
}

fn featureEnabled(entry: feature_catalog.Metadata) bool {
    return compileFlagEnabled(entry.compile_flag_field);
}

fn compileFlagEnabled(flag: []const u8) bool {
    const FlagState = struct {
        name: []const u8,
        enabled: bool,
    };

    inline for (.{
        FlagState{ .name = "feat_gpu", .enabled = build_options.feat_gpu },
        FlagState{ .name = "feat_ai", .enabled = build_options.feat_ai },
        FlagState{ .name = "feat_llm", .enabled = build_options.feat_llm },
        FlagState{ .name = "feat_training", .enabled = build_options.feat_training },
        FlagState{ .name = "feat_explore", .enabled = build_options.feat_explore },
        FlagState{ .name = "feat_reasoning", .enabled = build_options.feat_reasoning },
        FlagState{ .name = "feat_vision", .enabled = build_options.feat_vision },
        FlagState{ .name = "feat_database", .enabled = build_options.feat_database },
        FlagState{ .name = "feat_network", .enabled = build_options.feat_network },
        FlagState{ .name = "feat_observability", .enabled = build_options.feat_observability },
        FlagState{ .name = "feat_web", .enabled = build_options.feat_web },
        FlagState{ .name = "feat_cloud", .enabled = build_options.feat_cloud },
        FlagState{ .name = "feat_analytics", .enabled = build_options.feat_analytics },
        FlagState{ .name = "feat_auth", .enabled = build_options.feat_auth },
        FlagState{ .name = "feat_messaging", .enabled = build_options.feat_messaging },
        FlagState{ .name = "feat_cache", .enabled = build_options.feat_cache },
        FlagState{ .name = "feat_storage", .enabled = build_options.feat_storage },
        FlagState{ .name = "feat_search", .enabled = build_options.feat_search },
        FlagState{ .name = "feat_mobile", .enabled = build_options.feat_mobile },
        FlagState{ .name = "feat_gateway", .enabled = build_options.feat_gateway },
        FlagState{ .name = "feat_pages", .enabled = build_options.feat_pages },
        FlagState{ .name = "feat_benchmarks", .enabled = build_options.feat_benchmarks },
        FlagState{ .name = "feat_compute", .enabled = build_options.feat_compute },
        FlagState{ .name = "feat_documents", .enabled = build_options.feat_documents },
        FlagState{ .name = "feat_desktop", .enabled = build_options.feat_desktop },
        FlagState{ .name = "feat_tui", .enabled = build_options.feat_tui },
        FlagState{ .name = "feat_lsp", .enabled = build_options.feat_lsp },
        FlagState{ .name = "feat_mcp", .enabled = build_options.feat_mcp },
        FlagState{ .name = "feat_acp", .enabled = build_options.feat_acp },
        FlagState{ .name = "feat_ha", .enabled = build_options.feat_ha },
        FlagState{ .name = "feat_connectors", .enabled = build_options.feat_connectors },
        FlagState{ .name = "feat_tasks", .enabled = build_options.feat_tasks },
        FlagState{ .name = "feat_inference", .enabled = build_options.feat_inference },
    }) |entry_flag| {
        if (std.mem.eql(u8, flag, entry_flag.name)) return entry_flag.enabled;
    }

    return false;
}

fn selectedFeature(state: *const AppState) feature_catalog.Metadata {
    return feature_catalog.all[@min(state.selected_row, feature_catalog.feature_count - 1)];
}

fn selectedRuntime(state: *const AppState) RuntimeEntry {
    return runtime_entries[@min(state.selected_row, runtime_entries.len - 1)];
}

fn clampState(state: *AppState) void {
    if (state.nav_index >= nav_items.len) state.nav_index = nav_items.len - 1;
    const count = selectedItemCount(state.current_view);
    if (count == 0) {
        state.selected_row = 0;
        state.detail_scroll = 0;
        return;
    }
    if (state.selected_row >= count) state.selected_row = count - 1;
}

fn activateNavSelection(state: *AppState) void {
    state.current_view = nav_items[state.nav_index].view;
    state.selected_row = 0;
    state.detail_scroll = 0;
    state.focused_region = if (state.current_view == .overview) .nav else .detail;
}

fn moveNavSelection(state: *AppState, delta: i32) void {
    const current: i32 = @intCast(state.nav_index);
    const max_index: i32 = @intCast(nav_items.len - 1);
    const next = std.math.clamp(current + delta, 0, max_index);
    state.nav_index = @intCast(next);
}

fn moveDetailSelection(state: *AppState, delta: i32) void {
    const count = selectedItemCount(state.current_view);
    if (count == 0) return;
    const current: i32 = @intCast(state.selected_row);
    const max_index: i32 = @intCast(count - 1);
    const next = std.math.clamp(current + delta, 0, max_index);
    state.selected_row = @intCast(next);
    state.detail_scroll = 0;
}

fn scrollDetail(state: *AppState, delta: i32) void {
    const current: i32 = @intCast(state.detail_scroll);
    const next = std.math.clamp(current + delta, 0, 64);
    state.detail_scroll = @intCast(next);
}

pub fn computeLayout(full: Rect) DashboardLayout {
    const mode = classifyLayout(full);
    if (mode == .minimal) {
        return .{
            .mode = mode,
            .full = full,
            .header = full,
            .nav = emptyRect(),
            .summary = emptyRect(),
            .detail = emptyRect(),
            .footer = emptyRect(),
            .overlay = full,
        };
    }

    const header_height: u16 = if (mode == .compact) @min(@as(u16, 3), full.height) else @min(@as(u16, 4), full.height);
    const top_split = full.splitHorizontal(header_height);
    const footer_height: u16 = if (top_split.bottom.height > 0) 1 else 0;
    const body_and_footer = top_split.bottom.splitHorizontal(top_split.bottom.height -| footer_height);
    const body = body_and_footer.top;
    const footer = if (footer_height == 0) emptyRect() else body_and_footer.bottom;

    return switch (mode) {
        .wide => blk: {
            const nav_width = @min(@as(u16, 24), @max(@as(u16, 18), body.width / 4));
            const nav_split = body.splitVertical(nav_width);
            const summary_height = @min(@as(u16, 6), @max(@as(u16, 5), body.height / 4));
            const summary_split = nav_split.right.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.left,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = centeredRect(full, full.width -| 10, full.height -| 6),
            };
        },
        .medium => blk: {
            const nav_width = @min(@as(u16, 20), @max(@as(u16, 16), body.width / 4));
            const nav_split = body.splitVertical(nav_width);
            const summary_height = @min(@as(u16, 5), @max(@as(u16, 4), body.height / 4));
            const summary_split = nav_split.right.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.left,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = centeredRect(full, full.width -| 6, full.height -| 4),
            };
        },
        .compact => blk: {
            const nav_height: u16 = if (body.height >= 4) 2 else 1;
            const nav_split = body.splitHorizontal(nav_height);
            const summary_height: u16 = if (nav_split.bottom.height > 1) 1 else 0;
            const summary_split = nav_split.bottom.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.top,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = insetRect(full, 1, 1),
            };
        },
        .minimal => unreachable,
    };
}

pub fn handleKey(state: *AppState, key: Key) DashboardAction {
    switch (key) {
        .char => |c| switch (c) {
            'q', 'Q' => return .quit,
            '?' => {
                state.help_visible = !state.help_visible;
                return .none;
            },
            'g' => {
                state.current_view = .overview;
                state.nav_index = navIndexForView(.overview);
                state.selected_row = 0;
                state.detail_scroll = 0;
                state.focused_region = .nav;
                return .none;
            },
            'j' => {
                if (state.help_visible) return .none;
                if (state.focused_region == .nav) {
                    moveNavSelection(state, 1);
                } else {
                    moveDetailSelection(state, 1);
                }
                return .none;
            },
            'k' => {
                if (state.help_visible) return .none;
                if (state.focused_region == .nav) {
                    moveNavSelection(state, -1);
                } else {
                    moveDetailSelection(state, -1);
                }
                return .none;
            },
            else => {},
        },
        .ctrl => |c| {
            if (c == 'c') return .quit;
        },
        .tab => {
            if (!state.help_visible) {
                state.focused_region = if (state.focused_region == .nav) .detail else .nav;
            }
        },
        .enter => {
            if (state.help_visible) {
                state.help_visible = false;
                return .none;
            }
            if (state.focused_region == .nav) {
                activateNavSelection(state);
            } else {
                state.detail_scroll = 0;
            }
        },
        .escape => {
            if (state.help_visible) {
                state.help_visible = false;
                return .none;
            }
            return .quit;
        },
        .left => {
            if (!state.help_visible) state.focused_region = .nav;
        },
        .right => {
            if (!state.help_visible and state.current_view != .overview) state.focused_region = .detail;
        },
        .up => {
            if (state.help_visible) return .none;
            if (state.focused_region == .nav) {
                moveNavSelection(state, -1);
            } else {
                moveDetailSelection(state, -1);
            }
        },
        .down => {
            if (state.help_visible) return .none;
            if (state.focused_region == .nav) {
                moveNavSelection(state, 1);
            } else {
                moveDetailSelection(state, 1);
            }
        },
        .page_up => {
            if (!state.help_visible and state.focused_region == .detail) scrollDetail(state, -2);
        },
        .page_down => {
            if (!state.help_visible and state.focused_region == .detail) scrollDetail(state, 2);
        },
        else => {},
    }

    clampState(state);
    return .none;
}

pub fn hasVisibleCell(cells: []const types.Cell) bool {
    for (cells) |cell| {
        if (cell.char != ' ') return true;
    }
    return false;
}

pub fn containsText(cells: []const types.Cell, text: []const u8) bool {
    if (text.len == 0) return true;
    if (cells.len < text.len) return false;

    var start: usize = 0;
    while (start + text.len <= cells.len) : (start += 1) {
        var matched = true;
        for (text, 0..) |byte, idx| {
            if (cells[start + idx].char != byte) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}

fn fillRect(screen: *Screen, area: Rect, style: Style) void {
    var row: u16 = 0;
    while (row < area.height) : (row += 1) {
        var col: u16 = 0;
        while (col < area.width) : (col += 1) {
            screen.setCell(area.x + col, area.y + row, .{ .char = ' ', .style = style });
        }
    }
}

fn drawHorizontalDivider(screen: *Screen, area: Rect, y: u16, style: Style) void {
    if (y >= area.height) return;
    var col: u16 = 0;
    while (col < area.width) : (col += 1) {
        screen.setCell(area.x + col, area.y + y, .{ .char = 0x2500, .style = style });
    }
}

fn drawVerticalDivider(screen: *Screen, x: u16, area: Rect, style: Style) void {
    if (x >= area.x + area.width) return;
    var row: u16 = 0;
    while (row < area.height) : (row += 1) {
        screen.setCell(x, area.y + row, .{ .char = 0x2502, .style = style });
    }
}

fn renderCenteredText(screen: *Screen, area: Rect, row: u16, text: []const u8, style: Style) void {
    if (row >= area.height or area.width == 0) return;
    const text_len: u16 = @intCast(@min(text.len, @as(usize, area.width)));
    const start_x = area.x + (area.width -| text_len) / 2;
    widgets.renderText(screen, .{
        .x = start_x,
        .y = area.y + row,
        .width = text_len,
        .height = 1,
    }, text[0..text_len], style);
}

fn renderSectionTitle(screen: *Screen, area: Rect, title: []const u8, style: Style) void {
    if (area.width == 0 or area.height == 0) return;
    widgets.renderText(screen, rowRect(area, 0), title, style);
}

fn renderPlainLine(screen: *Screen, area: Rect, row: *u16, text: []const u8, style: Style) void {
    if (row.* >= area.height) return;
    widgets.renderText(screen, rowRect(area, row.*), text, style);
    row.* += 1;
}

fn renderKeyValue(screen: *Screen, area: Rect, row: *u16, label: []const u8, value: []const u8, value_style: Style) void {
    if (row.* >= area.height) return;

    var buf: [160]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{s}: {s}", .{ label, value }) catch value;
    widgets.renderText(screen, rowRect(area, row.*), line, value_style);
    row.* += 1;
}

fn renderScrolledLine(screen: *Screen, area: Rect, row: *u16, line_index: *usize, scroll: usize, text: []const u8, style: Style) void {
    const should_render = line_index.* >= scroll and row.* < area.height;
    if (should_render) {
        widgets.renderText(screen, rowRect(area, row.*), text, style);
        row.* += 1;
    }
    line_index.* += 1;
}

fn renderScrolledKeyValue(screen: *Screen, area: Rect, row: *u16, line_index: *usize, scroll: usize, label: []const u8, value: []const u8, style: Style) void {
    var buf: [192]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{s}: {s}", .{ label, value }) catch value;
    renderScrolledLine(screen, area, row, line_index, scroll, line, style);
}

fn visibleWindow(selected: usize, available_rows: usize, total_rows: usize) usize {
    if (available_rows == 0 or total_rows <= available_rows) return 0;
    const half = available_rows / 2;
    if (selected <= half) return 0;
    const max_start = total_rows - available_rows;
    return @min(selected - half, max_start);
}

fn compactLabel(item: NavItem, width: u16) []const u8 {
    return if (width < 28) item.compact_label else item.label;
}

fn renderHeader(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode) void {
    if (area.height == 0) return;

    renderCenteredText(screen, area, 0, "ABI DIAGNOSTIC SHELL", title_style);

    var subtitle_buf: [128]u8 = undefined;
    const subtitle = std.fmt.bufPrint(&subtitle_buf, "{s} | {s}", .{
        viewLabel(state.current_view),
        platform.getDescription(),
    }) catch viewLabel(state.current_view);
    if (area.height > 1) {
        renderCenteredText(screen, area, 1, subtitle, muted_style);
    }

    if (area.height > 2) {
        var detail_buf: [128]u8 = undefined;
        const detail = std.fmt.bufPrint(&detail_buf, "{d}/{d} features enabled | {s} layout", .{
            enabledFeatureCount(),
            feature_catalog.feature_count,
            modeLabel(mode),
        }) catch "Diagnostics";
        renderCenteredText(screen, area, 2, detail, body_style);
    }

    drawHorizontalDivider(screen, area, area.height - 1, muted_style);
}

fn renderNav(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        const rows = insetRect(area, 0, 0);
        const segment_width = @max(@as(u16, 1), rows.width / @as(u16, nav_items.len));
        for (nav_items, 0..) |item, idx| {
            const x = rows.x + @as(u16, @intCast(idx)) * segment_width;
            const width = if (idx == nav_items.len - 1) rows.width -| (@as(u16, @intCast(idx)) * segment_width) else segment_width;
            const segment = Rect{ .x = x, .y = rows.y, .width = width, .height = 1 };
            const is_selected = idx == state.nav_index;
            const is_current = item.view == state.current_view;
            const style = if (is_selected and state.focused_region == .nav)
                nav_selected_style
            else if (is_current)
                title_style
            else
                body_style;
            if (style.bg != .default) fillRect(screen, segment, style);
            renderCenteredText(screen, segment, 0, compactLabel(item, rows.width), style);
        }

        if (rows.height > 1) {
            var hint_buf: [80]u8 = undefined;
            const hint = std.fmt.bufPrint(&hint_buf, "focus {s} | enter open", .{focusLabel(state.focused_region)}) catch "focus";
            widgets.renderText(screen, rowRect(rows, 1), hint, muted_style);
        }

        drawHorizontalDivider(screen, area, area.height - 1, muted_style);
        return;
    }

    if (area.width > 0) {
        drawVerticalDivider(screen, area.x + area.width - 1, area, muted_style);
    }

    const content = insetRect(.{
        .x = area.x,
        .y = area.y,
        .width = area.width -| 1,
        .height = area.height,
    }, 1, 0);
    renderSectionTitle(screen, content, "VIEWS", if (state.focused_region == .nav) title_style else accent_style);

    const show_blurbs = content.height >= nav_items.len * 2 + 3;
    var row: u16 = 1;
    for (nav_items, 0..) |item, idx| {
        if (row >= content.height) break;
        const is_selected = idx == state.nav_index;
        const is_current = item.view == state.current_view;
        const style = if (is_selected and state.focused_region == .nav)
            nav_selected_style
        else if (is_current)
            title_style
        else if (is_selected)
            textStyle(true)
        else
            body_style;

        const marker = if (is_current) "*" else " ";
        const cursor = if (is_selected) ">" else " ";
        var line_buf: [96]u8 = undefined;
        const line = std.fmt.bufPrint(&line_buf, "{s}{s} {s}", .{ marker, cursor, item.label }) catch item.label;
        if (style.bg != .default) fillRect(screen, rowRect(content, row), style);
        widgets.renderText(screen, rowRect(content, row), line, style);
        row += 1;

        if (show_blurbs and row < content.height) {
            const blurb_style = if (is_selected and state.focused_region == .nav) style else muted_style;
            widgets.renderText(screen, rowRect(content, row), item.blurb, blurb_style);
            row += 1;
        }
    }

    if (content.height >= 2) {
        widgets.renderText(screen, rowRect(content, content.height - 1), "tab focus  enter open", muted_style);
    }
}

fn renderOverviewSummary(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;
    renderSectionTitle(screen, area, "OVERVIEW", title_style);
    var row: u16 = 1;

    var term_buf: [128]u8 = undefined;
    const term_line = std.fmt.bufPrint(&term_buf, "v{s} | {d}x{d} terminal | focus {s} | {s}", .{
        build_options.package_version,
        screen_width,
        screen_height,
        focusLabel(state.focused_region),
        modeLabel(mode),
    }) catch "terminal";
    renderPlainLine(screen, area, &row, term_line, body_style);

    var count_buf: [96]u8 = undefined;
    const count_line = std.fmt.bufPrint(&count_buf, "{d}/{d} catalog features | {d}/10 GPU | {d}/8 services", .{
        enabledFeatureCount(),
        feature_catalog.feature_count,
        enabledRuntimeCount("GPU"),
        enabledRuntimeCount("SERV"),
    }) catch "counts";
    renderPlainLine(screen, area, &row, count_line, body_style);
}

fn renderFeaturesSummary(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;
    const entry = selectedFeature(state);
    renderSectionTitle(screen, area, "FEATURES", title_style);
    var row: u16 = 1;

    var label_buf: [96]u8 = undefined;
    const label = std.fmt.bufPrint(&label_buf, "{s}", .{entry.feature.name()}) catch entry.feature.name();
    renderPlainLine(screen, area, &row, label, body_style);

    var meta_buf: [128]u8 = undefined;
    const meta_line = std.fmt.bufPrint(&meta_buf, "{d}/{d} selected | {s}", .{
        state.selected_row + 1,
        feature_catalog.feature_count,
        boolLabel(featureEnabled(entry)),
    }) catch "selected";
    renderPlainLine(screen, area, &row, meta_line, styleForState(featureEnabled(entry), false));

    widgets.renderText(screen, rowRect(area, row), entry.description, muted_style);
}

fn renderRuntimeSummary(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;
    const entry = selectedRuntime(state);
    renderSectionTitle(screen, area, "RUNTIME", title_style);
    var row: u16 = 1;

    var selection_buf: [96]u8 = undefined;
    const selection = std.fmt.bufPrint(&selection_buf, "{s} {s}", .{ entry.group, entry.name }) catch entry.name;
    renderPlainLine(screen, area, &row, selection, body_style);

    var count_buf: [96]u8 = undefined;
    const count_line = std.fmt.bufPrint(&count_buf, "{d}/10 GPU | {d}/8 services | {s}", .{
        enabledRuntimeCount("GPU"),
        enabledRuntimeCount("SERV"),
        boolLabel(entry.enabled),
    }) catch "runtime";
    renderPlainLine(screen, area, &row, count_line, styleForState(entry.enabled, false));

    widgets.renderText(screen, rowRect(area, row), entry.detail, muted_style);
}

fn renderSummary(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode != .compact) {
        drawHorizontalDivider(screen, area, area.height - 1, muted_style);
    }

    const divider_rows: u16 = if (mode == .compact) 0 else 1;
    const content: Rect = .{
        .x = area.x,
        .y = area.y,
        .width = area.width,
        .height = area.height -| divider_rows,
    };

    switch (state.current_view) {
        .overview => renderOverviewSummary(screen, content, state, mode, screen_width, screen_height),
        .features => renderFeaturesSummary(screen, content, state),
        .runtime => renderRuntimeSummary(screen, content, state),
    }
}

fn renderFeatureList(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;

    renderSectionTitle(screen, area, "CATALOG", if (state.focused_region == .detail) accent_style else muted_style);
    const content: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    const visible_rows: usize = content.height;
    const start = visibleWindow(state.selected_row, visible_rows, feature_catalog.feature_count);

    for (0..visible_rows) |visible_idx| {
        const item_idx = start + visible_idx;
        if (item_idx >= feature_catalog.feature_count) break;
        const entry = feature_catalog.all[item_idx];
        const selected = item_idx == state.selected_row;
        const row: u16 = @intCast(visible_idx);
        const row_area = rowRect(content, row);
        if (selected) fillRect(screen, row_area, row_highlight_style);

        widgets.renderText(screen, .{
            .x = row_area.x,
            .y = row_area.y,
            .width = @min(@as(u16, 4), row_area.width),
            .height = 1,
        }, boolIndicator(featureEnabled(entry)), styleForState(featureEnabled(entry), selected));

        var label_buf: [96]u8 = undefined;
        const indent = if (entry.parent != null) "  " else "";
        const label = std.fmt.bufPrint(&label_buf, "{s}{s}", .{ indent, entry.feature.name() }) catch entry.feature.name();
        widgets.renderText(screen, .{
            .x = row_area.x + 4,
            .y = row_area.y,
            .width = row_area.width -| 4,
            .height = 1,
        }, label, textStyle(selected));
    }
}

fn renderFeatureInspector(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;

    const entry = selectedFeature(state);
    const enabled = featureEnabled(entry);
    renderSectionTitle(screen, area, "INSPECTOR", if (state.focused_region == .detail) title_style else accent_style);

    var row: u16 = 1;
    var line_index: usize = 0;
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "feature", entry.feature.name(), body_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "state", boolLabel(enabled), styleForState(enabled, false));
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "flag", entry.compile_flag_field, body_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "parent", if (entry.parent) |parent| parent.name() else "root", body_style);
    renderScrolledLine(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, entry.description, muted_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "module", entry.real_module_path, muted_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "stub", entry.stub_module_path, muted_style);
}

fn renderRuntimeList(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;

    renderSectionTitle(screen, area, "SYSTEM", if (state.focused_region == .detail) accent_style else muted_style);
    const content: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    const visible_rows: usize = content.height;
    const start = visibleWindow(state.selected_row, visible_rows, runtime_entries.len);

    for (0..visible_rows) |visible_idx| {
        const item_idx = start + visible_idx;
        if (item_idx >= runtime_entries.len) break;
        const entry = runtime_entries[item_idx];
        const selected = item_idx == state.selected_row;
        const row: u16 = @intCast(visible_idx);
        const row_area = rowRect(content, row);
        if (selected) fillRect(screen, row_area, row_highlight_style);

        widgets.renderText(screen, .{
            .x = row_area.x,
            .y = row_area.y,
            .width = @min(@as(u16, 4), row_area.width),
            .height = 1,
        }, boolIndicator(entry.enabled), styleForState(entry.enabled, selected));

        var label_buf: [96]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "{s} {s}", .{ entry.group, entry.name }) catch entry.name;
        widgets.renderText(screen, .{
            .x = row_area.x + 4,
            .y = row_area.y,
            .width = row_area.width -| 4,
            .height = 1,
        }, label, textStyle(selected));
    }
}

fn renderRuntimeInspector(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width == 0 or area.height == 0) return;

    const entry = selectedRuntime(state);
    renderSectionTitle(screen, area, "DETAIL", if (state.focused_region == .detail) title_style else accent_style);

    var row: u16 = 1;
    var line_index: usize = 0;
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "group", entry.group, body_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "name", entry.name, body_style);
    renderScrolledKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, "state", boolLabel(entry.enabled), styleForState(entry.enabled, false));
    renderScrolledLine(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, &line_index, state.detail_scroll, entry.detail, muted_style);
}

fn renderOverviewDetail(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    renderSectionTitle(screen, area, "DETAIL", title_style);
    var row: u16 = 1;

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "version", build_options.package_version, body_style);

    var size_buf: [64]u8 = undefined;
    const size_line = std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ screen_width, screen_height }) catch "terminal";
    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "terminal", size_line, body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "platform", platform.getDescription(), body_style);

    var cpu_buf: [64]u8 = undefined;
    const cpu_line = std.fmt.bufPrint(&cpu_buf, "{d}", .{platform.getCpuCount()}) catch "1";
    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "cpu", cpu_line, body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "threading", if (platform.supportsThreading()) "supported" else "disabled", body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "gpu ready", if (platform.isGpuAvailable()) "possible" else "off", body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "network", if (platform.hasNetworkAccess()) "available" else "restricted", body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "filesystem", if (platform.hasFileSystem()) "available" else "restricted", body_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "layout", modeLabel(mode), muted_style);

    renderKeyValue(screen, .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    }, &row, "help", if (state.help_visible) "visible" else "hidden", muted_style);
}

fn renderFeaturesDetail(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        renderFeatureInspector(screen, area, state);
        return;
    }

    const split = layout_mod.split(area, .vertical, &.{
        .{ .percentage = 46 },
        .{ .min = 20 },
    });
    const list_area = if (split.len > 0) split.rects[0] else area;
    const detail_area = if (split.len > 1) split.rects[1] else area;

    renderFeatureList(screen, insetRect(.{
        .x = list_area.x,
        .y = list_area.y,
        .width = list_area.width -| 1,
        .height = list_area.height,
    }, 0, 0), state);
    if (list_area.width > 0) {
        drawVerticalDivider(screen, list_area.x + list_area.width -| 1, list_area, muted_style);
    }
    renderFeatureInspector(screen, insetRect(detail_area, 1, 0), state);
}

fn renderRuntimeDetail(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        renderRuntimeInspector(screen, area, state);
        return;
    }

    const split = layout_mod.split(area, .vertical, &.{
        .{ .percentage = 46 },
        .{ .min = 18 },
    });
    const list_area = if (split.len > 0) split.rects[0] else area;
    const detail_area = if (split.len > 1) split.rects[1] else area;

    renderRuntimeList(screen, insetRect(.{
        .x = list_area.x,
        .y = list_area.y,
        .width = list_area.width -| 1,
        .height = list_area.height,
    }, 0, 0), state);
    if (list_area.width > 0) {
        drawVerticalDivider(screen, list_area.x + list_area.width -| 1, list_area, muted_style);
    }
    renderRuntimeInspector(screen, insetRect(detail_area, 1, 0), state);
}

fn renderDetail(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    switch (state.current_view) {
        .overview => renderOverviewDetail(screen, area, state, mode, screen_width, screen_height),
        .features => renderFeaturesDetail(screen, area, state, mode),
        .runtime => renderRuntimeDetail(screen, area, state, mode),
    }
}

fn renderFooter(screen: *Screen, area: Rect, state: *const AppState, mode: LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    const left = if (state.help_visible)
        "? close help  q quit"
    else if (state.focused_region == .nav)
        "tab focus  enter open  j/k move  ? help  q quit"
    else
        "tab focus  j/k move  enter reset  g overview  ? help  q quit";

    var right_buf: [64]u8 = undefined;
    const right = std.fmt.bufPrint(&right_buf, "{s} | {s}", .{
        viewLabel(state.current_view),
        modeLabel(mode),
    }) catch viewLabel(state.current_view);

    widgets.renderStatusBar(screen, area, left, right, status_style);
}

fn renderHelpOverlay(screen: *Screen, area: Rect, state: *const AppState) void {
    if (area.width < 4 or area.height < 4) return;

    fillRect(screen, area, Style{ .bg = .black });
    widgets.renderPanel(screen, area, " Help ", help_panel_style);

    const content = insetRect(area, 2, 1);
    var row: u16 = 0;

    renderPlainLine(screen, content, &row, "Developer diagnostics shell", title_style);
    renderPlainLine(screen, content, &row, "tab: cycle nav/detail focus", body_style);
    renderPlainLine(screen, content, &row, "j/k or arrows: move selection", body_style);
    renderPlainLine(screen, content, &row, "enter: open nav selection", body_style);
    renderPlainLine(screen, content, &row, "g: return to overview", body_style);
    renderPlainLine(screen, content, &row, "page up/down: scroll detail", body_style);
    renderPlainLine(screen, content, &row, "? or esc: close help", body_style);
    renderPlainLine(screen, content, &row, "q or ctrl-c: quit", body_style);

    if (content.height > row + 1) {
        var hint_buf: [96]u8 = undefined;
        const hint = std.fmt.bufPrint(&hint_buf, "current: {s} | focus: {s}", .{
            viewLabel(state.current_view),
            focusLabel(state.focused_region),
        }) catch "state";
        widgets.renderText(screen, rowRect(content, content.height - 1), hint, muted_style);
    }
}

fn renderMinimal(screen: *Screen, area: Rect) void {
    if (area.width == 0 or area.height == 0) return;

    screen.clear();
    renderCenteredText(screen, area, 0, "ABI", title_style);
    if (area.height > 1) renderCenteredText(screen, area, 1, "dashboard", body_style);
    if (area.height > 2) renderCenteredText(screen, area, 2, "grow terminal", muted_style);
    if (area.height > 3) renderCenteredText(screen, area, 3, "? help  q quit", muted_style);
}

pub fn renderDashboard(screen: *Screen, state: *const AppState) void {
    screen.clear();

    var render_state = state.*;
    clampState(&render_state);

    const dl = computeLayout(screen.rect());
    if (dl.mode == .minimal) {
        renderMinimal(screen, dl.full);
        if (render_state.help_visible) renderHelpOverlay(screen, dl.overlay, &render_state);
        return;
    }

    renderHeader(screen, dl.header, &render_state, dl.mode);
    renderNav(screen, dl.nav, &render_state, dl.mode);
    renderSummary(screen, dl.summary, &render_state, dl.mode, screen.width, screen.height);
    renderDetail(screen, dl.detail, &render_state, dl.mode, screen.width, screen.height);
    renderFooter(screen, dl.footer, &render_state, dl.mode);

    if (render_state.help_visible) {
        renderHelpOverlay(screen, dl.overlay, &render_state);
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
            const action = handleKey(&app_state, event.key);
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

test "feature catalog count stays aligned with dashboard" {
    try std.testing.expectEqual(feature_catalog.feature_count, feature_catalog.all.len);
    try std.testing.expectEqual(feature_catalog.feature_count, selectedItemCount(.features));
}

test "runtime entries stay aligned with dashboard" {
    var gpu_entries: usize = 0;
    var service_entries: usize = 0;
    for (runtime_entries) |entry| {
        if (std.mem.eql(u8, entry.group, "GPU")) gpu_entries += 1;
        if (std.mem.eql(u8, entry.group, "SERV")) service_entries += 1;
    }

    try std.testing.expectEqual(@as(usize, 18), runtime_entries.len);
    try std.testing.expectEqual(@as(usize, 10), gpu_entries);
    try std.testing.expectEqual(@as(usize, 8), service_entries);
}

test "computeLayout classifies breakpoints" {
    try std.testing.expectEqual(LayoutMode.medium, computeLayout(.{ .x = 0, .y = 0, .width = 80, .height = 24 }).mode);
    try std.testing.expectEqual(LayoutMode.compact, computeLayout(.{ .x = 0, .y = 0, .width = 40, .height = 12 }).mode);
    try std.testing.expectEqual(LayoutMode.compact, computeLayout(.{ .x = 0, .y = 0, .width = 20, .height = 8 }).mode);
    try std.testing.expectEqual(LayoutMode.minimal, computeLayout(.{ .x = 0, .y = 0, .width = 17, .height = 6 }).mode);
}

test "handleKey tab cycles focus and enter opens selected view" {
    var state = AppState{};
    try std.testing.expectEqual(FocusRegion.nav, state.focused_region);

    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .tab));
    try std.testing.expectEqual(FocusRegion.detail, state.focused_region);

    state.focused_region = .nav;
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .down));
    try std.testing.expectEqual(@as(usize, 1), state.nav_index);
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .enter));
    try std.testing.expectEqual(View.features, state.current_view);
    try std.testing.expectEqual(FocusRegion.detail, state.focused_region);
}

test "handleKey supports help and overview reset" {
    var state = AppState{ .current_view = .runtime, .nav_index = navIndexForView(.runtime), .focused_region = .detail };
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, Key{ .char = '?' }));
    try std.testing.expect(state.help_visible);
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .escape));
    try std.testing.expect(!state.help_visible);
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, Key{ .char = 'g' }));
    try std.testing.expectEqual(View.overview, state.current_view);
    try std.testing.expectEqual(FocusRegion.nav, state.focused_region);
}

test "handleKey detail navigation clamps to selection bounds" {
    var state = AppState{
        .current_view = .features,
        .focused_region = .detail,
        .selected_row = feature_catalog.feature_count - 1,
    };

    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .down));
    try std.testing.expectEqual(feature_catalog.feature_count - 1, state.selected_row);
    try std.testing.expectEqual(DashboardAction.none, handleKey(&state, .page_down));
    try std.testing.expect(state.detail_scroll > 0);
}

test "containsText finds contiguous ASCII cells" {
    var cells = [_]types.Cell{.{}} ** 8;
    cells[1] = .{ .char = 'A' };
    cells[2] = .{ .char = 'B' };
    cells[3] = .{ .char = 'I' };
    try std.testing.expect(containsText(&cells, "ABI"));
    try std.testing.expect(!containsText(&cells, "GPU"));
}

test "renderDashboard overview prints shell title" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();

    const state = AppState{};
    renderDashboard(&screen, &state);
    try std.testing.expect(containsText(screen.back, "ABI DIAGNOSTIC SHELL"));
    try std.testing.expect(containsText(screen.back, "Overview"));
}

test "renderDashboard features view renders selected catalog entry" {
    var screen = try Screen.init(std.testing.allocator, 120, 40);
    defer screen.deinit();

    const state = AppState{
        .current_view = .features,
        .focused_region = .detail,
        .nav_index = navIndexForView(.features),
        .selected_row = feature_catalog.feature_count - 1,
    };
    renderDashboard(&screen, &state);
    try std.testing.expect(containsText(screen.back, "inference"));
    try std.testing.expect(containsText(screen.back, "feat_inference"));
}

test "renderDashboard compact features view stays readable" {
    var screen = try Screen.init(std.testing.allocator, 20, 8);
    defer screen.deinit();

    const state = AppState{
        .current_view = .features,
        .nav_index = navIndexForView(.features),
        .focused_region = .detail,
    };
    renderDashboard(&screen, &state);
    try std.testing.expect(hasVisibleCell(screen.back));
    try std.testing.expect(containsText(screen.back, "FEAT"));
}

test "renderDashboard runtime view renders service detail" {
    var screen = try Screen.init(std.testing.allocator, 100, 28);
    defer screen.deinit();

    const state = AppState{
        .current_view = .runtime,
        .nav_index = navIndexForView(.runtime),
        .focused_region = .detail,
        .selected_row = runtime_entries.len - 1,
    };
    renderDashboard(&screen, &state);
    try std.testing.expect(containsText(screen.back, "tui"));
}

test {
    std.testing.refAllDecls(@This());
}
