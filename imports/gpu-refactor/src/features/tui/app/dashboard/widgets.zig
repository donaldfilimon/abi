const std = @import("std");
const build_options = @import("build_options");
const feature_catalog = @import("../../../core/feature_catalog.zig");
const platform = @import("../../../../platform/mod.zig");
const types = @import("../../types.zig");
const render_mod = @import("../../render.zig");
const widgets = @import("../../widgets.zig");
const state_mod = @import("state.zig");
const layout_util = @import("layout.zig");

const Style = types.Style;
const Rect = types.Rect;
const Screen = render_mod.Screen;

pub const title_style = Style{ .fg = .cyan, .bold = true };
pub const accent_style = Style{ .fg = .cyan };
pub const body_style = Style{ .fg = .white };
pub const muted_style = Style{ .fg = .bright_black };
pub const ok_style = Style{ .fg = .green, .bold = true };
pub const off_style = Style{ .fg = .red };
pub const row_highlight_style = Style{ .bg = .bright_black };
pub const nav_selected_style = Style{ .fg = .black, .bg = .cyan, .bold = true };
pub const status_style = Style{ .fg = .black, .bg = .white };
pub const help_panel_style = Style{ .fg = .cyan, .bold = true };

pub fn viewLabel(view: state_mod.View) []const u8 {
    return switch (view) {
        .overview => "Overview",
        .features => "Features",
        .runtime => "Runtime",
    };
}

pub fn modeLabel(mode: state_mod.LayoutMode) []const u8 {
    return switch (mode) {
        .wide => "wide",
        .medium => "medium",
        .compact => "compact",
        .minimal => "minimal",
    };
}

pub fn focusLabel(region: state_mod.FocusRegion) []const u8 {
    return switch (region) {
        .nav => "nav",
        .detail => "detail",
    };
}

pub fn boolLabel(enabled: bool) []const u8 {
    return if (enabled) "enabled" else "disabled";
}

pub fn boolIndicator(enabled: bool) []const u8 {
    return if (enabled) "[+]" else "[-]";
}

pub fn coveragePercent(numerator: usize, denominator: usize) u8 {
    if (denominator == 0) return 0;
    const percent = (numerator * 100) / denominator;
    return @intCast(@min(percent, @as(usize, 100)));
}

pub fn renderMetricGauge(
    screen: *Screen,
    area: Rect,
    row: *u16,
    label: []const u8,
    numerator: usize,
    denominator: usize,
    filled_style: Style,
    empty_style: Style,
) void {
    if (row.* >= area.height) return;

    var buf: [96]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{s} {d}/{d}", .{
        label,
        numerator,
        denominator,
    }) catch label;
    widgets.renderGauge(
        screen,
        layout_util.rowRect(area, row.*),
        coveragePercent(numerator, denominator),
        line,
        filled_style,
        empty_style,
    );
    row.* += 1;
}

pub fn styleForState(enabled: bool, highlighted: bool) Style {
    var style = if (enabled) ok_style else off_style;
    if (highlighted) style.bg = .bright_black;
    return style;
}

pub fn textStyle(highlighted: bool) Style {
    var style = body_style;
    if (highlighted) style.bg = .bright_black;
    return style;
}

pub fn enabledFeatureCount() usize {
    const count = comptime blk: {
        var enabled: usize = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) enabled += 1;
        }
        break :blk enabled;
    };
    return count;
}

pub fn enabledRuntimeCount(group: []const u8) usize {
    var count: usize = 0;
    for (state_mod.runtime_entries) |entry| {
        if (std.mem.eql(u8, entry.group, group) and entry.enabled) count += 1;
    }
    return count;
}

pub fn compileFlagEnabled(flag: []const u8) bool {
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

pub fn featureEnabled(entry: feature_catalog.Metadata) bool {
    return compileFlagEnabled(entry.compile_flag_field);
}

pub fn fillRect(screen: *Screen, area: Rect, style: Style) void {
    var row: u16 = 0;
    while (row < area.height) : (row += 1) {
        var col: u16 = 0;
        while (col < area.width) : (col += 1) {
            screen.setCell(area.x + col, area.y + row, .{ .char = ' ', .style = style });
        }
    }
}

pub fn drawHorizontalDivider(screen: *Screen, area: Rect, y: u16, style: Style) void {
    if (y >= area.height) return;
    var col: u16 = 0;
    while (col < area.width) : (col += 1) {
        screen.setCell(area.x + col, area.y + y, .{ .char = 0x2500, .style = style });
    }
}

pub fn drawVerticalDivider(screen: *Screen, x: u16, area: Rect, style: Style) void {
    if (x >= area.x + area.width) return;
    var row: u16 = 0;
    while (row < area.height) : (row += 1) {
        screen.setCell(x, area.y + row, .{ .char = 0x2502, .style = style });
    }
}

pub fn renderCenteredText(screen: *Screen, area: Rect, row: u16, text: []const u8, style: Style) void {
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

pub fn renderSectionTitle(screen: *Screen, area: Rect, title: []const u8, style: Style) void {
    if (area.width == 0 or area.height == 0) return;
    widgets.renderText(screen, layout_util.rowRect(area, 0), title, style);
}

pub fn renderPlainLine(screen: *Screen, area: Rect, row: *u16, text: []const u8, style: Style) void {
    if (row.* >= area.height) return;
    widgets.renderText(screen, layout_util.rowRect(area, row.*), text, style);
    row.* += 1;
}

pub fn renderKeyValue(screen: *Screen, area: Rect, row: *u16, label: []const u8, value: []const u8, value_style: Style) void {
    if (row.* >= area.height) return;

    var buf: [160]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{s}: {s}", .{ label, value }) catch value;
    widgets.renderText(screen, layout_util.rowRect(area, row.*), line, value_style);
    row.* += 1;
}

pub fn renderScrolledLine(screen: *Screen, area: Rect, row: *u16, line_index: *usize, scroll: usize, text: []const u8, style: Style) void {
    const should_render = line_index.* >= scroll and row.* < area.height;
    if (should_render) {
        widgets.renderText(screen, layout_util.rowRect(area, row.*), text, style);
        row.* += 1;
    }
    line_index.* += 1;
}

pub fn renderScrolledKeyValue(screen: *Screen, area: Rect, row: *u16, line_index: *usize, scroll: usize, label: []const u8, value: []const u8, style: Style) void {
    var buf: [192]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{s}: {s}", .{ label, value }) catch value;
    renderScrolledLine(screen, area, row, line_index, scroll, line, style);
}

pub fn visibleWindow(selected: usize, available_rows: usize, total_rows: usize) usize {
    if (available_rows == 0 or total_rows <= available_rows) return 0;
    const half = available_rows / 2;
    if (selected <= half) return 0;
    const max_start = total_rows - available_rows;
    return @min(selected - half, max_start);
}

pub fn renderHeader(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode) void {
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

pub fn renderNav(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        const rows = layout_util.insetRect(area, 0, 0);
        const segment_width = @max(@as(u16, 1), rows.width / @as(u16, state_mod.nav_items.len));
        for (state_mod.nav_items, 0..) |item, idx| {
            const x = rows.x + @as(u16, @intCast(idx)) * segment_width;
            const width = if (idx == state_mod.nav_items.len - 1) rows.width -| (@as(u16, @intCast(idx)) * segment_width) else segment_width;
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
            renderCenteredText(screen, segment, 0, if (rows.width < 28) item.compact_label else item.label, style);
        }

        if (rows.height > 1) {
            var hint_buf: [80]u8 = undefined;
            const hint = std.fmt.bufPrint(&hint_buf, "focus {s} | enter open", .{focusLabel(state.focused_region)}) catch "focus";
            widgets.renderText(screen, layout_util.rowRect(rows, 1), hint, muted_style);
        }

        drawHorizontalDivider(screen, area, area.height - 1, muted_style);
        return;
    }

    if (area.width > 0) {
        drawVerticalDivider(screen, area.x + area.width - 1, area, muted_style);
    }

    const content = layout_util.insetRect(.{
        .x = area.x,
        .y = area.y,
        .width = area.width -| 1,
        .height = area.height,
    }, 1, 0);
    renderSectionTitle(screen, content, "VIEWS", if (state.focused_region == .nav) title_style else accent_style);

    const show_blurbs = content.height >= state_mod.nav_items.len * 2 + 3;
    var row: u16 = 1;
    for (state_mod.nav_items, 0..) |item, idx| {
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
        if (style.bg != .default) fillRect(screen, layout_util.rowRect(content, row), style);
        widgets.renderText(screen, layout_util.rowRect(content, row), line, style);
        row += 1;

        if (show_blurbs and row < content.height) {
            const blurb_style = if (is_selected and state.focused_region == .nav) style else muted_style;
            widgets.renderText(screen, layout_util.rowRect(content, row), item.blurb, blurb_style);
            row += 1;
        }
    }

    if (content.height >= 2) {
        widgets.renderText(screen, layout_util.rowRect(content, content.height - 1), "tab focus  enter open", muted_style);
    }
}

pub fn renderFooter(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode) void {
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

pub fn renderHelpOverlay(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width < 4 or area.height < 4) return;

    fillRect(screen, area, Style{ .bg = .black });
    widgets.renderPanel(screen, area, " Help ", help_panel_style);

    const content = layout_util.insetRect(area, 2, 1);
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
        widgets.renderText(screen, layout_util.rowRect(content, content.height - 1), hint, muted_style);
    }
}
