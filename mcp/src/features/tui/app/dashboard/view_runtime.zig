const std = @import("std");
const types = @import("../../types.zig");
const render_mod = @import("../../render.zig");
const widgets = @import("../../widgets.zig");
const state_mod = @import("state.zig");
const widget_util = @import("widgets.zig");
const layout_util = @import("layout.zig");

const Rect = types.Rect;
const Screen = render_mod.Screen;

pub fn selectedRuntime(state: *const state_mod.AppState) state_mod.RuntimeEntry {
    return state_mod.runtime_entries[@min(state.selected_row, state_mod.runtime_entries.len - 1)];
}

pub fn renderRuntimeSummary(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;
    const entry = selectedRuntime(state);
    widget_util.renderSectionTitle(screen, area, "RUNTIME", widget_util.title_style);
    var row: u16 = 1;

    var selection_buf: [96]u8 = undefined;
    const selection = std.fmt.bufPrint(&selection_buf, "{s} {s}", .{ entry.group, entry.name }) catch entry.name;
    widget_util.renderPlainLine(screen, area, &row, selection, widget_util.body_style);

    var count_buf: [96]u8 = undefined;
    const count_line = std.fmt.bufPrint(&count_buf, "{d}/10 GPU | {d}/8 services | {s}", .{
        widget_util.enabledRuntimeCount("GPU"),
        widget_util.enabledRuntimeCount("SERV"),
        widget_util.boolLabel(entry.enabled),
    }) catch "runtime";
    widget_util.renderPlainLine(screen, area, &row, count_line, widget_util.styleForState(entry.enabled, false));

    widgets.renderText(screen, layout_util.rowRect(area, row), entry.detail, widget_util.muted_style);
}

pub fn renderRuntimeList(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;

    widget_util.renderSectionTitle(screen, area, "SYSTEM", if (state.focused_region == .detail) widget_util.accent_style else widget_util.muted_style);
    const content: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    const visible_rows: usize = content.height;
    const start = widget_util.visibleWindow(state.selected_row, visible_rows, state_mod.runtime_entries.len);

    for (0..visible_rows) |visible_idx| {
        const item_idx = start + visible_idx;
        if (item_idx >= state_mod.runtime_entries.len) break;
        const entry = state_mod.runtime_entries[item_idx];
        const selected = item_idx == state.selected_row;
        const row: u16 = @intCast(visible_idx);
        const row_area = layout_util.rowRect(content, row);
        if (selected) widget_util.fillRect(screen, row_area, widget_util.row_highlight_style);

        widgets.renderText(screen, .{
            .x = row_area.x,
            .y = row_area.y,
            .width = @min(@as(u16, 4), row_area.width),
            .height = 1,
        }, widget_util.boolIndicator(entry.enabled), widget_util.styleForState(entry.enabled, selected));

        var label_buf: [96]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "{s} {s}", .{ entry.group, entry.name }) catch entry.name;
        widgets.renderText(screen, .{
            .x = row_area.x + 4,
            .y = row_area.y,
            .width = row_area.width -| 4,
            .height = 1,
        }, label, widget_util.textStyle(selected));
    }
}

pub fn renderRuntimeInspector(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;

    const entry = selectedRuntime(state);
    widget_util.renderSectionTitle(screen, area, "DETAIL", if (state.focused_region == .detail) widget_util.title_style else widget_util.accent_style);

    var row: u16 = 1;
    var line_index: usize = 0;
    const content_area: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "group", entry.group, widget_util.body_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "name", entry.name, widget_util.body_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "state", widget_util.boolLabel(entry.enabled), widget_util.styleForState(entry.enabled, false));
    widget_util.renderScrolledLine(screen, content_area, &row, &line_index, state.detail_scroll, entry.detail, widget_util.muted_style);
}

pub fn renderRuntimeDetail(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        renderRuntimeInspector(screen, area, state);
        return;
    }

    const layout_mod = @import("../../layout.zig");
    const split = layout_mod.split(area, .vertical, &.{
        .{ .percentage = 46 },
        .{ .min = 18 },
    });
    const list_area = if (split.len > 0) split.rects[0] else area;
    const detail_area = if (split.len > 1) split.rects[1] else area;

    renderRuntimeList(screen, layout_util.insetRect(.{
        .x = list_area.x,
        .y = list_area.y,
        .width = list_area.width -| 1,
        .height = list_area.height,
    }, 0, 0), state);
    if (list_area.width > 0) {
        widget_util.drawVerticalDivider(screen, list_area.x + list_area.width -| 1, list_area, widget_util.muted_style);
    }
    renderRuntimeInspector(screen, layout_util.insetRect(detail_area, 1, 0), state);
}
