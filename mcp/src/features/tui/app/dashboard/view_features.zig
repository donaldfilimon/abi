const std = @import("std");
const feature_catalog = @import("../../../core/feature_catalog.zig");
const types = @import("../../types.zig");
const render_mod = @import("../../render.zig");
const widgets = @import("../../widgets.zig");
const state_mod = @import("state.zig");
const widget_util = @import("widgets.zig");
const layout_util = @import("layout.zig");

const Rect = types.Rect;
const Screen = render_mod.Screen;

pub fn selectedFeature(state: *const state_mod.AppState) feature_catalog.Metadata {
    return feature_catalog.all[@min(state.selected_row, feature_catalog.feature_count - 1)];
}

pub fn renderFeaturesSummary(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;
    const entry = selectedFeature(state);
    widget_util.renderSectionTitle(screen, area, "FEATURES", widget_util.title_style);
    var row: u16 = 1;

    var label_buf: [96]u8 = undefined;
    const label = std.fmt.bufPrint(&label_buf, "{s}", .{entry.feature.name()}) catch entry.feature.name();
    widget_util.renderPlainLine(screen, area, &row, label, widget_util.body_style);

    var meta_buf: [128]u8 = undefined;
    const meta_line = std.fmt.bufPrint(&meta_buf, "{d}/{d} selected | {s}", .{
        state.selected_row + 1,
        feature_catalog.feature_count,
        widget_util.boolLabel(widget_util.featureEnabled(entry)),
    }) catch "selected";
    widget_util.renderPlainLine(screen, area, &row, meta_line, widget_util.styleForState(widget_util.featureEnabled(entry), false));

    widgets.renderText(screen, layout_util.rowRect(area, row), entry.description, widget_util.muted_style);
}

pub fn renderFeatureList(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;

    widget_util.renderSectionTitle(screen, area, "CATALOG", if (state.focused_region == .detail) widget_util.accent_style else widget_util.muted_style);
    const content: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    const visible_rows: usize = content.height;
    const start = widget_util.visibleWindow(state.selected_row, visible_rows, feature_catalog.feature_count);

    for (0..visible_rows) |visible_idx| {
        const item_idx = start + visible_idx;
        if (item_idx >= feature_catalog.feature_count) break;
        const entry = feature_catalog.all[item_idx];
        const selected = item_idx == state.selected_row;
        const row: u16 = @intCast(visible_idx);
        const row_area = layout_util.rowRect(content, row);
        if (selected) widget_util.fillRect(screen, row_area, widget_util.row_highlight_style);

        widgets.renderText(screen, .{
            .x = row_area.x,
            .y = row_area.y,
            .width = @min(@as(u16, 4), row_area.width),
            .height = 1,
        }, widget_util.boolIndicator(widget_util.featureEnabled(entry)), widget_util.styleForState(widget_util.featureEnabled(entry), selected));

        var label_buf: [96]u8 = undefined;
        const indent = if (entry.parent != null) "  " else "";
        const label = std.fmt.bufPrint(&label_buf, "{s}{s}", .{ indent, entry.feature.name() }) catch entry.feature.name();
        widgets.renderText(screen, .{
            .x = row_area.x + 4,
            .y = row_area.y,
            .width = row_area.width -| 4,
            .height = 1,
        }, label, widget_util.textStyle(selected));
    }
}

pub fn renderFeatureInspector(screen: *Screen, area: Rect, state: *const state_mod.AppState) void {
    if (area.width == 0 or area.height == 0) return;

    const entry = selectedFeature(state);
    const enabled = widget_util.featureEnabled(entry);
    widget_util.renderSectionTitle(screen, area, "INSPECTOR", if (state.focused_region == .detail) widget_util.title_style else widget_util.accent_style);

    var row: u16 = 1;
    var line_index: usize = 0;
    const content_area: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "feature", entry.feature.name(), widget_util.body_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "state", widget_util.boolLabel(enabled), widget_util.styleForState(enabled, false));
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "flag", entry.compile_flag_field, widget_util.body_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "parent", if (entry.parent) |parent| parent.name() else "root", widget_util.body_style);
    widget_util.renderScrolledLine(screen, content_area, &row, &line_index, state.detail_scroll, entry.description, widget_util.muted_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "module", entry.real_module_path, widget_util.muted_style);
    widget_util.renderScrolledKeyValue(screen, content_area, &row, &line_index, state.detail_scroll, "stub", entry.stub_module_path, widget_util.muted_style);
}

pub fn renderFeaturesDetail(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode) void {
    if (area.width == 0 or area.height == 0) return;

    if (mode == .compact) {
        renderFeatureInspector(screen, area, state);
        return;
    }

    const layout_mod = @import("../../layout.zig");
    const split = layout_mod.split(area, .vertical, &.{
        .{ .percentage = 46 },
        .{ .min = 20 },
    });
    const list_area = if (split.len > 0) split.rects[0] else area;
    const detail_area = if (split.len > 1) split.rects[1] else area;

    renderFeatureList(screen, layout_util.insetRect(.{
        .x = list_area.x,
        .y = list_area.y,
        .width = list_area.width -| 1,
        .height = list_area.height,
    }, 0, 0), state);
    if (list_area.width > 0) {
        widget_util.drawVerticalDivider(screen, list_area.x + list_area.width -| 1, list_area, widget_util.muted_style);
    }
    renderFeatureInspector(screen, layout_util.insetRect(detail_area, 1, 0), state);
}
