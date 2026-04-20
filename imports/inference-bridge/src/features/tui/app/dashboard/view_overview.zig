const std = @import("std");
const build_options = @import("build_options");
const feature_catalog = @import("../../../core/feature_catalog.zig");
const platform = @import("../../../../platform/mod.zig");
const types = @import("../../types.zig");
const render_mod = @import("../../render.zig");
const state_mod = @import("state.zig");
const widget_util = @import("widgets.zig");

const Rect = types.Rect;
const Screen = render_mod.Screen;

pub fn renderOverviewSummary(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;
    widget_util.renderSectionTitle(screen, area, "OVERVIEW", widget_util.title_style);
    var row: u16 = 1;

    var term_buf: [128]u8 = undefined;
    const term_line = std.fmt.bufPrint(&term_buf, "v{s} | {d}x{d} terminal | focus {s} | {s}", .{
        build_options.package_version,
        screen_width,
        screen_height,
        widget_util.focusLabel(state.focused_region),
        widget_util.modeLabel(mode),
    }) catch "terminal";
    widget_util.renderPlainLine(screen, area, &row, term_line, widget_util.body_style);

    var count_buf: [96]u8 = undefined;
    const count_line = std.fmt.bufPrint(&count_buf, "{d}/{d} catalog features | {d}/10 GPU | {d}/8 services", .{
        widget_util.enabledFeatureCount(),
        feature_catalog.feature_count,
        widget_util.enabledRuntimeCount("GPU"),
        widget_util.enabledRuntimeCount("SERV"),
    }) catch "counts";
    widget_util.renderPlainLine(screen, area, &row, count_line, widget_util.body_style);
}

pub fn renderOverviewDetail(screen: *Screen, area: Rect, state: *const state_mod.AppState, mode: state_mod.LayoutMode, screen_width: u16, screen_height: u16) void {
    if (area.width == 0 or area.height == 0) return;

    widget_util.renderSectionTitle(screen, area, "METRICS", widget_util.title_style);
    var row: u16 = 1;

    const content_area: Rect = .{
        .x = area.x,
        .y = area.y + 1,
        .width = area.width,
        .height = area.height -| 1,
    };

    widget_util.renderMetricGauge(
        screen,
        content_area,
        &row,
        "feature coverage",
        widget_util.enabledFeatureCount(),
        feature_catalog.feature_count,
        widget_util.ok_style,
        widget_util.muted_style,
    );
    widget_util.renderMetricGauge(
        screen,
        content_area,
        &row,
        "runtime surfaces",
        widget_util.enabledRuntimeCount("GPU") + widget_util.enabledRuntimeCount("SERV"),
        state_mod.runtime_entries.len,
        widget_util.accent_style,
        widget_util.muted_style,
    );

    // Live SMC metrics if available
    if (platform.smc.read()) |smc| {
        if (smc.cpu_temp_c > 0) {
            widget_util.renderMetricGauge(
                screen,
                content_area,
                &row,
                "CPU Temp (C)",
                @intFromFloat(smc.cpu_temp_c),
                100,
                if (smc.cpu_temp_c > 80) widget_util.off_style else widget_util.ok_style,
                widget_util.muted_style,
            );
        }
        if (smc.gpu_temp_c > 0) {
            widget_util.renderMetricGauge(
                screen,
                content_area,
                &row,
                "GPU Temp (C)",
                @intFromFloat(smc.gpu_temp_c),
                100,
                if (smc.gpu_temp_c > 80) widget_util.off_style else widget_util.ok_style,
                widget_util.muted_style,
            );
        }
        if (smc.fan_rpm[0]) |rpm| {
            // Assume 7000 RPM max for gauge
            widget_util.renderMetricGauge(
                screen,
                content_area,
                &row,
                "Fan RPM",
                @intCast(rpm),
                7000,
                widget_util.accent_style,
                widget_util.muted_style,
            );
        }
    } else |_| {}

    widget_util.renderKeyValue(screen, content_area, &row, "version", build_options.package_version, widget_util.body_style);

    var size_buf: [64]u8 = undefined;
    const size_line = std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ screen_width, screen_height }) catch "terminal";
    widget_util.renderKeyValue(screen, content_area, &row, "terminal", size_line, widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "platform", platform.getDescription(), widget_util.body_style);

    var cpu_buf: [64]u8 = undefined;
    const cpu_line = std.fmt.bufPrint(&cpu_buf, "{d}", .{platform.getCpuCount()}) catch "1";
    widget_util.renderKeyValue(screen, content_area, &row, "cpu", cpu_line, widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "threading", if (platform.supportsThreading()) "supported" else "disabled", widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "gpu ready", if (platform.isGpuAvailable()) "possible" else "off", widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "network", if (platform.hasNetworkAccess()) "available" else "restricted", widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "filesystem", if (platform.hasFileSystem()) "available" else "restricted", widget_util.body_style);

    widget_util.renderKeyValue(screen, content_area, &row, "layout", widget_util.modeLabel(mode), widget_util.muted_style);

    widget_util.renderKeyValue(screen, content_area, &row, "help", if (state.help_visible) "visible" else "hidden", widget_util.muted_style);
}
