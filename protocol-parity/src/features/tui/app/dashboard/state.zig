const std = @import("std");
const feature_catalog = @import("../../../core/feature_catalog.zig");
const types = @import("../../types.zig");
const Key = types.Key;

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

pub const DashboardAction = enum {
    none,
    quit,
};

pub const AppState = struct {
    current_view: View = .overview,
    focused_region: FocusRegion = .nav,
    nav_index: usize = 0,
    selected_row: usize = 0,
    detail_scroll: usize = 0,
    help_visible: bool = false,
};

pub const NavItem = struct {
    view: View,
    label: []const u8,
    compact_label: []const u8,
    blurb: []const u8,
};

pub const nav_items = [_]NavItem{
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

pub fn navIndexForView(view: View) usize {
    inline for (nav_items, 0..) |item, idx| {
        if (item.view == view) return idx;
    }
    return 0;
}

pub const RuntimeEntry = struct {
    group: []const u8,
    name: []const u8,
    enabled: bool,
    detail: []const u8,
};

const build_options = @import("build_options");

pub const runtime_entries = [_]RuntimeEntry{
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

pub fn selectedItemCount(view: View) usize {
    return switch (view) {
        .overview => 0,
        .features => feature_catalog.feature_count,
        .runtime => runtime_entries.len,
    };
}

pub fn clampState(state: *AppState) void {
    if (state.nav_index >= nav_items.len) state.nav_index = nav_items.len - 1;
    const count = selectedItemCount(state.current_view);
    if (count == 0) {
        state.selected_row = 0;
        state.detail_scroll = 0;
        return;
    }
    if (state.selected_row >= count) state.selected_row = count - 1;
}

pub fn activateNavSelection(state: *AppState) void {
    state.current_view = nav_items[state.nav_index].view;
    state.selected_row = 0;
    state.detail_scroll = 0;
    state.focused_region = if (state.current_view == .overview) .nav else .detail;
}

pub fn moveNavSelection(state: *AppState, delta: i32) void {
    const current: i32 = @intCast(state.nav_index);
    const max_index: i32 = @intCast(nav_items.len - 1);
    const next = std.math.clamp(current + delta, 0, max_index);
    state.nav_index = @intCast(next);
}

pub fn moveDetailSelection(state: *AppState, delta: i32) void {
    const count = selectedItemCount(state.current_view);
    if (count == 0) return;
    const current: i32 = @intCast(state.selected_row);
    const max_index: i32 = @intCast(count - 1);
    const next = std.math.clamp(current + delta, 0, max_index);
    state.selected_row = @intCast(next);
    state.detail_scroll = 0;
}

pub fn scrollDetail(state: *AppState, delta: i32) void {
    const current: i32 = @intCast(state.detail_scroll);
    const next = std.math.clamp(current + delta, 0, 64);
    state.detail_scroll = @intCast(next);
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
