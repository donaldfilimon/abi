const std = @import("std");
const abi = @import("../../root.zig");
const DashboardOptions = @import("dashboard.zig").DashboardOptions;
const test_helpers = @import("../../foundation/test_helpers.zig");

pub fn dashboardHealth(ds: abi.features.tui.DashboardState) []const u8 {
    return abi.features.tui.dashboardHealth(ds);
}

pub fn paneNameForIndex(selected: usize) []const u8 {
    if (selected < abi.features.tui.DASHBOARD_PANE_COUNT) {
        return abi.features.tui.dashboardPaneName(abi.features.tui.DASHBOARD_PANES[selected].kind);
    }
    return abi.features.tui.dashboardPaneName(abi.features.tui.DASHBOARD_PANES[0].kind);
}

pub fn writeVisiblePanesJson(json: anytype, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
    try json.objectField("visible_panes");
    try json.beginArray();
    if (options.compact) {
        try json.write(paneNameForIndex(state.selected_pane));
    } else {
        for (abi.features.tui.DASHBOARD_PANES) |pane| {
            try json.write(abi.features.tui.dashboardPaneName(pane.kind));
        }
    }
    try json.endArray();
}

pub fn paneHotkeyString(pane: abi.features.tui.DashboardPaneMeta) [1]u8 {
    return .{pane.hotkey};
}

pub fn paneVisible(idx: usize, state: abi.features.tui.DashboardState, options: DashboardOptions) bool {
    return !options.compact or idx == state.selected_pane;
}

pub fn writePaneMetaJson(json: anytype, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
    try json.objectField("panes");
    try json.beginArray();
    for (abi.features.tui.DASHBOARD_PANES, 0..) |pane, idx| {
        const hotkey = paneHotkeyString(pane);
        try json.beginObject();
        try json.objectField("name");
        try json.write(abi.features.tui.dashboardPaneName(pane.kind));
        try json.objectField("title");
        try json.write(pane.title);
        try json.objectField("hotkey");
        try json.write(hotkey[0..]);
        try json.objectField("selected");
        try json.write(idx == state.selected_pane);
        try json.objectField("visible");
        try json.write(paneVisible(idx, state, options));
        try json.endObject();
    }
    try json.endArray();
}

pub fn renderPaneListWriter(writer: anytype, allocator: std.mem.Allocator, options: DashboardOptions) !void {
    const selected = if (options.initial_pane < abi.features.tui.DASHBOARD_PANE_COUNT) options.initial_pane else 0;
    if (options.format == .json) {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();

        var json = std.json.Stringify{
            .writer = &out.writer,
            .options = .{ .whitespace = .minified },
        };
        try json.beginObject();
        try json.objectField("type");
        try json.write("abi.dashboard.panes");
        try json.objectField("selected_pane");
        try json.write(paneNameForIndex(selected));
        try writePaneMetaJson(&json, .{ .selected_pane = selected }, .{
            .initial_pane = selected,
            .color = options.color,
            .compact = options.compact,
            .force_one_shot = true,
            .refresh_interval_ms = options.refresh_interval_ms,
            .format = .json,
        });
        try json.endObject();
        try writer.writeAll(out.written());
        try writer.writeAll("\n");
        return;
    }

    try writer.writeAll("Dashboard panes:\n");
    for (abi.features.tui.DASHBOARD_PANES, 0..) |pane, idx| {
        const hotkey = paneHotkeyString(pane);
        const marker: []const u8 = if (idx == selected) "*" else " ";
        try writer.print("{s} {s} ({s}) hotkey={s}\n", .{
            marker,
            abi.features.tui.dashboardPaneName(pane.kind),
            pane.title,
            hotkey[0..],
        });
    }
}

pub fn renderJsonWriter(writer: anytype, allocator: std.mem.Allocator, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json = std.json.Stringify{
        .writer = &out.writer,
        .options = .{ .whitespace = .minified },
    };
    try json.beginObject();
    try json.objectField("type");
    try json.write("abi.dashboard");
    try json.objectField("health");
    try json.write(dashboardHealth(state));
    try json.objectField("selected_pane");
    try json.write(paneNameForIndex(state.selected_pane));
    try json.objectField("refresh_interval_ms");
    try json.write(options.refresh_interval_ms);

    try json.objectField("layout");
    try json.beginObject();
    try json.objectField("format");
    try json.write(@tagName(options.format));
    try json.objectField("color");
    try json.write(options.color);
    try json.objectField("compact");
    try json.write(options.compact);
    try writeVisiblePanesJson(&json, state, options);
    try writePaneMetaJson(&json, state, options);
    try json.endObject();

    try json.objectField("gpu");
    try json.beginObject();
    try json.objectField("backend");
    try json.write(state.gpu_backend);
    try json.objectField("accelerated");
    try json.write(state.gpu_accelerated);
    try json.objectField("linked");
    try json.write(state.gpu_linked);
    try json.endObject();

    try json.objectField("plugins");
    try json.beginObject();
    try json.objectField("count");
    try json.write(state.plugin_count);
    try json.objectField("names");
    try json.beginArray();
    for (state.plugin_names) |name| try json.write(name);
    try json.endArray();
    try json.endObject();

    try json.objectField("wdbx");
    try json.beginObject();
    try json.objectField("blocks");
    try json.write(state.wdbx_blocks);
    try json.objectField("vectors");
    try json.write(state.wdbx_vectors);
    try json.objectField("kv_entries");
    try json.write(state.wdbx_entries);
    try json.objectField("spatial_records");
    try json.write(state.wdbx_spatial_records);
    try json.endObject();

    try json.objectField("scheduler");
    try json.beginObject();
    try json.objectField("source");
    try json.write(state.scheduler_source);
    try json.objectField("running");
    try json.write(state.scheduler_running);
    try json.objectField("pending");
    try json.write(state.scheduler_pending);
    try json.objectField("completed");
    try json.write(state.scheduler_completed);
    try json.objectField("failed");
    try json.write(state.scheduler_failed);
    try json.endObject();

    try json.objectField("memory");
    try json.beginObject();
    try json.objectField("source");
    try json.write(state.memory_source);
    try json.objectField("peak_bytes");
    try json.write(state.memory_peak);
    try json.objectField("current_bytes");
    try json.write(state.memory_current);
    try json.objectField("leaked_bytes");
    try json.write(state.memory_leaked);
    try json.endObject();
    try json.endObject();

    try writer.writeAll(out.written());
    try writer.writeAll("\n");
}

test "dashboard json writer emits parseable snapshot" {
    const allocator = std.testing.allocator;

    const state: abi.features.tui.DashboardState = .{
        .gpu_backend = "test-gpu",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .plugin_count = 2,
        .plugin_names = &.{ "alpha", "beta" },
        .wdbx_blocks = 3,
        .wdbx_vectors = 4,
        .wdbx_entries = 5,
        .wdbx_spatial_records = 6,
        .scheduler_source = "test-scheduler",
        .scheduler_completed = 7,
        .memory_source = "test-memory",
        .memory_peak = 8,
        .selected_pane = 1,
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const writer = test_helpers.TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderJsonWriter(&writer, allocator, state, .{ .refresh_interval_ms = 250, .format = .json });

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("abi.dashboard", root.get("type").?.string);
    try std.testing.expectEqualStrings("nominal", root.get("health").?.string);
    try std.testing.expectEqualStrings("plugins", root.get("selected_pane").?.string);
    try std.testing.expectEqual(@as(i64, 250), root.get("refresh_interval_ms").?.integer);

    const layout = root.get("layout").?.object;
    try std.testing.expectEqualStrings("json", layout.get("format").?.string);
    try std.testing.expect(layout.get("color").?.bool);
    try std.testing.expect(!layout.get("compact").?.bool);
    try std.testing.expectEqual(@as(usize, abi.features.tui.DASHBOARD_PANE_COUNT), layout.get("visible_panes").?.array.items.len);
    try std.testing.expectEqualStrings("system", layout.get("visible_panes").?.array.items[0].string);
    const panes = layout.get("panes").?.array.items;
    try std.testing.expectEqual(@as(usize, abi.features.tui.DASHBOARD_PANE_COUNT), panes.len);
    const plugin_pane = panes[1].object;
    try std.testing.expectEqualStrings("plugins", plugin_pane.get("name").?.string);
    try std.testing.expectEqualStrings("Plugins", plugin_pane.get("title").?.string);
    try std.testing.expectEqualStrings("2", plugin_pane.get("hotkey").?.string);
    try std.testing.expect(plugin_pane.get("selected").?.bool);
    try std.testing.expect(plugin_pane.get("visible").?.bool);

    const plugins = root.get("plugins").?.object;
    try std.testing.expectEqual(@as(i64, 2), plugins.get("count").?.integer);
    try std.testing.expectEqual(@as(usize, 2), plugins.get("names").?.array.items.len);
    try std.testing.expectEqualStrings("alpha", plugins.get("names").?.array.items[0].string);

    const wdbx = root.get("wdbx").?.object;
    try std.testing.expectEqual(@as(i64, 3), wdbx.get("blocks").?.integer);
    try std.testing.expectEqual(@as(i64, 5), wdbx.get("kv_entries").?.integer);
}

test "dashboard json writer reports compact layout pane visibility" {
    const allocator = std.testing.allocator;
    const state: abi.features.tui.DashboardState = .{
        .gpu_backend = "test-gpu",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .selected_pane = 3,
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const writer = test_helpers.TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderJsonWriter(&writer, allocator, state, .{ .color = false, .compact = true, .format = .json });

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("scheduler", root.get("selected_pane").?.string);
    const layout = root.get("layout").?.object;
    try std.testing.expect(!layout.get("color").?.bool);
    try std.testing.expect(layout.get("compact").?.bool);
    const visible = layout.get("visible_panes").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), visible.len);
    try std.testing.expectEqualStrings("scheduler", visible[0].string);
    const panes = layout.get("panes").?.array.items;
    try std.testing.expect(panes[0].object.get("visible").?.bool == false);
    try std.testing.expect(panes[3].object.get("selected").?.bool);
    try std.testing.expect(panes[3].object.get("visible").?.bool);
}

test "dashboard pane list writer emits text and json metadata" {
    const allocator = std.testing.allocator;

    var text_buf = std.ArrayListUnmanaged(u8).empty;
    defer text_buf.deinit(allocator);
    var text_writer = test_helpers.TestWriter{ .allocator = allocator, .buffer = &text_buf };
    try renderPaneListWriter(&text_writer, allocator, .{ .initial_pane = 3 });
    try std.testing.expect(std.mem.indexOf(u8, text_buf.items, "Dashboard panes:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text_buf.items, "* scheduler (Scheduler) hotkey=4") != null);

    var json_buf = std.ArrayListUnmanaged(u8).empty;
    defer json_buf.deinit(allocator);
    var json_writer = test_helpers.TestWriter{ .allocator = allocator, .buffer = &json_buf };
    try renderPaneListWriter(&json_writer, allocator, .{ .initial_pane = 4, .format = .json });

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("abi.dashboard.panes", root.get("type").?.string);
    try std.testing.expectEqualStrings("memory", root.get("selected_pane").?.string);
    const panes = root.get("panes").?.array.items;
    try std.testing.expectEqual(@as(usize, abi.features.tui.DASHBOARD_PANE_COUNT), panes.len);
    try std.testing.expectEqualStrings("memory", panes[4].object.get("name").?.string);
    try std.testing.expectEqualStrings("5", panes[4].object.get("hotkey").?.string);
    try std.testing.expect(panes[4].object.get("selected").?.bool);
}

test {
    std.testing.refAllDecls(@This());
}
