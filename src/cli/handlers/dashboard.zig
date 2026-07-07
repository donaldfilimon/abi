const std = @import("std");
const abi = @import("../../root.zig");
const build_options = @import("build_options");

const GpuSnapshot = struct {
    backend: []const u8,
    accelerated: bool,
    linked: bool,
};

const DebugWriter = struct {
    pub fn writeAll(_: *@This(), bytes: []const u8) !void {
        std.debug.print("{s}", .{bytes});
    }

    pub fn print(_: *@This(), comptime format: []const u8, args: anytype) !void {
        std.debug.print(format, args);
    }
};

pub const DashboardFormat = enum {
    text,
    json,
};

pub const DEFAULT_REFRESH_INTERVAL_MS: i32 = 1000;
pub const MIN_REFRESH_INTERVAL_MS: i32 = 100;
pub const MAX_REFRESH_INTERVAL_MS: i32 = 60_000;

pub const DashboardOptions = struct {
    initial_pane: usize = 0,
    color: bool = true,
    compact: bool = false,
    force_one_shot: bool = false,
    refresh_interval_ms: i32 = DEFAULT_REFRESH_INTERVAL_MS,
    format: DashboardFormat = .text,
    list_panes: bool = false,
};

pub fn validRefreshInterval(raw_ms: u64) ?i32 {
    const ms = std.math.cast(i32, raw_ms) orelse return null;
    if (ms < MIN_REFRESH_INTERVAL_MS or ms > MAX_REFRESH_INTERVAL_MS) return null;
    return ms;
}

fn nextPane(current: usize, key: u8) ?usize {
    return abi.features.tui.nextDashboardPane(current, key);
}

pub fn dashboardPaneIndexForToken(token: []const u8) ?usize {
    return abi.features.tui.dashboardPaneIndexForToken(token);
}

/// `abi dashboard`: print a one-shot status dashboard — the loaded plugin names
/// from the registry and a snapshot of the WDBX store. Returns the exit code.
pub fn handleDashboard(allocator: std.mem.Allocator) !u8 {
    return handleDashboardWithOptions(allocator, .{});
}

pub fn handleDashboardWithOptions(allocator: std.mem.Allocator, options: DashboardOptions) !u8 {
    if (options.list_panes) {
        var writer = DebugWriter{};
        try renderPaneListWriter(&writer, allocator, options);
        return 0;
    }

    const gpu_status = abi.features.gpu.detectBackend();
    const native_gpu = abi.features.gpu.nativeKernelStatus();
    const gpu_snapshot = GpuSnapshot{
        .backend = abi.features.gpu.backendName(gpu_status.backend),
        .accelerated = gpu_status.accelerated,
        .linked = native_gpu.linked,
    };

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();
    try registry.loadPlugins();

    const plugin_names = try registry.snapshotPluginNames(allocator);
    defer abi.registry.Registry.freePluginNamesSnapshot(allocator, plugin_names);

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();

    var mem_tracker = abi.memory.MemoryTracker.init(allocator);
    defer mem_tracker.deinit();
    scheduler.setMemoryTracker(&mem_tracker);

    _ = try scheduler.submit("dashboard-init", .normal, struct {
        fn run(_: ?*anyopaque) anyerror!void {}
    }.run, null);
    _ = try scheduler.submit("wdbx-snapshot", .low, struct {
        fn run(_: ?*anyopaque) anyerror!void {}
    }.run, null);
    try scheduler.runAll();

    if (options.force_one_shot) {
        return renderOneShot(allocator, &scheduler, &store, &registry, plugin_names, gpu_snapshot, options.initial_pane, options);
    }

    if (options.format == .json) {
        return renderOneShot(allocator, &scheduler, &store, &registry, plugin_names, gpu_snapshot, options.initial_pane, options);
    }

    // Try interactive mode; fall back to one-shot on failure or non-TTY. The
    // session owns both raw-mode restoration and alternate-screen restoration.
    var session = abi.features.tui.ScreenSession.init(abi.features.tui.stdinFd()) catch {
        return renderOneShot(allocator, &scheduler, &store, &registry, plugin_names, gpu_snapshot, options.initial_pane, options);
    };
    defer session.deinit();

    var selected_pane: usize = options.initial_pane;
    var quit = false;
    while (!quit) {
        _ = try scheduler.submit("dashboard-refresh", .low, struct {
            fn run(_: ?*anyopaque) anyerror!void {}
        }.run, null);
        _ = try scheduler.runNext();

        try renderAndPrint(allocator, &scheduler, &store, &registry, plugin_names, gpu_snapshot, selected_pane, options);

        // Timeout auto-refreshes; r/R refreshes immediately; 1-5 or h/l to
        // select pane; unrelated keys ignored.
        while (session.term.pollInput(options.refresh_interval_ms)) {
            const key = session.term.readKey() orelse {
                quit = true;
                break;
            };
            if (abi.features.tui.isQuitKey(key)) {
                quit = true;
                break;
            }
            if (abi.features.tui.isRefreshKey(key)) break;
            if (nextPane(selected_pane, key)) |pane| {
                selected_pane = pane;
                break;
            }
        }
    }

    return 0;
}

fn renderOneShot(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions) !u8 {
    var writer = DebugWriter{};
    try renderSnapshotWriter(&writer, allocator, scheduler, store, registry, plugin_names, gpu_snapshot, selected, options, false);
    return 0;
}

fn renderAndPrint(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions) !void {
    try renderFrame(allocator, scheduler, store, registry, plugin_names, gpu_snapshot, selected, options, true);
}

fn renderFrame(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions, screen_control: bool) !void {
    var writer = DebugWriter{};
    try renderFrameWriter(&writer, allocator, scheduler, store, registry, plugin_names, gpu_snapshot, selected, options, screen_control);
}

fn renderSnapshotWriter(writer: anytype, allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions, screen_control: bool) !void {
    switch (options.format) {
        .text => try renderFrameWriter(writer, allocator, scheduler, store, registry, plugin_names, gpu_snapshot, selected, options, screen_control),
        .json => {
            const state = collectDashboardState(scheduler, store, registry, plugin_names, gpu_snapshot, selected);
            try renderJsonWriter(writer, allocator, state, options);
        },
    }
}

fn renderFrameWriter(writer: anytype, allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions, screen_control: bool) !void {
    const state = collectDashboardState(scheduler, store, registry, plugin_names, gpu_snapshot, selected);
    const rendered = try abi.features.tui.renderDiagnosticsWithOptions(allocator, state, .{
        .color = options.color,
        .refresh_interval_ms = @intCast(options.refresh_interval_ms),
        .compact = options.compact,
    });
    defer allocator.free(rendered);

    // Flicker-free redraw: home the cursor, overwrite the frame in place, then
    // clear any trailing rows a shorter frame would have left behind.
    if (screen_control) try abi.features.tui.homeScreenWriter(writer);
    try writer.writeAll(rendered);
    if (screen_control) try abi.features.tui.clearToEndWriter(writer);
}

fn dashboardHealth(ds: abi.features.tui.DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (!ds.gpu_accelerated or !ds.gpu_linked) return "degraded";
    return "nominal";
}

fn paneNameForIndex(selected: usize) []const u8 {
    if (selected < abi.features.tui.DASHBOARD_PANE_COUNT) {
        return abi.features.tui.dashboardPaneName(abi.features.tui.DASHBOARD_PANES[selected].kind);
    }
    return abi.features.tui.dashboardPaneName(abi.features.tui.DASHBOARD_PANES[0].kind);
}

fn writeVisiblePanesJson(json: anytype, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
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

fn paneHotkeyString(pane: abi.features.tui.DashboardPaneMeta) [1]u8 {
    return .{pane.hotkey};
}

fn paneVisible(idx: usize, state: abi.features.tui.DashboardState, options: DashboardOptions) bool {
    return !options.compact or idx == state.selected_pane;
}

fn writePaneMetaJson(json: anytype, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
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

fn renderPaneListWriter(writer: anytype, allocator: std.mem.Allocator, options: DashboardOptions) !void {
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

fn renderJsonWriter(writer: anytype, allocator: std.mem.Allocator, state: abi.features.tui.DashboardState, options: DashboardOptions) !void {
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

fn collectDashboardState(scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize) abi.features.tui.DashboardState {
    const wdbx_stats = store.stats();
    const scheduler_stats = scheduler.stats();

    const mem_tracker = scheduler.getMemoryTracker();
    const memory_source: []const u8 = if (mem_tracker) |_| "MemoryTracker (live)" else "not attached";
    const memory_peak: usize = if (mem_tracker) |t| t.getPeakUsage() else 0;
    const memory_current: usize = if (mem_tracker) |t| t.getCurrentUsage() else 0;
    const memory_leaked: usize = if (mem_tracker) |t| t.getLeakedBytes() else 0;

    return .{
        .gpu_backend = gpu_snapshot.backend,
        .gpu_accelerated = gpu_snapshot.accelerated,
        .gpu_linked = gpu_snapshot.linked,
        .plugin_count = registry.pluginCount(),
        .plugin_names = plugin_names,
        .wdbx_blocks = wdbx_stats.blocks,
        .wdbx_vectors = wdbx_stats.vectors,
        .wdbx_entries = wdbx_stats.kv_entries,
        .wdbx_spatial_records = wdbx_stats.spatial_records,
        .scheduler_source = "CLI dashboard (live)",
        .scheduler_running = scheduler_stats.running,
        .scheduler_pending = scheduler_stats.pending,
        .scheduler_completed = scheduler_stats.completed,
        .scheduler_failed = scheduler_stats.failed,
        .memory_source = memory_source,
        .memory_peak = memory_peak,
        .memory_current = memory_current,
        .memory_leaked = memory_leaked,
        .selected_pane = selected,
    };
}

pub const renderTui = handleDashboard;

test {
    std.testing.refAllDecls(@This());
}

test "dashboard pane navigation maps digits and wraps" {
    try std.testing.expectEqual(@as(?usize, 0), nextPane(0, '1'));
    try std.testing.expectEqual(@as(?usize, 4), nextPane(0, '5'));
    try std.testing.expectEqual(@as(?usize, 1), nextPane(0, 'l'));
    try std.testing.expectEqual(@as(?usize, 0), nextPane(abi.features.tui.DASHBOARD_PANE_COUNT - 1, 'L'));
    try std.testing.expectEqual(@as(?usize, abi.features.tui.DASHBOARD_PANE_COUNT - 1), nextPane(0, 'h'));
    try std.testing.expect(nextPane(0, '9') == null);
    try std.testing.expect(nextPane(0, 'x') == null);
}

test "dashboard state collection mirrors live scheduler store and registry snapshots" {
    const allocator = std.testing.allocator;

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();
    try store.store("dashboard:test", "ok");

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();
    var tracker = abi.memory.MemoryTracker.init(allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);
    _ = try scheduler.submit("dashboard-state-test", .normal, struct {
        fn run(_: ?*anyopaque) anyerror!void {}
    }.run, null);
    try scheduler.runAll();

    const state = collectDashboardState(
        &scheduler,
        &store,
        &registry,
        &.{ "alpha", "beta" },
        .{ .backend = "test-gpu", .accelerated = true, .linked = false },
        2,
    );

    try std.testing.expectEqualStrings("test-gpu", state.gpu_backend);
    try std.testing.expect(state.gpu_accelerated);
    try std.testing.expect(!state.gpu_linked);
    try std.testing.expectEqual(@as(usize, 2), state.plugin_names.len);
    try std.testing.expectEqual(@as(usize, 1), state.wdbx_entries);
    try std.testing.expectEqual(@as(usize, 1), state.scheduler_completed);
    try std.testing.expectEqualStrings("MemoryTracker (live)", state.memory_source);
    try std.testing.expectEqual(@as(usize, 2), state.selected_pane);
}

test "dashboard frame writer omits redraw controls without screen control" {
    const allocator = std.testing.allocator;

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderFrameWriter(
        &writer,
        allocator,
        &scheduler,
        &store,
        &registry,
        &.{},
        .{ .backend = "test-gpu", .accelerated = true, .linked = true },
        0,
        .{},
        false,
    );

    if (!build_options.feat_tui) {
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "TUI diagnostics are disabled in this build") != null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[H") == null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[0J") == null);
        return;
    }

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[H") == null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[0J") == null);
}

test "dashboard frame writer wraps screen-controlled redraw" {
    const allocator = std.testing.allocator;

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderFrameWriter(
        &writer,
        allocator,
        &scheduler,
        &store,
        &registry,
        &.{},
        .{ .backend = "test-gpu", .accelerated = true, .linked = true },
        0,
        .{},
        true,
    );

    if (!build_options.feat_tui) {
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "TUI diagnostics are disabled in this build") != null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[H") == null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[0J") == null);
        return;
    }

    try std.testing.expect(std.mem.startsWith(u8, buf.items, "\x1b[H"));
    try std.testing.expect(std.mem.endsWith(u8, buf.items, "\x1b[0J"));
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "ABI Diagnostics Dashboard") != null);
}

test "dashboard frame writer can render plain diagnostics without style escapes" {
    const allocator = std.testing.allocator;

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderFrameWriter(
        &writer,
        allocator,
        &scheduler,
        &store,
        &registry,
        &.{},
        .{ .backend = "test-gpu", .accelerated = true, .linked = true },
        4,
        .{ .color = false, .refresh_interval_ms = 250 },
        false,
    );

    if (!build_options.feat_tui) {
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "TUI diagnostics are disabled in this build") != null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[") == null);
        return;
    }

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Memory") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "live snapshot every 250ms") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\x1b[") == null);
}

test "dashboard frame writer can render compact selected pane only" {
    const allocator = std.testing.allocator;

    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();

    var scheduler = abi.scheduler.Scheduler.init(allocator);
    defer scheduler.deinit();

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
    try renderFrameWriter(
        &writer,
        allocator,
        &scheduler,
        &store,
        &registry,
        &.{},
        .{ .backend = "test-gpu", .accelerated = true, .linked = true },
        3,
        .{ .color = false, .compact = true },
        false,
    );

    if (!build_options.feat_tui) {
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "TUI diagnostics are disabled in this build") != null);
        try std.testing.expect(std.mem.indexOf(u8, buf.items, "Scheduler") == null);
        return;
    }

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "ABI Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Scheduler") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Completed") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "System") == null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Plugins") == null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Memory") == null);
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

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
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

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = allocator, .buffer = &buf };
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

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }

        pub fn print(self: *@This(), comptime format: []const u8, args: anytype) !void {
            const rendered = try std.fmt.allocPrint(self.allocator, format, args);
            defer self.allocator.free(rendered);
            try self.buffer.appendSlice(self.allocator, rendered);
        }
    };

    var text_buf = std.ArrayListUnmanaged(u8).empty;
    defer text_buf.deinit(allocator);
    var text_writer = TestWriter{ .allocator = allocator, .buffer = &text_buf };
    try renderPaneListWriter(&text_writer, allocator, .{ .initial_pane = 3 });
    try std.testing.expect(std.mem.indexOf(u8, text_buf.items, "Dashboard panes:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text_buf.items, "* scheduler (Scheduler) hotkey=4") != null);

    var json_buf = std.ArrayListUnmanaged(u8).empty;
    defer json_buf.deinit(allocator);
    var json_writer = TestWriter{ .allocator = allocator, .buffer = &json_buf };
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

test "dashboard refresh interval validation enforces interactive bounds" {
    try std.testing.expect(validRefreshInterval(MIN_REFRESH_INTERVAL_MS - 1) == null);
    try std.testing.expectEqual(@as(?i32, MIN_REFRESH_INTERVAL_MS), validRefreshInterval(MIN_REFRESH_INTERVAL_MS));
    try std.testing.expectEqual(@as(?i32, DEFAULT_REFRESH_INTERVAL_MS), validRefreshInterval(DEFAULT_REFRESH_INTERVAL_MS));
    try std.testing.expectEqual(@as(?i32, MAX_REFRESH_INTERVAL_MS), validRefreshInterval(MAX_REFRESH_INTERVAL_MS));
    try std.testing.expect(validRefreshInterval(MAX_REFRESH_INTERVAL_MS + 1) == null);
}
