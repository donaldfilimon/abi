const std = @import("std");
const abi = @import("../../root.zig");
const build_options = @import("build_options");
const dashboard_json = @import("dashboard_json.zig");

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
        try dashboard_json.renderPaneListWriter(&writer, allocator, options);
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
    var focused_pane: abi.features.tui.FocusedPane = .left;
    var agent_scroll: usize = 0;
    var quit = false;
    while (!quit) {
        _ = try scheduler.submit("dashboard-refresh", .low, struct {
            fn run(_: ?*anyopaque) anyerror!void {}
        }.run, null);
        _ = try scheduler.runNext();

        try renderWithSplit(allocator, &scheduler, &store, &registry, plugin_names, gpu_snapshot, selected_pane, options, focused_pane, agent_scroll);

        // Timeout auto-refreshes; r/R refreshes immediately; 1-5 or h/l to
        // select pane when left-focused; Tab toggles focus; j/k scrolls
        // agent output when right-focused.
        while (session.term.pollInput(options.refresh_interval_ms)) {
            const key = session.term.readKey() orelse {
                quit = true;
                break;
            };
            if (abi.features.tui.isQuitKey(key)) {
                quit = true;
                break;
            }
            if (abi.features.tui.isTabKey(key)) {
                focused_pane = if (focused_pane == .left) .right else .left;
                break;
            }
            if (focused_pane == .right) {
                if (abi.features.tui.isScrollUpKey(key)) {
                    if (agent_scroll > 0) agent_scroll -= 1;
                    break;
                }
                if (abi.features.tui.isScrollDownKey(key)) {
                    agent_scroll += 1;
                    break;
                }
            }
            if (focused_pane == .left) {
                if (abi.features.tui.isRefreshKey(key)) break;
                if (nextPane(selected_pane, key)) |pane| {
                    selected_pane = pane;
                    break;
                }
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

fn renderWithSplit(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions, focused_pane: abi.features.tui.FocusedPane, agent_scroll: usize) !void {
    var writer = DebugWriter{};
    try renderFrameWriterSplit(&writer, allocator, scheduler, store, registry, plugin_names, gpu_snapshot, selected, options, true, focused_pane, agent_scroll);
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
            try dashboard_json.renderJsonWriter(writer, allocator, state, options);
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

/// Render the split-pane dashboard (diagnostics left, agent output right).
/// Same screen-control protocol as renderFrameWriter for flicker-free redraw.
fn renderFrameWriterSplit(writer: anytype, allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8, gpu_snapshot: GpuSnapshot, selected: usize, options: DashboardOptions, screen_control: bool, focused_pane: abi.features.tui.FocusedPane, agent_scroll: usize) !void {
    var state = collectDashboardState(scheduler, store, registry, plugin_names, gpu_snapshot, selected);
    state.focused_pane = focused_pane;
    state.agent_output_scroll = agent_scroll;

    const rendered = try abi.features.tui.renderDiagnosticsSplitWithOptions(allocator, state, .{
        .color = options.color,
        .refresh_interval_ms = @intCast(options.refresh_interval_ms),
        .compact = options.compact,
    });
    defer allocator.free(rendered);

    if (screen_control) try abi.features.tui.homeScreenWriter(writer);
    try writer.writeAll(rendered);
    if (screen_control) try abi.features.tui.clearToEndWriter(writer);
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

test "dashboard refresh interval validation enforces interactive bounds" {
    try std.testing.expect(validRefreshInterval(MIN_REFRESH_INTERVAL_MS - 1) == null);
    try std.testing.expectEqual(@as(?i32, MIN_REFRESH_INTERVAL_MS), validRefreshInterval(MIN_REFRESH_INTERVAL_MS));
    try std.testing.expectEqual(@as(?i32, DEFAULT_REFRESH_INTERVAL_MS), validRefreshInterval(DEFAULT_REFRESH_INTERVAL_MS));
    try std.testing.expectEqual(@as(?i32, MAX_REFRESH_INTERVAL_MS), validRefreshInterval(MAX_REFRESH_INTERVAL_MS));
    try std.testing.expect(validRefreshInterval(MAX_REFRESH_INTERVAL_MS + 1) == null);
}
