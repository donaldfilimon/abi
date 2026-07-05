const std = @import("std");
const abi = @import("../../root.zig");

/// `abi dashboard`: print a one-shot status dashboard — the loaded plugin names
/// from the registry and a snapshot of the WDBX store. Returns the exit code.
pub fn handleDashboard(allocator: std.mem.Allocator) !u8 {
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

    // Try interactive mode; fall back to one-shot on failure or non-TTY
    var term = abi.features.tui.InteractiveTerminal.init(abi.features.tui.stdinFd()) catch {
        return renderOneShot(allocator, &scheduler, &store, &registry, plugin_names);
    };
    defer term.deinit();

    try abi.features.tui.initScreen();
    defer abi.features.tui.deinitScreen();

    var quit = false;
    while (!quit) {
        _ = try scheduler.submit("dashboard-refresh", .low, struct {
            fn run(_: ?*anyopaque) anyerror!void {}
        }.run, null);
        _ = try scheduler.runNext();

        try renderAndPrint(allocator, &scheduler, &store, &registry, plugin_names);

        // Wait up to 1s for input. Timeout auto-refreshes; r/R refreshes
        // immediately; unrelated keys are ignored instead of redrawing.
        while (term.pollInput(1000)) {
            const key = term.readKey() orelse {
                quit = true;
                break;
            };
            if (abi.features.tui.isQuitKey(key)) {
                quit = true;
                break;
            }
            if (abi.features.tui.isRefreshKey(key)) break;
        }
    }

    return 0;
}

fn renderOneShot(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8) !u8 {
    try abi.features.tui.initScreen();
    defer abi.features.tui.deinitScreen();
    try renderAndPrint(allocator, scheduler, store, registry, plugin_names);
    return 0;
}

fn renderAndPrint(allocator: std.mem.Allocator, scheduler: anytype, store: anytype, registry: anytype, plugin_names: []const []const u8) !void {
    const wdbx_stats = store.stats();
    const gpu_status = abi.features.gpu.detectBackend();
    const native_gpu = abi.features.gpu.nativeKernelStatus();

    const mem_tracker = scheduler.getMemoryTracker();
    const memory_source: []const u8 = if (mem_tracker) |_| "MemoryTracker (live)" else "not attached";
    const memory_peak: usize = if (mem_tracker) |t| t.getPeakUsage() else 0;
    const memory_current: usize = if (mem_tracker) |t| t.getCurrentUsage() else 0;
    const memory_leaked: usize = if (mem_tracker) |t| t.getLeakedBytes() else 0;

    const rendered = try abi.features.tui.renderDiagnostics(allocator, .{
        .gpu_backend = abi.features.gpu.backendName(gpu_status.backend),
        .gpu_accelerated = gpu_status.accelerated,
        .gpu_linked = native_gpu.linked,
        .plugin_count = registry.pluginCount(),
        .plugin_names = plugin_names,
        .wdbx_blocks = wdbx_stats.blocks,
        .wdbx_vectors = wdbx_stats.vectors,
        .wdbx_entries = wdbx_stats.kv_entries,
        .wdbx_spatial_records = wdbx_stats.spatial_records,
        .scheduler_source = "CLI dashboard (live)",
        .scheduler_running = scheduler.stats().running,
        .scheduler_pending = scheduler.stats().pending,
        .scheduler_completed = scheduler.stats().completed,
        .scheduler_failed = scheduler.stats().failed,
        .memory_source = memory_source,
        .memory_peak = memory_peak,
        .memory_current = memory_current,
        .memory_leaked = memory_leaked,
    });
    defer allocator.free(rendered);

    // Flicker-free redraw: home the cursor, overwrite the frame in place, then
    // clear any trailing rows a shorter frame would have left behind.
    abi.features.tui.homeScreen();
    std.debug.print("{s}\n[q] quit   [r] refresh   (live, auto-refresh 1s)\n", .{rendered});
    abi.features.tui.clearToEnd();
}

pub const renderTui = handleDashboard;
