const std = @import("std");
const abi = @import("../../root.zig");

extern fn isatty(fd: std.posix.fd_t) callconv(.c) c_int;

fn setRawMode(fd: std.posix.fd_t) !std.posix.termios {
    const original = try std.posix.tcgetattr(fd);
    var raw = original;
    raw.lflag.ICANON = false;
    raw.lflag.ECHO = false;
    raw.cc[@intFromEnum(std.posix.system.V.MIN)] = 1;
    raw.cc[@intFromEnum(std.posix.system.V.TIME)] = 0;
    try std.posix.tcsetattr(fd, .FLUSH, raw);
    return original;
}

fn restoreMode(fd: std.posix.fd_t, original: std.posix.termios) void {
    std.posix.tcsetattr(fd, .FLUSH, original) catch |err| {
        std.log.warn("failed to restore terminal mode: {s}", .{@errorName(err)});
    };
}

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

    const fd = std.posix.STDIN_FILENO;
    const is_tty = isatty(fd) != 0;

    var original_termios: ?std.posix.termios = null;
    if (is_tty) {
        original_termios = setRawMode(fd) catch |err| blk: {
            std.log.warn("failed to set raw mode: {s}; running in non-interactive one-shot mode", .{@errorName(err)});
            break :blk null;
        };
    }
    defer {
        if (original_termios) |orig| {
            restoreMode(fd, orig);
        }
    }

    try abi.features.tui.initScreen();
    defer abi.features.tui.deinitScreen();

    while (true) {
        const wdbx_stats = store.stats();
        const gpu_status = abi.features.gpu.detectBackend();
        const native_gpu = abi.features.gpu.nativeKernelStatus();

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
            .scheduler_source = "standalone CLI snapshot",
            .scheduler_running = scheduler.getRunningCount(),
            .scheduler_pending = scheduler.getPendingCount(),
            .scheduler_completed = scheduler.getCompletedCount(),
            .scheduler_failed = scheduler.getFailedCount(),
        });
        defer allocator.free(rendered);

        try abi.features.tui.clearScreen();
        std.debug.print("{s}", .{rendered});

        if (original_termios == null) {
            // Non-interactive/one-shot fallback (not a TTY or termios failed)
            break;
        }

        // Read a single character from stdin
        var key_buf: [1]u8 = undefined;
        const n = std.posix.read(fd, &key_buf) catch |err| {
            std.log.warn("read stdin failed: {s}", .{@errorName(err)});
            break;
        };
        if (n == 0) break;

        const key = key_buf[0];
        if (abi.features.tui.isQuitKey(key)) {
            break;
        }
        // If it's a refresh key or any other key, it will just loop and redraw
    }

    return 0;
}

pub const renderTui = handleDashboard;
