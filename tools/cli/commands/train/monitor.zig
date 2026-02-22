//! Training monitor and resume handlers.
//!
//! Handles the `abi train resume` and `abi train monitor` subcommands.
//! Resume loads a checkpoint and displays its info. Monitor provides a
//! TUI dashboard for tracking training progress.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const tui = @import("../../tui/mod.zig");
const mod = @import("mod.zig");

pub fn runResume(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi train resume <checkpoint-path>\n", .{});
        return;
    }

    const checkpoint_path = std.mem.sliceTo(args[0], 0);
    std.debug.print("Loading checkpoint: {s}\n", .{checkpoint_path});

    // Load checkpoint
    var ckpt = abi.ai.training.loadCheckpoint(allocator, checkpoint_path) catch |err| {
        std.debug.print("Error loading checkpoint: {t}\n", .{err});
        std.debug.print("\nNote: Resume functionality loads model weights from a saved checkpoint.\n", .{});
        std.debug.print("Use 'abi train run --checkpoint-path <path>' to save checkpoints during training.\n", .{});
        return;
    };
    defer ckpt.deinit(allocator);

    std.debug.print("\nCheckpoint Info:\n", .{});
    std.debug.print("  Step:      {d}\n", .{ckpt.step});
    std.debug.print("  Timestamp: {d}\n", .{ckpt.timestamp});
    std.debug.print("  Weights:   {d} parameters\n", .{ckpt.weights.len});
    std.debug.print("\nNote: Full resume training not yet implemented.\n", .{});
    std.debug.print("Checkpoint loaded successfully.\n", .{});
}

pub fn runMonitor(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printMonitorHelp();
        return;
    }

    // Parse optional run-id argument
    var run_id: ?[]const u8 = null;
    var log_dir: []const u8 = "logs";
    var refresh_ms: u64 = 500;
    var non_interactive = false;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-dir")) {
            if (i < args.len) {
                log_dir = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-tui") or
            std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--non-interactive"))
        {
            non_interactive = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--refresh-ms")) {
            if (i < args.len) {
                const value = std.mem.sliceTo(args[i], 0);
                refresh_ms = std.fmt.parseInt(u64, value, 10) catch {
                    std.debug.print("Invalid --refresh-ms value: {s}\n", .{value});
                    return;
                };
                if (refresh_ms == 0) {
                    std.debug.print("--refresh-ms must be greater than 0\n", .{});
                    return;
                }
                i += 1;
            }
            continue;
        }

        // First non-option argument is run-id
        if (run_id == null and arg[0] != '-') {
            run_id = std.mem.sliceTo(arg, 0);
        }
    }

    // Use the default theme from the theme system
    const theme = &tui.themes.themes.default;

    var panel = tui.TrainingPanel.init(allocator, theme, .{
        .log_dir = log_dir,
        .run_id = run_id,
        .refresh_ms = refresh_ms,
    });
    defer panel.deinit();

    // Try interactive mode if terminal is supported
    if (!non_interactive and tui.Terminal.isSupported()) {
        var terminal = tui.Terminal.init(allocator);
        defer terminal.deinit();

        panel.runInteractive(&terminal) catch |err| {
            // Fall back to non-interactive mode on error
            std.debug.print("Interactive mode failed ({t}), falling back to snapshot mode.\n\n", .{err});
            non_interactive = true;
        };

        if (!non_interactive) return;
    }

    // Non-interactive fallback: render single snapshot
    const DebugWriter = struct {
        pub const Error = error{};
        pub fn print(_: @This(), comptime fmt: []const u8, print_args: anytype) Error!void {
            std.debug.print(fmt, print_args);
        }
    };

    // Load metrics before rendering
    panel.loadMetricsFile(panel.buildMetricsPath()) catch {};

    panel.render(DebugWriter{}) catch |err| {
        std.debug.print("Error rendering panel: {t}\n", .{err});
        return;
    };

    std.debug.print("\nTraining Monitor (snapshot mode)\n", .{});
    std.debug.print("Log directory: {s}\n", .{log_dir});
    if (run_id) |id| {
        std.debug.print("Run ID: {s}\n", .{id});
    } else {
        std.debug.print("Monitoring: current/latest run\n", .{});
    }
    std.debug.print("\nRun without --no-tui for interactive mode.\n", .{});
}

pub fn printMonitorHelp() void {
    const help_text =
        \\Usage: abi train monitor [run-id] [options]
        \\
        \\Monitor training progress with a TUI dashboard.
        \\
        \\Options:
        \\  --log-dir <path>    Log directory (default: logs)
        \\  --refresh-ms <n>    Refresh interval in milliseconds (default: 500)
        \\  --no-tui            Render a single snapshot and exit
        \\
        \\Arguments:
        \\  run-id              Optional run ID to monitor (default: latest)
        \\
        \\Keyboard controls:
        \\  r       Refresh display
        \\  h       Toggle history mode
        \\  q       Quit
        \\  ?       Show help
        \\  ←/→     Switch between runs (history mode)
        \\
        \\Examples:
        \\  abi train monitor                    # Monitor latest run
        \\  abi train monitor run-2026-01-24     # Monitor specific run
        \\  abi train monitor --log-dir ./logs   # Custom log directory
        \\  abi train monitor --refresh-ms 250   # Faster refresh cadence
        \\  abi train monitor --no-tui           # Snapshot mode
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
