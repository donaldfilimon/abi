//! Training monitor and resume handlers.
//!
//! Handles the `abi train resume` and `abi train monitor` subcommands.
//! Resume loads a checkpoint and displays its info. Monitor provides a
//! TUI dashboard for tracking training progress.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const tui = @import("../../tui/mod.zig");
const mod = @import("mod.zig");

pub fn runResume(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        utils.output.println("Usage: abi train resume <checkpoint-path>", .{});
        return;
    }

    const checkpoint_path = std.mem.sliceTo(args[0], 0);
    utils.output.println("Loading checkpoint: {s}", .{checkpoint_path});

    // Load checkpoint
    var ckpt = abi.ai.training.loadCheckpoint(allocator, checkpoint_path) catch |err| {
        utils.output.printError("loading checkpoint: {t}", .{err});
        utils.output.println("", .{});
        utils.output.println("Note: Resume functionality loads model weights from a saved checkpoint.", .{});
        utils.output.println("Use 'abi train run --checkpoint-path <path>' to save checkpoints during training.", .{});
        return;
    };
    defer ckpt.deinit(allocator);

    utils.output.println("", .{});
    utils.output.println("Checkpoint Info:", .{});
    utils.output.printKeyValueFmt("Step", "{d}", .{ckpt.step});
    utils.output.printKeyValueFmt("Timestamp", "{d}", .{ckpt.timestamp});
    utils.output.printKeyValueFmt("Weights", "{d} parameters", .{ckpt.weights.len});
    utils.output.println("", .{});
    utils.output.printInfo("Full resume training not yet implemented.", .{});
    utils.output.printSuccess("Checkpoint loaded successfully.", .{});
}

pub fn runMonitor(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printMonitorHelp();
        return;
    }

    // Parse optional run-id argument
    var run_id: ?[]const u8 = null;
    var log_dir: []const u8 = "logs";
    var refresh_ms: u64 = 500;
    var non_interactive = false;
    var brain_mode = false;

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

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--brain")) {
            brain_mode = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--refresh-ms")) {
            if (i < args.len) {
                const value = std.mem.sliceTo(args[i], 0);
                refresh_ms = std.fmt.parseInt(u64, value, 10) catch {
                    utils.output.printError("Invalid --refresh-ms value: {s}", .{value});
                    return;
                };
                if (refresh_ms == 0) {
                    utils.output.printError("--refresh-ms must be greater than 0", .{});
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

    // Brain mode: launch brain dashboard with training data source
    if (brain_mode) {
        const brain = @import("../ui/brain.zig");
        const metrics_path = std.fmt.allocPrintSentinel(allocator, "{s}/metrics.jsonl", .{log_dir}, 0) catch |err| {
            utils.output.printError("Failed to build metrics path: {t}", .{err});
            return;
        };
        defer allocator.free(metrics_path);

        const brain_args = [_][:0]const u8{ "--training", metrics_path };
        brain.run(ctx, &brain_args) catch {
            utils.output.printWarning("Brain dashboard mode requires a terminal.", .{});
            utils.output.println("Use 'abi ui brain --training {s}/metrics.jsonl' instead.", .{log_dir});
        };
        return;
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
            utils.output.printWarning("Interactive mode failed ({t}), falling back to snapshot mode.", .{err});
            utils.output.println("", .{});
            non_interactive = true;
        };

        if (!non_interactive) return;
    }

    // Non-interactive fallback: render single snapshot
    const DebugWriter = struct {
        pub const Error = error{};
        pub fn print(_: @This(), comptime fmt: []const u8, print_args: anytype) Error!void {
            utils.output.print(fmt, print_args);
        }
    };

    // Load metrics before rendering
    panel.loadMetricsFile(panel.buildMetricsPath()) catch {};

    panel.render(DebugWriter{}) catch |err| {
        utils.output.printError("rendering panel: {t}", .{err});
        return;
    };

    utils.output.println("", .{});
    utils.output.println("Training Monitor (snapshot mode)", .{});
    utils.output.printKeyValueFmt("Log directory", "{s}", .{log_dir});
    if (run_id) |id| {
        utils.output.printKeyValueFmt("Run ID", "{s}", .{id});
    } else {
        utils.output.println("Monitoring: current/latest run", .{});
    }
    utils.output.println("", .{});
    utils.output.printInfo("Run without --no-tui for interactive mode.", .{});
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
        \\  --brain             Launch 3D brain visualization instead of panel
        \\
        \\Arguments:
        \\  run-id              Optional run ID to monitor (default: latest)
        \\
        \\Keyboard controls:
        \\  r       Refresh display
        \\  h       Toggle history mode
        \\  q       Quit
        \\  ?       Show help
        \\  <-/->     Switch between runs (history mode)
        \\
        \\Examples:
        \\  abi train monitor                    # Monitor latest run
        \\  abi train monitor run-2026-01-24     # Monitor specific run
        \\  abi train monitor --log-dir ./logs   # Custom log directory
        \\  abi train monitor --refresh-ms 250   # Faster refresh cadence
        \\  abi train monitor --no-tui           # Snapshot mode
        \\
    ;
    utils.output.print("{s}", .{help_text});
}
