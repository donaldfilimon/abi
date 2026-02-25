//! Self-training orchestration: auto-train + Ralph self-improvement + optional visualization.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const auto = @import("auto.zig");
const ralph_improve = @import("../ralph/improve.zig");
const ralph_workspace = @import("../ralph/workspace.zig");
const neural_ui = @import("../ui/neural.zig");

pub fn runSelfTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var run_auto = true;
    var run_improve = true;
    var run_visualize = false;
    var multimodal = false;
    var iterations: usize = 5;
    var custom_task: ?[]const u8 = null;
    var visualize_frames: u32 = 240;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (std.mem.eql(u8, arg, "--skip-auto")) {
            run_auto = false;
            continue;
        }
        if (std.mem.eql(u8, arg, "--skip-improve")) {
            run_improve = false;
            continue;
        }
        if (std.mem.eql(u8, arg, "--multimodal")) {
            multimodal = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--visualize")) {
            run_visualize = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            iterations = try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--task")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            custom_task = std.mem.sliceTo(args[i], 0);
            continue;
        }
        if (std.mem.eql(u8, arg, "--visualize-frames")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            visualize_frames = try std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10);
            continue;
        }
    }

    if (!run_auto and !run_improve and !run_visualize) {
        utils.output.printWarning("Nothing to do: all stages are disabled.", .{});
        return;
    }

    var continue_on_error = false;
    // Check for --continue-on-error flag
    {
        var j: usize = 0;
        while (j < args.len) : (j += 1) {
            if (std.mem.eql(u8, std.mem.sliceTo(args[j], 0), "--continue-on-error")) {
                continue_on_error = true;
            }
        }
    }

    var stages_passed: u8 = 0;
    var stages_failed: u8 = 0;
    var stages_skipped: u8 = 0;

    if (run_auto) {
        utils.output.println("==> Stage 1/3: self-training", .{});
        var auto_args = std.ArrayListUnmanaged([:0]const u8).empty;
        defer auto_args.deinit(allocator);
        if (multimodal) {
            try auto_args.append(allocator, "--multimodal");
        }
        if (auto.runAutoTrain(ctx, auto_args.items)) |_| {
            utils.output.printSuccess("Stage 1/3: PASSED", .{});
            stages_passed += 1;
        } else |err| {
            utils.output.printError("Stage 1/3: FAILED — {t}", .{err});
            stages_failed += 1;
            if (!continue_on_error) {
                utils.output.printInfo("Pipeline stopped. Use --continue-on-error to proceed past failures.", .{});
                return;
            }
        }
    } else {
        utils.output.println("==> Stage 1/3: self-training skipped", .{});
        stages_skipped += 1;
    }

    if (run_improve) {
        utils.output.println("==> Stage 2/3: self-improvement loop", .{});
        var improve_args = std.ArrayListUnmanaged([:0]const u8).empty;
        defer improve_args.deinit(allocator);

        const iter_str = try std.fmt.allocPrintSentinel(allocator, "{d}", .{iterations}, 0);
        defer allocator.free(iter_str);
        try improve_args.appendSlice(allocator, &[_][:0]const u8{ "--iterations", iter_str });

        if (custom_task) |task| {
            const task_z = try allocator.dupeZ(u8, task);
            defer allocator.free(task_z);
            try improve_args.appendSlice(allocator, &[_][:0]const u8{ "--task", task_z });
        }

        if (ralph_improve.runImprove(ctx, improve_args.items)) |_| {
            utils.output.printSuccess("Stage 2/3: PASSED", .{});
            stages_passed += 1;
        } else |err| {
            utils.output.printError("Stage 2/3: FAILED — {t}", .{err});
            stages_failed += 1;
            if (!continue_on_error) {
                utils.output.printInfo("Pipeline stopped. Use --continue-on-error to proceed past failures.", .{});
                return;
            }
        }
    } else {
        utils.output.println("==> Stage 2/3: self-improvement loop skipped", .{});
        stages_skipped += 1;
    }

    if (run_visualize) {
        utils.output.println("==> Stage 3/3: dynamic neural visualization", .{});
        printLatestRalphBanner(allocator);
        const frames_str = try std.fmt.allocPrintSentinel(allocator, "{d}", .{visualize_frames}, 0);
        defer allocator.free(frames_str);
        const viz_args = [_][:0]const u8{
            "--frames",
            frames_str,
        };
        if (neural_ui.runVisualizer(allocator, &viz_args)) |_| {
            utils.output.printSuccess("Stage 3/3: PASSED", .{});
            stages_passed += 1;
        } else |err| {
            utils.output.printError("Stage 3/3: FAILED — {t}", .{err});
            stages_failed += 1;
        }
    } else {
        utils.output.println("==> Stage 3/3: visualization skipped", .{});
        stages_skipped += 1;
    }

    // Summary
    utils.output.println("", .{});
    utils.output.printHeader("Pipeline Summary");
    utils.output.printKeyValueFmt("Passed", "{d}", .{stages_passed});
    utils.output.printKeyValueFmt("Failed", "{d}", .{stages_failed});
    utils.output.printKeyValueFmt("Skipped", "{d}", .{stages_skipped});
    if (stages_failed > 0) {
        utils.output.printError("Pipeline completed with {d} failure(s).", .{stages_failed});
    } else {
        utils.output.printSuccess("Pipeline completed successfully.", .{});
    }
}

fn printLatestRalphBanner(allocator: std.mem.Allocator) void {
    var io_backend = utils.io_backend.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (ralph_workspace.latestReportPath(allocator, io)) |path| {
        defer allocator.free(path);
        utils.output.println("   latest Ralph report: {s}", .{path});
    }
}

pub fn printHelp() void {
    utils.output.print(
        \\Usage: abi train self [options]
        \\
        \\Run ABI self-improvement pipeline:
        \\  1) Self-learning auto-train
        \\  2) Ralph improve loop (code/task reasoning)
        \\  3) Optional dynamic 3D neural visualization
        \\
        \\Options:
        \\  --skip-auto               Skip auto-train stage
        \\  --skip-improve            Skip Ralph improve stage
        \\  --multimodal              Enable multimodal micro-steps in auto-train
        \\  --iterations <n>          Ralph improve loop iterations (default: 5)
        \\  --task <text>             Override Ralph improve task prompt
        \\  --visualize               Run 3D neural visualization after pipeline
        \\  --visualize-frames <n>    Frames for visualization (default: 240)
        \\  --continue-on-error       Continue pipeline even if a stage fails
        \\  -h, --help                Show this help
        \\
        \\Examples:
        \\  abi train self
        \\  abi train self --multimodal --iterations 7
        \\  abi train self --task "Analyze training drift and suggest fixes"
        \\  abi train self --visualize --visualize-frames 0
        \\  abi train self --skip-auto --iterations 3
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
