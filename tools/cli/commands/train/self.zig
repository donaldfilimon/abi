//! Self-training orchestration: auto-train + Ralph self-improvement + optional visualization.

const std = @import("std");
const utils = @import("../../utils/mod.zig");
const auto = @import("auto.zig");
const ralph_improve = @import("../ralph/improve.zig");
const neural_ui = @import("../ui/neural.zig");

pub fn runSelfTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
        std.debug.print("Nothing to do: all stages are disabled.\n", .{});
        return;
    }

    if (run_auto) {
        std.debug.print("==> Stage 1/3: self-training\n", .{});
        var auto_args = std.ArrayListUnmanaged([:0]const u8).empty;
        defer auto_args.deinit(allocator);
        if (multimodal) {
            try auto_args.append(allocator, "--multimodal");
        }
        try auto.runAutoTrain(allocator, auto_args.items);
    } else {
        std.debug.print("==> Stage 1/3: self-training skipped\n", .{});
    }

    if (run_improve) {
        std.debug.print("==> Stage 2/3: self-improvement loop\n", .{});
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

        try ralph_improve.runImprove(allocator, improve_args.items);
    } else {
        std.debug.print("==> Stage 2/3: self-improvement loop skipped\n", .{});
    }

    if (run_visualize) {
        std.debug.print("==> Stage 3/3: dynamic neural visualization\n", .{});
        const frames_str = try std.fmt.allocPrintSentinel(allocator, "{d}", .{visualize_frames}, 0);
        defer allocator.free(frames_str);
        const viz_args = [_][:0]const u8{
            "--frames",
            frames_str,
        };
        try neural_ui.runVisualizer(allocator, &viz_args);
    } else {
        std.debug.print("==> Stage 3/3: visualization skipped\n", .{});
    }
}

pub fn printHelp() void {
    std.debug.print(
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
