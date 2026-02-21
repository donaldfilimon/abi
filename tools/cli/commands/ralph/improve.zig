//! ralph improve — self-improvement loop

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runImprove(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task: []const u8 =
        "Review the ABI framework source in src/ for Zig 0.16 migration issues. " ++
        "Check mod.zig/stub.zig parity. Verify build flags in build/options.zig. " ++
        "Run zig build test mentally and identify likely failures. " ++
        "Produce a prioritized list of fixes with exact file paths.";
    var max_iterations: usize = 5;
    var auto_skill = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) task = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) max_iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_iterations;
        } else if (std.mem.eql(u8, arg, "--no-auto-skill")) {
            auto_skill = false;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph improve [options]
                \\
                \\Self-improvement loop: analyze source, identify issues, extract lesson.
                \\Auto-skill is enabled by default.
                \\
                \\Options:
                \\  -t, --task <text>      Custom improvement task
                \\  -i, --iterations <n>   Max iterations (default: 5)
                \\      --no-auto-skill    Disable automatic skill extraction
                \\  -h, --help             Show this help
                \\
            , .{});
            return;
        }
    }

    std.debug.print(
        "Starting Ralph self-improvement loop ({d} iterations)...\n",
        .{max_iterations},
    );
    std.debug.print("Task: {s}\n\n", .{task});

    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    const result = engine.runRalphLoop(task, max_iterations) catch |err| {
        std.debug.print("Improve loop failed: {t}\n", .{err});
        return;
    };
    defer allocator.free(result);

    engine.recordRalphRun(task, max_iterations, result.len, 1.0) catch {};

    if (auto_skill) {
        const stored = engine.extractAndStoreSkill(task, result) catch false;
        if (stored) {
            std.debug.print("Lesson extracted and stored in Abbey memory.\n\n", .{});
        }
    }

    std.debug.print("\n=== Improvement Analysis ===\n{s}\n", .{result});
}
