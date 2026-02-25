//! ralph multi — Zig-native multithreaded multi-agent (ThreadPool + RalphBus)

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runMulti(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var tasks_list = std.ArrayListUnmanaged([]const u8).empty;
    defer tasks_list.deinit(allocator);
    var max_iterations: usize = 20;
    var workers: u32 = 0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) try tasks_list.append(allocator, std.mem.sliceTo(args[i], 0));
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) max_iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_iterations;
        } else if (std.mem.eql(u8, arg, "--workers")) {
            i += 1;
            if (i < args.len) workers = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch 0;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            utils.output.print(
                \\Usage: abi ralph multi [options]
                \\
                \\Zig-native multithreaded multi-agent: N Ralph loops in parallel via ThreadPool + RalphBus.
                \\
                \\Options:
                \\  -t, --task <text>     Add a task (repeat for multiple agents)
                \\  -i, --iterations <n>  Max iterations per agent (default: 20)
                \\      --workers <n>     Thread pool size (default: CPU count)
                \\  -h, --help            Show this help
                \\
            , .{});
            return;
        }
    }

    if (tasks_list.items.len == 0) {
        utils.output.printWarning("No tasks. Use -t/--task at least once (e.g. abi ralph multi -t \"goal1\" -t \"goal2\").", .{});
        return;
    }

    const goals = tasks_list.items;
    var bus = try abi.ai.abbey.ralph_multi.RalphBus.init(allocator, 64);
    defer bus.deinit();

    var pool = try abi.runtime.ThreadPool.init(allocator, .{
        .thread_count = if (workers > 0) workers else 0,
    });
    defer pool.deinit();

    const results = try allocator.alloc(?[]const u8, goals.len);
    defer {
        for (results) |r| if (r) |s| allocator.free(s);
        allocator.free(results);
    }
    for (results) |*r| r.* = null;

    var parallel_ctx: abi.ai.abbey.ralph_swarm.ParallelRalphContext = .{
        .allocator = allocator,
        .bus = &bus,
        .goals = goals,
        .results = results,
        .max_iterations = max_iterations,
        .post_result_to_bus = true,
    };

    utils.output.printInfo("Ralph multi: {d} agents, {d} threads, {d} iterations each.", .{
        goals.len,
        pool.thread_count,
        max_iterations,
    });

    for (goals, 0..) |_, idx| {
        const uidx: u32 = @intCast(idx);
        if (!pool.schedule(abi.ai.abbey.ralph_swarm.parallelRalphWorker, .{ &parallel_ctx, uidx })) {
            utils.output.printError("Schedule failed for agent {d}", .{idx});
            return;
        }
    }
    pool.waitIdle();

    utils.output.printHeader("Results");
    for (results, 0..) |r, idx| {
        utils.output.println("--- Agent {d} ---", .{idx});
        if (r) |s| utils.output.println("{s}", .{s}) else utils.output.println("(failed)", .{});
    }

    var msg_count: usize = 0;
    while (bus.tryRecv()) |msg| {
        msg_count += 1;
        utils.output.println("[bus] from={d} to={d} kind={t}: {s}", .{
            msg.from_id,
            msg.to_id,
            msg.kind,
            msg.getContent(),
        });
    }
    if (msg_count > 0) utils.output.println("Bus messages: {d}", .{msg_count});
}
