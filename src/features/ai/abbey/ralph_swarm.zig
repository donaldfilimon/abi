//! Multi-agent Ralph swarm: Zig-native, fast multithreading.
//!
//! Runs N Ralph loops in parallel using the runtime ThreadPool and a shared
//! RalphBus for lock-free message passing. Each agent gets its own Abbey engine
//! and a goal; results are written to a shared slice and optionally posted to
//! the bus (task_result / skill_share).
//!
//! Usage (from CLI or app that has both abbey and runtime):
//!   var bus = try RalphBus.init(allocator, 64);
//!   defer bus.deinit();
//!   var results = try allocator.alloc(?[]const u8, goals.len);
//!   defer for (results) |r| if (r) |s| allocator.free(s);
//!   defer allocator.free(results);
//!   try runParallelRalphs(allocator, pool, &bus, goals, results, 20);
//!   pool.waitIdle();

const std = @import("std");
const ralph_multi = @import("ralph_multi.zig");
const engine_mod = @import("engine.zig");

/// Context for one parallel Ralph worker. Use with ThreadPool.schedule(parallelRalphWorker, .{ ctx, index }).
/// Allocator must be thread-safe (e.g. std.heap.page_allocator or std.heap.ThreadSafeAllocator).
pub const ParallelRalphContext = struct {
    allocator: std.mem.Allocator,
    bus: *ralph_multi.RalphBus,
    goals: []const []const u8,
    results: []?[]const u8,
    max_iterations: usize,
    /// Post a short summary to the bus when done (content truncated to max_message_content_len).
    post_result_to_bus: bool = true,
};

/// Worker to run one Ralph loop. Schedule from CLI/app: pool.schedule(abi.ai.abbey.ralph_swarm.parallelRalphWorker, .{ &ctx, index }).
/// Fits ThreadPool.Task capture: *ParallelRalphContext + u32.
pub fn parallelRalphWorker(ctx: *ParallelRalphContext, index: u32) void {
    if (index >= ctx.goals.len) return;
    const goal = ctx.goals[index];
    var eng = engine_mod.AbbeyEngine.init(ctx.allocator, .{}) catch {
        ctx.results[index] = null;
        return;
    };
    defer eng.deinit();

    const result = eng.runRalphLoop(goal, ctx.max_iterations) catch {
        ctx.results[index] = null;
        return;
    };
    defer ctx.allocator.free(result);

    ctx.results[index] = ctx.allocator.dupe(u8, result) catch {
        ctx.results[index] = null;
        return;
    };

    if (ctx.post_result_to_bus) {
        var msg: ralph_multi.RalphMessage = .{
            .from_id = index + 1,
            .to_id = 0,
            .kind = .task_result,
        };
        const copy_len = @min(result.len, ralph_multi.max_message_content_len);
        @memcpy(msg.content[0..copy_len], result[0..copy_len]);
        msg.content_len = @intCast(copy_len);
        _ = ctx.bus.trySend(msg);
    }
}

test "ralph_swarm worker args fit typical task capture" {
    const capture_size = @sizeOf(*ParallelRalphContext) + @sizeOf(u32);
    try std.testing.expect(capture_size <= 128);
}
