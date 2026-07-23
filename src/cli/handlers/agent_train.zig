//! `abi agent train <abbey|aviva|abi|all>` — train profiles against the durable store.

const std = @import("std");
const abi = @import("abi");
const format = @import("agent_format.zig");

pub fn handleAgentTrainProfile(io: std.Io, allocator: std.mem.Allocator, profile_arg: []const u8) !u8 {
    var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
    const artifact_dir = "zig-cache/agent-artifacts";
    const is_all = std.mem.eql(u8, profile_arg, "all");

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    var mem_tracker = abi.memory.MemoryTracker.init(allocator);
    defer mem_tracker.deinit();
    var tracking_alloc = abi.memory.TrackingAllocator.init(allocator, &mem_tracker);
    sched.setMemoryTracker(&mem_tracker);

    var arena = std.heap.ArenaAllocator.init(tracking_alloc.allocator());
    defer arena.deinit();
    const task_alloc = arena.allocator();

    var training_contexts: std.ArrayListUnmanaged(*abi.features.ai.TrainingTaskContext) = .empty;
    defer {
        for (training_contexts.items) |ctx| ctx.deinitResult();
        training_contexts.deinit(allocator);
    }

    if (is_all) {
        for (abi.features.ai.known_profiles) |p| {
            const label = p.label();
            const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{label});
            const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
            ctx.* = .{
                .allocator = allocator,
                .store = store,
                .config = .{
                    .profile = label,
                    .dataset = dataset,
                    .artifact_dir = artifact_dir,
                },
            };
            if (abi.features.ai.submitTrainingTask(&sched, name, ctx)) |_| {
                try training_contexts.append(allocator, ctx);
            } else |err| {
                if (!abi.features.ai.isFeatureDisabled(err)) return err;
            }
        }
    } else {
        const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{profile_arg});
        const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
        ctx.* = .{
            .allocator = allocator,
            .store = store,
            .config = .{
                .profile = profile_arg,
                .dataset = dataset,
                .artifact_dir = artifact_dir,
            },
        };
        if (abi.features.ai.submitTrainingTask(&sched, name, ctx)) |_| {
            try training_contexts.append(allocator, ctx);
        } else |err| {
            if (!abi.features.ai.isFeatureDisabled(err)) return err;
        }
    }

    try sched.runAll();

    format.printTrainingHeader();
    format.printSchedulerStats(sched.stats());
    format.printMemoryTrackerStats(mem_tracker.getPeakUsage(), mem_tracker.getRecordCount());

    if (training_contexts.items.len == 0) {
        const result = if (is_all)
            try abi.features.ai.trainKnownProfiles(allocator, store, dataset, artifact_dir)
        else
            try abi.features.ai.trainWithStore(allocator, store, .{
                .profile = profile_arg,
                .dataset = dataset,
                .artifact_dir = artifact_dir,
            });
        defer result.deinit(allocator);

        format.printProfileTrainingResult(result.profile, result.message, store.count(), result.acceleration_backend);
        return 0;
    }

    if (is_all) {
        var records_stored: usize = 0;
        var backend: []const u8 = "unknown";
        for (training_contexts.items) |ctx| {
            const result = ctx.result orelse return error.MissingTrainingResult;
            records_stored += result.records_stored;
            backend = result.acceleration_backend;
        }
        const message: []const u8 = if (records_stored == 0)
            "known agent profiles accepted; wdbx feature is disabled for this build"
        else
            "known agent profiles recorded in wdbx";
        format.printAllProfilesTrainingResult(message, store.count(), backend);
    } else {
        const result = training_contexts.items[0].result orelse return error.MissingTrainingResult;
        format.printProfileTrainingResult(result.profile, result.message, store.count(), result.acceleration_backend);
    }
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
