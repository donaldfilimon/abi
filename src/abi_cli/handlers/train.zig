const std = @import("std");
const features = @import("../../features/mod.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");

pub fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

pub fn handleComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8) !u8 {
    var session = try features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    var scheduler = scheduler_mod.Scheduler.init(allocator);
    defer scheduler.deinit();
    var tracker = memory_mod.MemoryTracker.init(allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);

    var result = try features.ai.completeWithScheduler(
        allocator,
        store,
        &scheduler,
        "complete:cli",
        .{ .input = input, .model = "abi-local", .store_result = true },
    );
    defer result.deinit(allocator);

    const stats = store.stats();
    const persisted = result.query_vector_id != null and result.response_vector_id != null and result.block_id != null;

    std.debug.print("model={s} profile={s} audit_passed={s} persisted={s}\n", .{ result.model, result.selected_profile.label(), if (result.audit.passed) "true" else "false", if (persisted) "true" else "false" });
    std.debug.print("wdbx kv_entries={d} vectors={d} blocks={d}\n", .{ stats.kv_entries, stats.vectors, stats.blocks });
    if (result.query_vector_id) |qid| {
        std.debug.print("query_vector_id={d}\n", .{qid});
        std.debug.print("metadata_key=completion:{d}\n", .{qid});
    }
    if (result.response_vector_id) |rid| std.debug.print("response_vector_id={d}\n", .{rid});
    if (result.block_id) |block_id| {
        const block_hex = std.fmt.bytesToHex(block_id, .lower);
        std.debug.print("block_id={s}\n", .{&block_hex});
    }
    if (!persisted) std.debug.print("wdbx_status={s}\n", .{stats.acceleration.message});
    std.debug.print("{s}\n", .{result.output});
    return 0;
}
