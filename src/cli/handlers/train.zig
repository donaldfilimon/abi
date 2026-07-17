const std = @import("std");
const test_helpers = @import("abi").foundation.test_helpers;
const features = @import("abi").features;
const scheduler_mod = @import("abi").scheduler;
const memory_mod = @import("abi").memory;
const usage_mod = @import("../usage.zig");
const env = @import("abi").foundation.env;
const credentials = @import("abi").foundation.credentials;
const anthropic = @import("abi").connectors.anthropic;
const fm = @import("abi").connectors.fm;
const connectors = @import("abi").connectors;
const complete = @import("complete_handlers.zig");

/// `abi train <input>`: run the local AI persona router over `input` and print
/// the response. Returns the process exit code.
pub fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

/// `abi complete [--live] [--confirm] [--model <id>] [--learn] [--stream] [--soul <file>] [--soul-alpha <f32>] <input>`.
pub const CompleteOptions = struct {
    input: []const u8,
    model: ?[]const u8 = null,
    live: bool = false,
    confirmed: bool = false,
    learn: bool = false,
    stream: bool = false,
    soul: ?[]const u8 = null,
    soul_alpha: f32 = 0.5,
};

const LearnMetadata = struct {
    evidence_count: usize,
    adapted: bool,
};

fn printCompletionMetadata(completion: *const features.ai.CompletionResult, stats: anytype, learn: ?LearnMetadata, skip_output: bool) void {
    const persisted = completion.query_vector_id != null and completion.response_vector_id != null and completion.block_id != null;

    if (learn) |meta| {
        std.debug.print("model={s} profile={s} audit_passed={s} audit_escore={d:.3} audit_vetoed={s} persisted={s} learn=true evidence_count={d} adapted={s}\n", .{
            completion.model,
            completion.selected_profile.label(),
            if (completion.audit.passed) "true" else "false",
            completion.audit.escore,
            if (completion.audit.vetoed) "true" else "false",
            if (persisted) "true" else "false",
            meta.evidence_count,
            if (meta.adapted) "true" else "false",
        });
    } else {
        std.debug.print("model={s} profile={s} audit_passed={s} audit_escore={d:.3} audit_vetoed={s} persisted={s}\n", .{
            completion.model,
            completion.selected_profile.label(),
            if (completion.audit.passed) "true" else "false",
            completion.audit.escore,
            if (completion.audit.vetoed) "true" else "false",
            if (persisted) "true" else "false",
        });
    }

    std.debug.print("wdbx kv_entries={d} vectors={d} blocks={d}\n", .{ stats.kv_entries, stats.vectors, stats.blocks });
    if (completion.query_vector_id) |qid| {
        std.debug.print("query_vector_id={d}\n", .{qid});
        std.debug.print("metadata_key=completion:{d}\n", .{qid});
    }
    if (completion.response_vector_id) |rid| std.debug.print("response_vector_id={d}\n", .{rid});
    if (completion.block_id) |block_id| {
        const block_hex = std.fmt.bytesToHex(block_id, .lower);
        std.debug.print("block_id={s}\n", .{&block_hex});
    }
    if (!persisted) std.debug.print("wdbx_status={s}\n", .{stats.acceleration.message});
    if (!skip_output) std.debug.print("{s}\n", .{completion.output});
}

pub fn handleComplete(io: std.Io, allocator: std.mem.Allocator, opts: CompleteOptions) !u8 {
    const input = opts.input;
    const selected_model = if (opts.model) |m| features.ai.models.canonical(m) else features.ai.models.default_model;

    if (opts.model) |m| {
        if (!features.ai.models.isKnown(m)) {
            std.debug.print("warning: '{s}' is not a recognized model id; passing it through unchanged\n", .{m});
        }
    }

    if (opts.live) {
        if (features.ai.models.providerOf(selected_model) == .fm) {
            return complete.handleFmComplete(allocator, input, selected_model, opts.confirmed);
        }
        return complete.handleLiveComplete(io, allocator, input, selected_model, opts.stream);
    }

    if (connectors.local_bridge.isLocalBridgeModel(selected_model)) {
        return complete.handleLocalBridgeComplete(io, allocator, input, selected_model, opts.stream);
    }

    if (opts.soul) |soul_path| {
        return complete.handleSoulComplete(io, allocator, input, selected_model, opts.soul_alpha, soul_path);
    }

    var session = try features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    if (opts.learn) {
        if (opts.stream) {
            const StreamCtx = struct {
                fn callback(_: *anyopaque, chunk: features.ai.StreamChunk) anyerror!void {
                    if (chunk.delta.len > 0) std.debug.print("{s}", .{chunk.delta});
                }
            };
            var dummy: u8 = 0;
            var result = try features.sea.runLearnLoop(allocator, store, input, selected_model, .{
                .stream_callback = StreamCtx.callback,
                .stream_ctx = &dummy,
            });
            defer result.deinit(allocator);
            const completion = result.completion;
            const stats = store.stats();
            std.debug.print("\n", .{});
            printCompletionMetadata(&completion, stats, .{ .evidence_count = result.evidence_count, .adapted = result.adapted }, true);
            return 0;
        }
        return complete.handleLearnComplete(allocator, store, input, selected_model);
    }

    var scheduler = scheduler_mod.Scheduler.init(allocator);
    defer scheduler.deinit();
    var tracker = memory_mod.MemoryTracker.init(allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);

    if (opts.stream) {
        const StreamCtx = struct {
            fn callback(_: *anyopaque, chunk: features.ai.StreamChunk) anyerror!void {
                if (chunk.delta.len > 0) std.debug.print("{s}", .{chunk.delta});
            }
        };
        var dummy: u8 = 0;
        var result = try features.ai.completeWithSchedulerStreaming(
            allocator,
            store,
            &scheduler,
            "complete:cli",
            .{ .input = input, .model = selected_model, .store_result = true },
            StreamCtx.callback,
            @ptrCast(&dummy),
        );
        defer result.deinit(allocator);
        std.debug.print("\n", .{});
        const stats = store.stats();
        printCompletionMetadata(&result, stats, null, false);
        return 0;
    }

    var result = try features.ai.completeWithScheduler(
        allocator,
        store,
        &scheduler,
        "complete:cli",
        .{ .input = input, .model = selected_model, .store_result = true },
    );
    defer result.deinit(allocator);

    const stats = store.stats();
    printCompletionMetadata(&result, stats, null, false);
    return 0;
}

test "complete --live rejects non-anthropic models before any network or credential read" {
    const allocator = std.testing.allocator;
    const code = try handleComplete(std.testing.io, allocator, .{ .input = "hello", .model = "abi-local", .live = true });
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "complete --live --stream rejects non-anthropic models on the same branch" {
    const allocator = std.testing.allocator;
    const code = try handleComplete(std.testing.io, allocator, .{
        .input = "hello",
        .model = "abi-local",
        .live = true,
        .stream = true,
    });
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "complete --live apple-fm without --confirm rejects with usage before any inference" {
    const allocator = std.testing.allocator;
    const code = try handleComplete(std.testing.io, allocator, .{ .input = "hello", .model = "apple-fm", .live = true });
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "complete --live apple-fm with --confirm tracks on-device availability" {
    const allocator = std.testing.allocator;
    const code = try handleComplete(std.testing.io, allocator, .{ .input = "hello", .model = "apple-fm", .live = true, .confirmed = true });
    const expected: u8 = if (fm.fmAvailable()) 0 else 1;
    try std.testing.expectEqual(expected, code);
}

test "complete --soul with a missing layout file fails before any store or session" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.FileNotFound, handleComplete(std.testing.io, allocator, .{
        .input = "hello",
        .soul = "zig-out/definitely-missing-soul-layout.json",
    }));
}

test "complete --learn routes through the SEA loop against an in-memory store" {
    const allocator = std.testing.allocator;
    var store = features.wdbx.Store.init(allocator);
    defer store.deinit();
    const code = try complete.handleLearnComplete(allocator, &store, "learn from this", features.ai.models.default_model);
    try std.testing.expectEqual(@as(u8, 0), code);
}

test {
    std.testing.refAllDecls(@This());
}
