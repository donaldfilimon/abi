const std = @import("std");
const features = @import("../../features/mod.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");
const usage_mod = @import("../usage.zig");
const credentials = @import("../../foundation/credentials.zig");
const anthropic = @import("../../connectors/anthropic.zig");

pub fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

/// `abi complete [--live] [--model <id>] <input>`.
///
/// `model` is the raw `--model` value (or null for the default). It is
/// alias-resolved at this edge through the model catalog so `fable-5` records
/// the canonical `claude-fable-5`. With `live`, an anthropic-provider model is
/// served by the real `anthropic.messageLive` path using stored credentials;
/// otherwise the local persona router runs and the completion is persisted.
pub fn handleComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: ?[]const u8, live: bool) !u8 {
    const selected_model = if (model) |m| features.ai.models.canonical(m) else features.ai.models.default_model;

    // Pass-through of unknown ids is the documented contract, but a silent
    // pass-through hides typos (e.g. `claud-fable-5`). Surface a one-line note
    // on stderr without changing what gets recorded.
    if (model) |m| {
        if (!features.ai.models.isKnown(m)) {
            std.debug.print("warning: '{s}' is not a recognized model id; passing it through unchanged\n", .{m});
        }
    }

    if (live) return handleLiveComplete(io, allocator, input, selected_model);

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
        .{ .input = input, .model = selected_model, .store_result = true },
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

/// Stage 2: the live anthropic path behind `--live`. Only anthropic-provider
/// models are supported; the API key is read from the credential store and the
/// request crosses the explicit `.live` transport boundary.
fn handleLiveComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: []const u8) !u8 {
    if (features.ai.models.providerOf(model) != .anthropic) {
        return usage_mod.usageError("--live currently supports anthropic models only (e.g. --model fable-5)");
    }

    var creds = credentials.loadCredentials(allocator) catch {
        std.debug.print("error: no credentials found; run `abi auth signin anthropic`\n", .{});
        return 2;
    };
    defer creds.deinit(allocator);

    const api_key = creds.anthropic_api_key orelse {
        std.debug.print("error: no anthropic credentials configured; run `abi auth signin anthropic`\n", .{});
        return 2;
    };

    var client = anthropic.Client.init(allocator, .{
        .api_key = api_key,
        .base_url = "https://api.anthropic.com",
        .transport = .live,
    });
    defer client.deinit();

    var resp = client.messageLive(io, allocator, model, input, 1024) catch |err| {
        std.debug.print("error: anthropic live request failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer resp.deinit(allocator);

    std.debug.print("model={s} provider=anthropic transport=live status={d}\n", .{ model, resp.status });
    std.debug.print("{s}\n", .{resp.body});
    return if (resp.status >= 200 and resp.status < 300) 0 else 1;
}

test "complete --live rejects non-anthropic models before any network or credential read" {
    const allocator = std.testing.allocator;
    // `abi-local` is a known catalog model whose provider is `.local`, so the
    // live path must reject it with usage (exit 2) before touching the
    // credential store or the network transport.
    const code = try handleComplete(std.testing.io, allocator, "hello", "abi-local", true);
    try std.testing.expectEqual(@as(u8, 2), code);
}

test {
    std.testing.refAllDecls(@This());
}
