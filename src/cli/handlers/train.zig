const std = @import("std");
const features = @import("../../features/mod.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");
const usage_mod = @import("../usage.zig");
const credentials = @import("../../foundation/credentials.zig");
const anthropic = @import("../../connectors/anthropic.zig");
const fm = @import("../../connectors/fm.zig");

/// `abi train <input>`: run the local AI persona router over `input` and print
/// the response. Returns the process exit code.
pub fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

/// `abi complete [--live] [--confirm] [--model <id>] <input>`.
///
/// `model` is the raw `--model` value (or null for the default). It is
/// alias-resolved at this edge through the model catalog so `fable-5` records
/// the canonical `claude-fable-5`. With `live`, an anthropic-provider model is
/// served by the real `anthropic.messageLive` path using stored credentials;
/// an `apple-fm` (on-device FoundationModels) model requires `--confirm` and is
/// routed to `handleFmComplete`; otherwise the local persona router runs and the
/// completion is persisted.
pub fn handleComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: ?[]const u8, live: bool, confirmed: bool, learn: bool) !u8 {
    const selected_model = if (model) |m| features.ai.models.canonical(m) else features.ai.models.default_model;

    // Pass-through of unknown ids is the documented contract, but a silent
    // pass-through hides typos (e.g. `claud-fable-5`). Surface a one-line note
    // on stderr without changing what gets recorded.
    if (model) |m| {
        if (!features.ai.models.isKnown(m)) {
            std.debug.print("warning: '{s}' is not a recognized model id; passing it through unchanged\n", .{m});
        }
    }

    if (live) {
        if (features.ai.models.providerOf(selected_model) == .fm) {
            return handleFmComplete(allocator, input, selected_model, confirmed);
        }
        return handleLiveComplete(io, allocator, input, selected_model);
    }

    var session = try features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    // `--learn` routes through the SEA self-learning loop instead of the plain
    // scheduler-backed completion. `features.sea` is selected at build time, so
    // the flag is always accepted: with `-Dfeat-sea=false` the stub degrades to
    // a plain persisted completion (evidence_count=0, adapted=false); with the
    // feature on it recalls evidence and adapts the persona-router weights.
    if (learn) return handleLearnComplete(allocator, store, input, selected_model);

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

/// `abi complete --learn`: run one SEA self-learning pass against the durable
/// store and report a one-line meta that includes `evidence_count` (recalled
/// records) and `adapted` (whether the persona-router weights were updated).
/// `runLearnLoop` owns the completion; `LearnLoopResult.deinit` frees it.
fn handleLearnComplete(
    allocator: std.mem.Allocator,
    store: *features.wdbx.Store,
    input: []const u8,
    model: []const u8,
) !u8 {
    var result = try features.sea.runLearnLoop(allocator, store, input, model, .{});
    defer result.deinit(allocator);

    const completion = result.completion;
    const stats = store.stats();
    const persisted = completion.query_vector_id != null and completion.response_vector_id != null and completion.block_id != null;

    std.debug.print("model={s} profile={s} audit_passed={s} persisted={s} learn=true evidence_count={d} adapted={s}\n", .{
        completion.model,
        completion.selected_profile.label(),
        if (completion.audit.passed) "true" else "false",
        if (persisted) "true" else "false",
        result.evidence_count,
        if (result.adapted) "true" else "false",
    });
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
    std.debug.print("{s}\n", .{completion.output});
    return 0;
}

/// Stage 2: the live anthropic path behind `--live`. Only anthropic-provider
/// models are supported; the API key is read from the credential store and the
/// request crosses the explicit `.live` transport boundary.
fn handleLiveComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: []const u8) !u8 {
    if (features.ai.models.providerOf(model) != .anthropic) {
        return usage_mod.usageError("--live currently supports anthropic models only (e.g. --model fable-5)");
    }

    var creds = credentials.loadCredentials(allocator) catch |err| switch (err) {
        // A genuinely absent store is the friendly first-run case.
        error.FileNotFound => {
            std.debug.print("error: no credentials found; run `abi auth signin anthropic`\n", .{});
            return 2;
        },
        // A present-but-unreadable/corrupt store is a distinct failure: surface
        // the underlying error so it isn't mistaken for "not signed in".
        else => {
            std.debug.print("error: failed to load credentials ({s}); run `abi auth signin anthropic`\n", .{@errorName(err)});
            return 2;
        },
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

/// On-device Apple FoundationModels path behind `--live --model apple-fm`.
///
/// On-device generation runs local model weights, so it is gated behind an
/// explicit `--confirm` (mirroring `agent os execute --confirm`). The connector's
/// bridge is comptime-gated behind `-Dfeat-foundationmodels` on macOS; absent
/// that flag — or because FoundationModels is a Swift-only framework with no
/// reachable ObjC entry point yet (see `connectors/fm.zig`) — `completeLive`
/// reports `error.FMUnavailable`, surfaced as a clear diagnostic that exits 1
/// rather than pretending to run.
fn handleFmComplete(allocator: std.mem.Allocator, input: []const u8, model: []const u8, confirmed: bool) !u8 {
    if (!confirmed) {
        return usage_mod.usageError("on-device apple-fm requires --confirm (e.g. `abi complete --live --model apple-fm --confirm <input>`)");
    }

    var client = fm.Client.init(allocator, .{});
    defer client.deinit();

    var resp = client.completeLive(allocator, input) catch |err| switch (err) {
        error.FMUnavailable => {
            std.debug.print(
                "error: on-device FoundationModels unavailable for model={s}: not built with -Dfeat-foundationmodels, not running on macOS, or the on-device runtime is not reachable on this host\n",
                .{model},
            );
            return 1;
        },
        else => {
            std.debug.print("error: on-device FoundationModels request failed: {s}\n", .{@errorName(err)});
            return 1;
        },
    };
    defer resp.deinit(allocator);

    std.debug.print("model={s} provider=fm transport=on-device status={d}\n", .{ model, resp.status });
    std.debug.print("{s}\n", .{resp.body});
    return if (resp.status >= 200 and resp.status < 300) 0 else 1;
}

test "complete --live rejects non-anthropic models before any network or credential read" {
    const allocator = std.testing.allocator;
    // `abi-local` is a known catalog model whose provider is `.local`, so the
    // live path must reject it with usage (exit 2) before touching the
    // credential store or the network transport.
    const code = try handleComplete(std.testing.io, allocator, "hello", "abi-local", true, false, false);
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "complete --live apple-fm without --confirm rejects with usage before any inference" {
    const allocator = std.testing.allocator;
    // apple-fm routes to the on-device path, which must refuse with usage
    // (exit 2) until `--confirm` is supplied — no client is constructed.
    const code = try handleComplete(std.testing.io, allocator, "hello", "apple-fm", true, false, false);
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "complete --live apple-fm with --confirm tracks on-device availability" {
    const allocator = std.testing.allocator;
    // With --confirm the FM client is constructed and the on-device path runs.
    // The exit code must track REAL availability, not merely fall in {0,1}:
    //   exit 0 — FoundationModels is built in AND reachable (arm64 macOS +
    //            -Dfeat-foundationmodels + Apple-Intelligence hardware); it served
    //            the on-device completion.
    //   exit 1 — otherwise (flag off, off-platform, x86_64 macOS, or a
    //            non-Apple-Intelligence host): the FMUnavailable diagnostic fires.
    // `fm.fmAvailable()` is the exact same gate handleComplete resolves, so pinning
    // the expected code to it keeps the suite green on every host while still
    // catching a real regression — e.g. a fabricated success when FM is unavailable
    // (would-be exit 0), a dropped FMUnavailable guard, or a fall-through to usage
    // (exit 2) — that the loose `0 or 1` union would have silently accepted.
    const code = try handleComplete(std.testing.io, allocator, "hello", "apple-fm", true, true, false);
    const expected: u8 = if (fm.fmAvailable()) 0 else 1;
    try std.testing.expectEqual(expected, code);
}

test "complete --learn routes through the SEA loop against an in-memory store" {
    const allocator = std.testing.allocator;
    // Exercise the routing/printing path directly with an in-memory store so the
    // test never touches the durable session. With `-Dfeat-sea=false` (default)
    // the SEA stub degrades to a plain persisted completion; the helper must
    // still report a meta line and exit 0.
    var store = features.wdbx.Store.init(allocator);
    defer store.deinit();
    const code = try handleLearnComplete(allocator, &store, "learn from this", features.ai.models.default_model);
    try std.testing.expectEqual(@as(u8, 0), code);
}

test {
    std.testing.refAllDecls(@This());
}
