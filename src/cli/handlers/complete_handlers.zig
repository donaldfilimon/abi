const std = @import("std");
const features = @import("../../features/mod.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");
const usage_mod = @import("../usage.zig");
const env = @import("../../foundation/env.zig");
const credentials = @import("../../foundation/credentials.zig");
const anthropic = @import("../../connectors/anthropic.zig");
const fm = @import("../../connectors/fm.zig");
const connectors = @import("../../connectors/mod.zig");

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

/// `abi complete --learn`: run one SEA self-learning pass against the durable
/// store and report a one-line meta that includes `evidence_count` (recalled
/// records) and `adapted` (whether the persona-router weights were updated).
pub fn handleLearnComplete(
    allocator: std.mem.Allocator,
    store: *features.wdbx.Store,
    input: []const u8,
    model: []const u8,
) !u8 {
    var result = try features.sea.runLearnLoop(allocator, store, input, model, .{});
    defer result.deinit(allocator);

    const completion = result.completion;
    const stats = store.stats();
    printCompletionMetadata(&completion, stats, .{ .evidence_count = result.evidence_count, .adapted = result.adapted }, false);
    return 0;
}

/// `--soul`: load a SoulLayout from JSON, bootstrap its neural network,
/// and use it to blend with keyword-based routing.
pub fn handleSoulComplete(
    io: std.Io,
    allocator: std.mem.Allocator,
    input: []const u8,
    model: []const u8,
    blend_alpha: ?f32,
    soul_path: []const u8,
) !u8 {
    _ = model;
    const alpha = blend_alpha orelse 0.5;
    const json = try std.Io.Dir.cwd().readFileAlloc(io, soul_path, allocator, .limited(64 * 1024));
    defer allocator.free(json);

    var layout = try features.ai.soul_layout.SoulLayout.fromJson(allocator, json);
    defer layout.deinit();

    var net = try features.ai.point_neural_net.PointNeuralNetwork.init(allocator, &.{ 3, 8, 3 }, 0.01);
    defer net.deinit();

    _ = try layout.bootstrap(&net);

    const response = try features.ai.routeInputWithSoul(allocator, &net, alpha, input);
    defer allocator.free(response);

    std.debug.print("{s}\n", .{response});
    return 0;
}

/// Stage 2: the live anthropic path behind `--live`. Only anthropic-provider
/// models are supported; the API key is read from the credential store and the
/// request crosses the explicit `.live` transport boundary.
pub fn handleLiveComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: []const u8, stream: bool) !u8 {
    if (features.ai.models.providerOf(model) != .anthropic) {
        return usage_mod.usageError("--live currently supports anthropic models only (e.g. --model fable-5)");
    }

    var creds = credentials.loadCredentials(allocator) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("error: no credentials found; run `abi auth signin anthropic`\n", .{});
            return 2;
        },
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

    if (stream) {
        const StreamCtx = struct {
            fn callback(_: *anyopaque, chunk: connectors.http.StreamChunk) connectors.connector.ConnectorError!void {
                if (chunk.delta.len > 0) std.debug.print("{s}", .{chunk.delta});
            }
        };
        var dummy: u8 = 0;
        const full = client.streamMessageLiveIncremental(
            io,
            allocator,
            model,
            input,
            1024,
            StreamCtx.callback,
            @ptrCast(&dummy),
        ) catch |err| {
            std.debug.print("error: anthropic live stream failed: {s}\n", .{@errorName(err)});
            return 1;
        };
        defer allocator.free(full);
        std.debug.print("\n", .{});
        std.debug.print("model={s} provider=anthropic transport=live stream=sse\n", .{model});
        return 0;
    }

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
pub fn handleFmComplete(allocator: std.mem.Allocator, input: []const u8, model: []const u8, confirmed: bool) !u8 {
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

/// Local inference bridge completion: dispatch to a user-run local
/// OpenAI-compatible server (llama-server, ollama, mlx-server) via HTTP.
pub fn handleLocalBridgeComplete(io: std.Io, allocator: std.mem.Allocator, input: []const u8, model: []const u8, stream: bool) !u8 {
    const is_mlx = std.mem.startsWith(u8, model, "mlx/") or std.mem.startsWith(u8, model, "mlx-");
    const env_key = if (is_mlx) "ABI_MLX_ENDPOINT" else "ABI_LLAMA_CPP_ENDPOINT";
    const override = env.get(env_key);
    const endpoint = connectors.local_bridge.endpointFor(model, override);
    if (!connectors.local_bridge.healthCheck(io, allocator, endpoint)) {
        std.debug.print("warning: local inference server not reachable at {s}; falling back to in-process router\n", .{endpoint});
        var session = try features.wdbx.durable_store.Session.open(io, allocator);
        defer session.deinit();
        const store = session.storePtr();
        var scheduler = scheduler_mod.Scheduler.init(allocator);
        defer scheduler.deinit();
        if (stream) {
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
                "complete:local-bridge-fallback",
                .{ .input = input, .model = model, .store_result = true },
                StreamCtx.callback,
                @ptrCast(&dummy),
            );
            defer result.deinit(allocator);
            std.debug.print("\n", .{});
            const stats = store.stats();
            printCompletionMetadata(&result, stats, null, false);
            return 0;
        }
        var result = try features.ai.completeWithScheduler(allocator, store, &scheduler, "complete:local-bridge-fallback", .{
            .input = input,
            .model = model,
            .store_result = true,
        });
        defer result.deinit(allocator);
        const stats = store.stats();
        printCompletionMetadata(&result, stats, null, false);
        return 0;
    }

    if (stream) {
        const StreamCtx = struct {
            fn callback(_: *anyopaque, chunk: connectors.http.StreamChunk) connectors.connector.ConnectorError!void {
                if (chunk.delta.len > 0) std.debug.print("{s}", .{chunk.delta});
            }
        };
        var dummy: u8 = 0;
        const full = connectors.local_bridge.completeLiveStreaming(
            io,
            allocator,
            model,
            input,
            StreamCtx.callback,
            @ptrCast(&dummy),
        ) catch |err| {
            std.debug.print("error: local bridge stream failed: {s}\n", .{@errorName(err)});
            return 1;
        };
        defer allocator.free(full);
        std.debug.print("\n", .{});
        std.debug.print("[model={s} | bridge={s} | local=true | stream=sse]\n", .{ model, endpoint });
        return 0;
    }

    var response = connectors.local_bridge.completeLive(io, allocator, model, input) catch |err| {
        std.debug.print("error: local bridge request failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer response.deinit(allocator);

    const completion_text = connectors.local_bridge.extractCompletion(allocator, response.body) catch |err| {
        std.debug.print("error: failed to parse local bridge response: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer allocator.free(completion_text);

    std.debug.print("{s}\n", .{completion_text});
    std.debug.print("[model={s} | bridge={s} | local=true]\n", .{ model, endpoint });
    return 0;
}
