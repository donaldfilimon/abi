//! Prompt-completion path for the interactive REPL.
//!
//! Extracted from `repl.zig` so the dispatch hub stays focused on
//! slash-command routing. This leaf owns `completePrompt` — the ordinary-input
//! path that resolves `@file` mentions, builds the multi-turn context prefix,
//! and streams a completion via the SEA learn loop, the local-bridge SSE
//! transport, live Anthropic SSE (`/live`), or the in-process persona router —
//! plus the stream-callback contexts those paths share. Free functions take the
//! needed pieces (allocator, store, scheduler, state, io) as parameters.

const std = @import("std");
const build_options = @import("build_options");
const env = @import("../../foundation/env.zig");
const credentials = @import("../../foundation/credentials.zig");
const ai = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const sea = if (build_options.feat_sea) @import("../sea/mod.zig") else @import("../sea/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const sanitize = @import("sanitize.zig");
const file_context = @import("../ai/file_context.zig");
const cmds = @import("repl_commands.zig");
const local_bridge = @import("../../connectors/local_bridge.zig");
const connector_http = @import("../../connectors/http.zig");
const connector_mod = @import("../../connectors/connector.zig");
const anthropic = @import("../../connectors/anthropic.zig");
const repl_types = @import("repl_types.zig");

const LineOutcome = repl_types.LineOutcome;

/// Best-effort flush so stream deltas appear as they arrive on a tty.
/// Non-tty degrades silently (keeps OS buffering); sync errors are ignored.
fn flushStreamPaint() void {
    if (comptime @hasDecl(std.posix.system, "isatty")) {
        if (std.posix.system.isatty(std.posix.STDERR_FILENO) == 0) return;
    }
    std.Io.File.stderr().sync(std.Options.debug_io) catch |err| std.log.warn("flushStreamPaint stderr sync: {s}", .{@errorName(err)});
}

const StreamCtx = struct {
    allocator: std.mem.Allocator,
    fn callback(ctx: *anyopaque, chunk: ai.StreamChunk) anyerror!void {
        if (chunk.delta.len == 0) return;
        const self: *@This() = @ptrCast(@alignCast(ctx));
        const safe = try sanitize.sanitizeControlBytes(self.allocator, chunk.delta);
        defer self.allocator.free(safe);
        std.debug.print("{s}", .{safe});
        flushStreamPaint();
    }
    /// Connector SSE path uses ConnectorError; print-only and never abort mid-stream.
    fn bridgeCallback(ctx: *anyopaque, chunk: connector_http.StreamChunk) connector_mod.ConnectorError!void {
        if (chunk.delta.len == 0) return;
        const self: *@This() = @ptrCast(@alignCast(ctx));
        const safe = sanitize.sanitizeControlBytes(self.allocator, chunk.delta) catch return;
        defer self.allocator.free(safe);
        std.debug.print("{s}", .{safe});
        flushStreamPaint();
    }
};

/// True when `/live` is on and the active model is an Anthropic catalog id.
pub fn shouldUseLiveAnthropic(state: *const repl_types.ReplState) bool {
    return state.config.live_mode and ai.models.providerOf(state.config.model) == .anthropic;
}

/// Run one ordinary (non-slash) input line through the completion path and
/// persist the turn into `state`.
pub fn completePrompt(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    scheduler: *scheduler_mod.Scheduler,
    state: *repl_types.ReplState,
    line: []const u8,
    io: std.Io,
) !LineOutcome {
    // Resolve @file mentions in the input before building context
    const resolved = try resolveFileMentions(allocator, line, ".", io);
    defer allocator.free(resolved);

    var augmented_line = resolved;
    var augmented_buf: ?[]u8 = null;
    defer if (augmented_buf) |b| allocator.free(b);

    const maybe_prefix = try cmds.buildCompletionContext(
        allocator,
        state.config.context_snippets,
        &state.turn_history,
        state.turn_history_count,
        state.turn_history_head,
        state.open_path,
        state.open_content,
        resolved,
    );
    if (maybe_prefix) |prefix| {
        augmented_buf = prefix;
        augmented_line = prefix;
    }

    if (state.config.learn_mode and build_options.feat_sea) {
        // SEA self-learning path: evidence-augmented completion
        var stream_ctx = StreamCtx{ .allocator = allocator };
        var result = try sea.runLearnLoop(allocator, store, augmented_line, state.config.model, .{
            .persist = state.config.store_turns,
            .adapt_router = true,
            .stream_callback = StreamCtx.callback,
            .stream_ctx = &stream_ctx,
        });
        defer result.deinit(allocator);

        state.recordConstitution(result.completion.audit.escore, result.completion.audit.passed, result.completion.audit.vetoed);

        // All chunks emitted; print turn metadata (post-hoc chunking is honest).
        std.debug.print("\n", .{});
        std.debug.print("[turn {d} | model={s} | profile={s} | sea | evidence={d} | adapted={s} | stream=incremental | constitution=escore={d:.2} passed={s}]\n", .{
            state.turn_count + 1,
            result.completion.model,
            result.completion.selected_profile.label(),
            result.evidence_count,
            if (result.adapted) "true" else "false",
            result.completion.audit.escore,
            if (result.completion.audit.passed) "true" else "false",
        });

        state.pushTurn(allocator, line, result.completion.output);
    } else if (shouldUseLiveAnthropic(state)) {
        try completeLiveAnthropic(allocator, state, line, augmented_line, io);
    } else if (local_bridge.isLocalBridgeModel(state.config.model)) {
        // Local OpenAI-compatible server (llama-server / ollama / mlx-server).
        // Prefer SSE token streaming when the server is reachable; fall back
        // to the in-process persona router with post-hoc chunked output.
        const model = state.config.model;
        const is_mlx = std.mem.startsWith(u8, model, "mlx/") or std.mem.startsWith(u8, model, "mlx-");
        const env_key = if (is_mlx) "ABI_MLX_ENDPOINT" else "ABI_LLAMA_CPP_ENDPOINT";
        const override = env.get(env_key);
        const endpoint = local_bridge.endpointFor(model, override);

        if (local_bridge.healthCheck(io, allocator, endpoint)) {
            var stream_ctx = StreamCtx{ .allocator = allocator };
            const full = local_bridge.completeLiveStreaming(
                io,
                allocator,
                model,
                augmented_line,
                StreamCtx.bridgeCallback,
                &stream_ctx,
            ) catch |err| {
                std.debug.print("\n[bridge stream error: {s}; falling back to in-process (stream=incremental)]\n", .{@errorName(err)});
                var stream_ctx_fb = StreamCtx{ .allocator = allocator };
                var result = try ai.completeWithSchedulerStreaming(allocator, store, scheduler, "complete:agent-tui-bridge-fb", .{
                    .input = augmented_line,
                    .model = model,
                    .store_result = state.config.store_turns,
                }, StreamCtx.callback, &stream_ctx_fb);
                defer result.deinit(allocator);
                state.recordConstitution(result.audit.escore, result.audit.passed, result.audit.vetoed);
                std.debug.print("\n", .{});
                std.debug.print("[turn {d} | model={s} | profile={s} | bridge=fallback | stream=incremental | constitution=escore={d:.2} | persisted={s}]\n", .{
                    state.turn_count + 1,
                    result.model,
                    result.selected_profile.label(),
                    result.audit.escore,
                    if (result.query_vector_id != null) "true" else "false",
                });
                state.pushTurn(allocator, line, result.output);
                state.turn_count += 1;
                return .keep_going;
            };
            defer allocator.free(full);
            std.debug.print("\n", .{});
            std.debug.print("[turn {d} | model={s} | bridge={s} | stream=sse]\n", .{
                state.turn_count + 1,
                model,
                endpoint,
            });
            state.pushTurn(allocator, line, full);
        } else {
            std.debug.print("warning: local inference server not reachable at {s}; falling back to in-process (stream=incremental)\n", .{endpoint});
            var stream_ctx = StreamCtx{ .allocator = allocator };
            var result = try ai.completeWithSchedulerStreaming(allocator, store, scheduler, "complete:agent-tui-bridge-down", .{
                .input = augmented_line,
                .model = model,
                .store_result = state.config.store_turns,
            }, StreamCtx.callback, &stream_ctx);
            defer result.deinit(allocator);
            state.recordConstitution(result.audit.escore, result.audit.passed, result.audit.vetoed);
            std.debug.print("\n", .{});
            std.debug.print("[turn {d} | model={s} | profile={s} | bridge=unreachable | stream=incremental | constitution=escore={d:.2} | persisted={s}]\n", .{
                state.turn_count + 1,
                result.model,
                result.selected_profile.label(),
                result.audit.escore,
                if (result.query_vector_id != null) "true" else "false",
            });
            state.pushTurn(allocator, line, result.output);
        }
    } else {
        var stream_ctx = StreamCtx{ .allocator = allocator };
        var result = try ai.completeWithSchedulerStreaming(allocator, store, scheduler, "complete:agent-tui", .{
            .input = augmented_line,
            .model = state.config.model,
            .store_result = state.config.store_turns,
        }, StreamCtx.callback, &stream_ctx);
        defer result.deinit(allocator);

        state.recordConstitution(result.audit.escore, result.audit.passed, result.audit.vetoed);

        // All chunks emitted; print turn metadata (post-hoc = template split, not token gen).
        std.debug.print("\n", .{});
        std.debug.print("[turn {d} | model={s} | profile={s} | stream=incremental | constitution=escore={d:.2} passed={s} | persisted={s}]\n", .{
            state.turn_count + 1,
            result.model,
            result.selected_profile.label(),
            result.audit.escore,
            if (result.audit.passed) "true" else "false",
            if (result.query_vector_id != null) "true" else "false",
        });

        state.pushTurn(allocator, line, result.output);
    }
    state.turn_count += 1;
    return .keep_going;
}

fn completeLiveAnthropic(
    allocator: std.mem.Allocator,
    state: *repl_types.ReplState,
    line: []const u8,
    augmented_line: []const u8,
    io: std.Io,
) !void {
    var creds = credentials.loadCredentials(allocator) catch |err| {
        std.debug.print("error: live anthropic requires credentials ({s}); run `abi auth signin anthropic` or `/live` off\n", .{@errorName(err)});
        return;
    };
    defer creds.deinit(allocator);

    const api_key = creds.anthropic_api_key orelse {
        std.debug.print("error: no anthropic credentials; run `abi auth signin anthropic` or `/live` off\n", .{});
        return;
    };

    var client = anthropic.Client.init(allocator, .{
        .api_key = api_key,
        .base_url = "https://api.anthropic.com",
        .transport = .live,
    });
    defer client.deinit();

    var stream_ctx = StreamCtx{ .allocator = allocator };
    const full = client.streamMessageLiveIncremental(
        io,
        allocator,
        state.config.model,
        augmented_line,
        1024,
        StreamCtx.bridgeCallback,
        &stream_ctx,
    ) catch |err| {
        std.debug.print("\n[live anthropic stream error: {s}]\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(full);

    std.debug.print("\n", .{});
    std.debug.print("[turn {d} | model={s} | provider=anthropic | transport=live | stream=sse]\n", .{
        state.turn_count + 1,
        state.config.model,
    });
    state.pushTurn(allocator, line, full);
}

/// Resolve `@file` mentions in input text by reading file contents.
/// Each `@path` mention is replaced with `[file: path]\ncontent\n[/file]`.
/// If the file cannot be read, the mention is replaced with
/// `[file: path] (not found)[/file]`.
/// Returns an owned string; caller must free.
fn resolveFileMentions(allocator: std.mem.Allocator, input: []const u8, root: []const u8, io: std.Io) ![]u8 {
    const max_file_read: usize = 16384;

    const mentions = file_context.parseFileMentions(allocator, input) catch {
        return try allocator.dupe(u8, input);
    };
    defer allocator.free(mentions);

    if (mentions.len == 0) {
        return try allocator.dupe(u8, input);
    }

    var result = std.ArrayListUnmanaged(u8).empty;
    defer result.deinit(allocator);

    var last_end: usize = 0;
    for (mentions) |mention| {
        // Append text before the mention
        try result.appendSlice(allocator, input[last_end..mention.start]);

        // Try to read the file (bounded to 16 KB)
        const file_contents = file_context.readFileBounded(io, allocator, root, mention.path, max_file_read) catch {
            // File not found or unreadable — emit placeholder
            try result.appendSlice(allocator, "[file: ");
            try result.appendSlice(allocator, mention.path);
            try result.appendSlice(allocator, "] (not found)[/file]");
            last_end = mention.end;
            continue;
        };
        defer allocator.free(file_contents);

        // Inject file contents with markers
        try result.appendSlice(allocator, "[file: ");
        try result.appendSlice(allocator, mention.path);
        try result.appendSlice(allocator, "]\n");
        try result.appendSlice(allocator, file_contents);
        try result.appendSlice(allocator, "\n[/file]");

        last_end = mention.end;
    }
    // Append remaining text after the last mention
    try result.appendSlice(allocator, input[last_end..]);

    return try result.toOwnedSlice(allocator);
}

test "resolveFileMentions passes through input with no mentions" {
    const allocator = std.testing.allocator;
    const result = try resolveFileMentions(allocator, "hello world", ".", std.testing.io);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "resolveFileMentions replaces unfound file with placeholder" {
    const allocator = std.testing.allocator;
    const result = try resolveFileMentions(allocator, "read @nonexistent-file.xyz end", ".", std.testing.io);
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "(not found)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[file: nonexistent-file.xyz]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[/file]") != null);
    try std.testing.expect(std.mem.startsWith(u8, result, "read "));
    try std.testing.expect(std.mem.endsWith(u8, result, " end"));
}

test "resolveFileMentions preserves text before and after the mention" {
    // Tests that surrounding text is preserved when a file mention resolves to
    // the (not found) placeholder. Covers the position tracking logic.
    const allocator = std.testing.allocator;
    const result = try resolveFileMentions(allocator, "before @missing.txt after", ".", std.testing.io);
    defer allocator.free(result);
    try std.testing.expect(std.mem.startsWith(u8, result, "before "));
    try std.testing.expect(std.mem.endsWith(u8, result, " after"));
    try std.testing.expect(std.mem.indexOf(u8, result, "(not found)") != null);
}

test "resolveFileMentions ignores email-like @ tokens" {
    const allocator = std.testing.allocator;
    const result = try resolveFileMentions(allocator, "contact user@example.com for details", ".", std.testing.io);
    defer allocator.free(result);
    // No @file resolution; text passes through unchanged
    try std.testing.expectEqualStrings("contact user@example.com for details", result);
}

test "resolveFileMentions handles multiple mentions with mix of found and missing" {
    const allocator = std.testing.allocator;
    const result = try resolveFileMentions(allocator, "process @missing-one.xyz and @missing-two.abc", ".", std.testing.io);
    defer allocator.free(result);
    // Both should produce (not found) placeholders
    try std.testing.expect(std.mem.indexOf(u8, result, "(not found)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "missing-one.xyz") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "missing-two.abc") != null);
    try std.testing.expect(std.mem.startsWith(u8, result, "process "));
}

test "shouldUseLiveAnthropic requires live_mode and anthropic provider" {
    var state = repl_types.ReplState.init(.{});
    try std.testing.expect(!shouldUseLiveAnthropic(&state));
    state.config.live_mode = true;
    // default model is anthropic (claude-fable-5)
    try std.testing.expect(shouldUseLiveAnthropic(&state));
    state.config.model = "abi-local";
    try std.testing.expect(!shouldUseLiveAnthropic(&state));
}

test {
    std.testing.refAllDecls(@This());
}
