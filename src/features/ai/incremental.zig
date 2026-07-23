//! Iterative in-process persona/template generation with per-step emission.
//!
//! Unlike post-hoc byte slicing of a finished string, this generator builds the
//! response by appending planned segments (prefix tokens, input word tokens,
//! suffix tokens) and invokes `stream_callback` after **each** append — emission
//! happens during generation, not after.
//!
//! Honest scope: this is true incremental decode of the local persona/template
//! model (word/token steps). It is **not** a neural LM / ggml token sampler.
//! Live Anthropic SSE and local-bridge OpenAI-compatible SSE remain the
//! network-incremental baselines.

const std = @import("std");
const types = @import("types.zig");

pub const StreamMode = enum {
    /// Word/token iterative emit during persona/template generation.
    incremental,
    /// Reserved label for callers that still post-hoc-chunk a finished buffer.
    post_hoc,
};

pub fn streamModeLabel(mode: StreamMode) []const u8 {
    return switch (mode) {
        .incremental => "incremental",
        .post_hoc => "post-hoc",
    };
}

const ProfileParts = struct {
    prefix: []const u8,
    suffix: []const u8,
};

fn partsFor(profile: types.AgentProfile) ProfileParts {
    return switch (profile) {
        .abbey => .{
            .prefix = "Abbey: ",
            .suffix = "\n\nI’ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit.",
        },
        .aviva => .{
            .prefix = "Aviva direct expert: ",
            .suffix = "\n\nLeading with the concrete answer, assumptions, and next action.",
        },
        .abi => .{
            .prefix = "ABI orchestration review: ",
            .suffix = "\n\nEvaluating intent, risk, context, and the appropriate response mode.",
        },
    };
}

fn emitAppend(
    out: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    piece: []const u8,
    on_chunk: ?types.StreamCallback,
    callback_ctx: ?*anyopaque,
) !void {
    if (piece.len == 0) return;
    const start = out.items.len;
    try out.appendSlice(allocator, piece);
    if (on_chunk) |cb| {
        const ctx = callback_ctx orelse return error.MissingStreamContext;
        try cb(ctx, .{ .delta = out.items[start..], .done = false });
    }
}

/// Emit whitespace-separated tokens of `text` one at a time (preserving spaces
/// as separate steps so the reconstructed string matches `text` exactly).
fn emitTextTokens(
    out: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    text: []const u8,
    on_chunk: ?types.StreamCallback,
    callback_ctx: ?*anyopaque,
) !void {
    var i: usize = 0;
    while (i < text.len) {
        if (std.ascii.isWhitespace(text[i])) {
            try emitAppend(out, allocator, text[i .. i + 1], on_chunk, callback_ctx);
            i += 1;
            continue;
        }
        var j = i + 1;
        while (j < text.len and !std.ascii.isWhitespace(text[j])) : (j += 1) {}
        try emitAppend(out, allocator, text[i..j], on_chunk, callback_ctx);
        i = j;
    }
}

/// Generate a persona/template response iteratively. When `on_chunk` is set,
/// each word/token segment is emitted via the callback **as it is produced**.
/// Returns an owned full string (same content as the non-streaming path).
pub fn generateProfileIncremental(
    allocator: std.mem.Allocator,
    profile: types.AgentProfile,
    input: []const u8,
    on_chunk: ?types.StreamCallback,
    callback_ctx: ?*anyopaque,
) ![]u8 {
    const parts = partsFor(profile);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try emitTextTokens(&out, allocator, parts.prefix, on_chunk, callback_ctx);
    try emitTextTokens(&out, allocator, input, on_chunk, callback_ctx);
    try emitTextTokens(&out, allocator, parts.suffix, on_chunk, callback_ctx);

    if (on_chunk) |cb| {
        const ctx = callback_ctx orelse return error.MissingStreamContext;
        try cb(ctx, .{ .delta = "", .done = true });
    }

    return try out.toOwnedSlice(allocator);
}

test "incremental generation matches one-shot persona strings" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { types.AgentProfile, []const u8, []const u8 }{
        .{ .abbey, "hello world", "Abbey: hello world\n\nI’ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit." },
        .{ .aviva, "execute", "Aviva direct expert: execute\n\nLeading with the concrete answer, assumptions, and next action." },
        .{ .abi, "route", "ABI orchestration review: route\n\nEvaluating intent, risk, context, and the appropriate response mode." },
    };
    for (cases) |c| {
        const got = try generateProfileIncremental(allocator, c[0], c[1], null, null);
        defer allocator.free(got);
        try std.testing.expectEqualStrings(c[2], got);
    }
}

test "incremental callback fires during generation with real per-step deltas" {
    const allocator = std.testing.allocator;
    const Ctx = struct {
        chunks: usize = 0,
        saw_done: bool = false,
        last_delta_len: usize = 0,
        fn cb(ctx: *anyopaque, chunk: types.StreamChunk) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk.done) {
                self.saw_done = true;
                return;
            }
            try std.testing.expect(chunk.delta.len > 0);
            self.chunks += 1;
            self.last_delta_len = chunk.delta.len;
        }
    };
    var ctx = Ctx{};
    const got = try generateProfileIncremental(allocator, .abbey, "alpha beta", Ctx.cb, &ctx);
    defer allocator.free(got);
    // prefix tokens + "alpha" + space + "beta" => multiple steps, not one post-hoc dump
    try std.testing.expect(ctx.chunks >= 4);
    try std.testing.expect(ctx.saw_done);
    try std.testing.expectEqualStrings(
        "Abbey: alpha beta\n\nI’ll approach this with warmth, creativity, and technical care while keeping uncertainty explicit.",
        got,
    );
}

test {
    std.testing.refAllDecls(@This());
}
