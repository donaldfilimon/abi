const std = @import("std");

/// Provider that ultimately serves a given model id. The local persona router
/// is the default; the remote providers map to the matching connector under
/// `src/connectors` and are only reachable across the explicit live-transport
/// boundary (see `connectors/connector.zig` `TransportMode`).
///
/// This module is intentionally dependency-free (std only) so both the real AI
/// `mod.zig` and the disabled `stub.zig` can export it and keep declaration
/// parity — the catalog is plain data, independent of `-Dfeat-ai`.
pub const Provider = enum {
    local,
    anthropic,
    openai,
    grok,
    /// Apple FoundationModels on-device runtime (see `connectors/fm.zig`).
    fm,

    pub fn label(self: Provider) []const u8 {
        return switch (self) {
            .local => "local",
            .anthropic => "anthropic",
            .openai => "openai",
            .grok => "grok",
            .fm => "fm",
        };
    }
};

/// Anthropic Fable 5 — the model this catalog was introduced to make
/// first-class and selectable across the CLI and MCP surfaces, and now the
/// default recorded when a caller does not request a model.
pub const fable5 = "claude-fable-5";

/// Default model label recorded when a caller does not request one. Kept in
/// sync with `CompletionRequest.model` and the MCP `ai_complete` default.
/// `abi-local` remains a selectable local model (see the catalog below); the
/// default now points at first-class Claude Fable 5.
pub const default_model = fable5;

pub const Entry = struct {
    id: []const u8,
    provider: Provider,
    /// Short, user-friendly ids that resolve to the canonical `id`.
    aliases: []const []const u8 = &.{},
};

/// Recognized model catalog. Freeform ids are still accepted by callers; this
/// list drives recognition, alias resolution, and provider routing.
pub const catalog = [_]Entry{
    .{ .id = "abi-local", .provider = .local },
    .{ .id = fable5, .provider = .anthropic, .aliases = &.{ "fable-5", "fable5" } },
    .{ .id = "claude-opus-4-8", .provider = .anthropic },
    .{ .id = "claude-sonnet-4-6", .provider = .anthropic },
    .{ .id = "claude-haiku-4-5", .provider = .anthropic },
    .{ .id = "apple-fm", .provider = .fm, .aliases = &.{ "fm-local", "fm" } },
};

/// Resolve a user-supplied id (including short aliases) to its canonical id,
/// or null if the id is not in the catalog.
pub fn resolve(id: []const u8) ?[]const u8 {
    for (catalog) |entry| {
        if (std.mem.eql(u8, id, entry.id)) return entry.id;
        for (entry.aliases) |alias| {
            if (std.mem.eql(u8, id, alias)) return entry.id;
        }
    }
    return null;
}

pub fn isKnown(id: []const u8) bool {
    return resolve(id) != null;
}

/// Canonicalize a user-supplied id at the CLI/MCP edge: resolve a known alias
/// or id to its canonical catalog id, or pass a freeform id through unchanged.
/// This is the entry point that makes the catalog reachable — `--model fable-5`
/// and `--model claude-fable-5` both record the canonical `claude-fable-5`.
pub fn canonical(id: []const u8) []const u8 {
    return resolve(id) orelse id;
}

/// Classify a model id to its serving provider. Catalog ids are authoritative;
/// unknown ids fall back to prefix heuristics so freeform ids still route
/// sensibly, and anything unrecognized stays `.local`.
pub fn providerOf(id: []const u8) Provider {
    if (resolve(id)) |cid| {
        for (catalog) |entry| {
            if (std.mem.eql(u8, cid, entry.id)) return entry.provider;
        }
    }
    if (std.mem.startsWith(u8, id, "claude-")) return .anthropic;
    if (std.mem.startsWith(u8, id, "gpt-")) return .openai;
    if (std.mem.startsWith(u8, id, "grok")) return .grok;
    if (std.mem.startsWith(u8, id, "apple-")) return .fm;
    return .local;
}

test "fable 5 is recognized and routes to anthropic" {
    try std.testing.expect(isKnown(fable5));
    try std.testing.expectEqualStrings(fable5, resolve("fable-5").?);
    try std.testing.expectEqualStrings(fable5, resolve("fable5").?);
    try std.testing.expectEqual(Provider.anthropic, providerOf(fable5));
}

test "canonical resolves aliases and passes freeform ids through" {
    try std.testing.expectEqualStrings(fable5, canonical("fable-5"));
    try std.testing.expectEqualStrings(fable5, canonical(fable5));
    try std.testing.expectEqualStrings(default_model, canonical(default_model));
    try std.testing.expectEqualStrings("gpt-5-mystery", canonical("gpt-5-mystery"));
}

test "apple-fm is recognized and routes to the on-device fm provider" {
    try std.testing.expect(isKnown("apple-fm"));
    try std.testing.expectEqualStrings("apple-fm", resolve("fm").?);
    try std.testing.expectEqualStrings("apple-fm", resolve("fm-local").?);
    try std.testing.expectEqualStrings("apple-fm", canonical("fm"));
    try std.testing.expectEqual(Provider.fm, providerOf("apple-fm"));
    try std.testing.expectEqual(Provider.fm, providerOf("apple-future-model"));
    try std.testing.expectEqualStrings("fm", Provider.fm.label());
}

test "default model stays local and is recognized" {
    try std.testing.expect(isKnown(default_model));
    try std.testing.expectEqual(Provider.anthropic, providerOf(default_model));
}

test "unknown ids are not known but prefixes still classify" {
    try std.testing.expect(!isKnown("mystery"));
    try std.testing.expect(resolve("mystery") == null);
    try std.testing.expectEqual(Provider.local, providerOf("mystery"));
    try std.testing.expectEqual(Provider.anthropic, providerOf("claude-future-9"));
    try std.testing.expectEqual(Provider.openai, providerOf("gpt-5"));
    try std.testing.expectEqual(Provider.grok, providerOf("grok-3"));
}

test {
    std.testing.refAllDecls(@This());
}
