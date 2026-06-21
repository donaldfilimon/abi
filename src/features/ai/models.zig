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

    pub fn label(self: Provider) []const u8 {
        return switch (self) {
            .local => "local",
            .anthropic => "anthropic",
            .openai => "openai",
            .grok => "grok",
        };
    }
};

/// Default model label recorded when a caller does not request one. Kept in
/// sync with `CompletionRequest.model` and the MCP `ai_complete` default.
pub const default_model = "abi-local";

/// Anthropic Fable 5 — the model this catalog was introduced to make
/// first-class and selectable across the CLI and MCP surfaces.
pub const fable5 = "claude-fable-5";

pub const Entry = struct {
    id: []const u8,
    provider: Provider,
    /// Short, user-friendly ids that resolve to the canonical `id`.
    aliases: []const []const u8 = &.{},
};

/// Recognized model catalog. Freeform ids are still accepted by callers; this
/// list drives recognition, alias resolution, and provider routing.
pub const catalog = [_]Entry{
    .{ .id = default_model, .provider = .local },
    .{ .id = fable5, .provider = .anthropic, .aliases = &.{ "fable-5", "fable5" } },
    .{ .id = "claude-opus-4-8", .provider = .anthropic },
    .{ .id = "claude-sonnet-4-6", .provider = .anthropic },
    .{ .id = "claude-haiku-4-5", .provider = .anthropic },
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

/// Classify a model id to its serving provider. Catalog ids are authoritative;
/// unknown ids fall back to prefix heuristics so freeform ids still route
/// sensibly, and anything unrecognized stays `.local`.
pub fn providerOf(id: []const u8) Provider {
    if (resolve(id)) |canonical| {
        for (catalog) |entry| {
            if (std.mem.eql(u8, canonical, entry.id)) return entry.provider;
        }
    }
    if (std.mem.startsWith(u8, id, "claude-")) return .anthropic;
    if (std.mem.startsWith(u8, id, "gpt-")) return .openai;
    if (std.mem.startsWith(u8, id, "grok")) return .grok;
    return .local;
}

test "fable 5 is recognized and routes to anthropic" {
    try std.testing.expect(isKnown(fable5));
    try std.testing.expectEqualStrings(fable5, resolve("fable-5").?);
    try std.testing.expectEqualStrings(fable5, resolve("fable5").?);
    try std.testing.expectEqual(Provider.anthropic, providerOf(fable5));
}

test "default model stays local and is recognized" {
    try std.testing.expect(isKnown(default_model));
    try std.testing.expectEqual(Provider.local, providerOf(default_model));
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
