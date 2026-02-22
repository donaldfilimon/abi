const std = @import("std");
const types = @import("types.zig");

pub fn parseProviderId(value: []const u8) ?types.ProviderId {
    if (types.ProviderId.fromString(value)) |provider| return provider;

    if (std.mem.eql(u8, value, "llama-cpp")) return .llama_cpp;
    if (std.mem.eql(u8, value, "lm-studio")) return .lm_studio;
    if (std.mem.eql(u8, value, "plugin-http")) return .plugin_http;
    if (std.mem.eql(u8, value, "plugin-native")) return .plugin_native;
    if (std.mem.eql(u8, value, "local-gguf")) return .local_gguf;
    if (std.mem.eql(u8, value, "ollama-passthrough")) return .ollama_passthrough;

    return null;
}

test "parseProviderId supports canonical and alias names" {
    try std.testing.expectEqual(types.ProviderId.llama_cpp, parseProviderId("llama-cpp").?);
    try std.testing.expectEqual(types.ProviderId.lm_studio, parseProviderId("lm-studio").?);
    try std.testing.expectEqual(types.ProviderId.plugin_http, parseProviderId("plugin-http").?);
    try std.testing.expectEqual(types.ProviderId.plugin_native, parseProviderId("plugin-native").?);
    try std.testing.expectEqual(types.ProviderId.local_gguf, parseProviderId("local-gguf").?);
    try std.testing.expectEqual(types.ProviderId.ollama_passthrough, parseProviderId("ollama-passthrough").?);
    try std.testing.expectEqual(types.ProviderId.codex, parseProviderId("codex").?);
    try std.testing.expectEqual(types.ProviderId.opencode, parseProviderId("opencode").?);
    try std.testing.expectEqual(types.ProviderId.claude, parseProviderId("claude").?);
    try std.testing.expectEqual(types.ProviderId.gemini, parseProviderId("gemini").?);
    try std.testing.expect(parseProviderId("does-not-exist") == null);
}

// Pull in tests from this file.
test {
    std.testing.refAllDecls(@This());
}
