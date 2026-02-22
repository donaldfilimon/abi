const std = @import("std");
const connectors = @import("../../../../../services/connectors/mod.zig");
const types = @import("../types.zig");
const errors = @import("../errors.zig");
const manifest = @import("manifest.zig");

pub fn generate(
    allocator: std.mem.Allocator,
    plugin_entry: manifest.PluginEntry,
    cfg: types.GenerateConfig,
) !types.GenerateResult {
    if (plugin_entry.kind != .http) return errors.ProviderError.InvalidPlugin;
    if (!plugin_entry.enabled) return errors.ProviderError.PluginDisabled;

    const host = plugin_entry.base_url orelse return errors.ProviderError.InvalidPlugin;

    var api_key: ?[]u8 = null;
    if (plugin_entry.api_key_env) |env_name| {
        api_key = try connectors.getEnvOwned(allocator, env_name);
    }
    errdefer if (api_key) |key| allocator.free(key);

    const model = if (plugin_entry.model) |value| value else cfg.model;

    const config = connectors.vllm.Config{
        .host = try allocator.dupe(u8, host),
        .api_key = api_key,
        .model = if (plugin_entry.model != null) try allocator.dupe(u8, model) else model,
        .model_owned = plugin_entry.model != null,
        .timeout_ms = 120_000,
    };

    var client = connectors.vllm.Client.init(allocator, config) catch |err| {
        if (api_key) |key| allocator.free(key);
        return err;
    };
    defer client.deinit();

    const messages = [_]connectors.vllm.Message{
        .{ .role = "user", .content = cfg.prompt },
    };

    var response = try client.chatCompletion(.{
        .model = model,
        .messages = &messages,
        .temperature = cfg.temperature,
        .max_tokens = cfg.max_tokens,
        .top_p = cfg.top_p,
        .stream = false,
    });
    defer deinitResponse(allocator, &response);

    if (response.choices.len == 0) return errors.ProviderError.GenerationFailed;

    return .{
        .provider = .plugin_http,
        .model_used = try allocator.dupe(u8, response.model),
        .content = try allocator.dupe(u8, response.choices[0].message.content),
    };
}

fn deinitResponse(allocator: std.mem.Allocator, response: *connectors.vllm.ChatCompletionResponse) void {
    allocator.free(response.id);
    allocator.free(response.model);
    for (response.choices) |*choice| {
        allocator.free(choice.message.role);
        allocator.free(choice.message.content);
        allocator.free(choice.finish_reason);
    }
    allocator.free(response.choices);
    response.* = undefined;
}

// ============================================================================
// Tests
// ============================================================================

test "generate rejects non-http plugin kind" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("native-plugin"),
        .kind = .native,
        .enabled = true,
        .base_url = @constCast("http://localhost:8080"),
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.InvalidPlugin, result);
}

test "generate rejects disabled plugin" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("disabled-plugin"),
        .kind = .http,
        .enabled = false,
        .base_url = @constCast("http://localhost:8080"),
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.PluginDisabled, result);
}

test "generate rejects plugin without base_url" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("no-url"),
        .kind = .http,
        .enabled = true,
        .base_url = null,
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.InvalidPlugin, result);
}

test {
    std.testing.refAllDecls(@This());
}
