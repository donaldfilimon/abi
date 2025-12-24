const std = @import("std");

pub const CallRequest = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: u16,
    temperature: f32 = 0.2,
};

pub const CallResult = struct {
    ok: bool,
    content: []const u8,
    tokens_in: u32 = 0,
    tokens_out: u32 = 0,
    status_code: u16 = 200,
    err_msg: ?[]const u8 = null,
};

pub const OllamaConfig = struct {
    host: []const u8 = "http://127.0.0.1:11434",
    model: []const u8 = "nomic-embed-text",
};

pub const OpenAIConfig = struct {
    base_url: []const u8 = "https://api.openai.com/v1",
    api_key: []const u8,
    model: []const u8 = "text-embedding-3-small",
};

pub const ProviderConfig = union(enum) {
    ollama: OllamaConfig,
    openai: OpenAIConfig,
};

pub fn embedText(allocator: std.mem.Allocator, config: ProviderConfig, text: []const u8) ![]f32 {
    return switch (config) {
        .ollama => |cfg| ollama.embedText(allocator, cfg.host, cfg.model, text),
        .openai => |cfg| openai.embedText(allocator, cfg, text),
    };
}

// Connector implementations
pub const hf_inference = @import("hf_inference.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const mock = @import("mock.zig");
pub const ollama = @import("ollama.zig");
pub const openai = @import("openai.zig");
pub const plugin = @import("plugin.zig");

pub const Connector = struct {
    name: []const u8,
    init: *const fn (allocator: std.mem.Allocator) anyerror!void,
    call: *const fn (allocator: std.mem.Allocator, req: CallRequest) anyerror!CallResult,
    health: *const fn () bool,
};

pub fn getByName(name: []const u8) ?Connector {
    if (std.ascii.eqlIgnoreCase(name, "openai")) return openai.get();
    if (std.ascii.eqlIgnoreCase(name, "hf_inference") or std.ascii.eqlIgnoreCase(name, "hf") or std.ascii.eqlIgnoreCase(name, "huggingface")) return hf_inference.get();
    if (std.ascii.eqlIgnoreCase(name, "local_scheduler") or std.ascii.eqlIgnoreCase(name, "local")) return local_scheduler.get();
    if (std.ascii.eqlIgnoreCase(name, "ollama")) return ollama.get();
    if (std.ascii.eqlIgnoreCase(name, "mock")) return mock.get();
    return null;
}

/// Initialize the connectors feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global connector state to initialize
}

/// Deinitialize the connectors feature module
pub fn deinit() void {
    // Currently no global connector state to cleanup
}

test "JSON parsing and error mapping" {
    const testing = std.testing;

    // Test successful JSON response parsing
    const success_json =
        \\{
        \\  "choices": [
        \\    {
        \\      "message": {
        \\        "content": "Hello, world!"
        \\      }
        \\    }
        \\  ],
        \\  "usage": {
        \\    "prompt_tokens": 10,
        \\    "completion_tokens": 20
        \\  }
        \\}
    ;

    // Parse JSON
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, success_json, .{});
    defer parsed.deinit();

    // Verify structure
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.contains("choices"));
    try testing.expect(parsed.value.object.contains("usage"));

    // Test error response parsing
    const error_json =
        \\{
        \\  "error": {
        \\    "message": "Invalid API key",
        \\    "type": "authentication_error",
        \\    "code": "invalid_api_key"
        \\  }
        \\}
    ;

    var error_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, error_json, .{});
    defer error_parsed.deinit();

    // Verify error structure
    try testing.expect(error_parsed.value == .object);
    try testing.expect(error_parsed.value.object.contains("error"));

    if (error_parsed.value.object.get("error")) |error_obj| {
        try testing.expect(error_obj == .object);
        try testing.expect(error_obj.object.contains("message"));
        try testing.expect(error_obj.object.contains("type"));
    }
}

test "connector name matching with aliases" {
    const testing = std.testing;

    // Test exact matches
    try testing.expect(getByName("openai") != null);
    try testing.expect(getByName("ollama") != null);
    try testing.expect(getByName("mock") != null);

    // Test case insensitive matching
    try testing.expect(getByName("OpenAI") != null);
    try testing.expect(getByName("OLLAMA") != null);

    // Test aliases
    try testing.expect(getByName("hf") != null);
    try testing.expect(getByName("huggingface") != null);
    try testing.expect(getByName("local") != null);

    // Test non-existent connector
    try testing.expect(getByName("nonexistent") == null);
}

test "CallResult JSON serialization" {
    const testing = std.testing;

    // Test successful result serialization
    const success_result = CallResult{
        .ok = true,
        .content = "Hello, world!",
        .tokens_in = 10,
        .tokens_out = 20,
        .status_code = 200,
        .err_msg = null,
    };

    const json = try std.json.stringifyAlloc(testing.allocator, success_result, .{});
    defer testing.allocator.free(json);

    // Verify JSON contains expected fields
    try testing.expect(std.mem.indexOf(u8, json, "\"ok\":true") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"content\":\"Hello, world!\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"tokens_in\":10") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"status_code\":200") != null);

    // Test error result serialization
    const error_result = CallResult{
        .ok = false,
        .content = "",
        .status_code = 401,
        .err_msg = "Invalid API key",
    };

    const error_json = try std.json.stringifyAlloc(testing.allocator, error_result, .{});
    defer testing.allocator.free(error_json);

    try testing.expect(std.mem.indexOf(u8, error_json, "\"ok\":false") != null);
    try testing.expect(std.mem.indexOf(u8, error_json, "\"err_msg\":\"Invalid API key\"") != null);
}

test "ProviderConfig JSON parsing" {
    const testing = std.testing;

    // Test OpenAI config parsing
    const openai_config_json =
        \\{
        \\  "base_url": "https://api.openai.com/v1",
        \\  "api_key": "sk-test123",
        \\  "model": "text-embedding-3-small"
        \\}
    ;

    var openai_parsed = try std.json.parseFromSlice(OpenAIConfig, testing.allocator, openai_config_json, .{});
    defer openai_parsed.deinit();

    try testing.expectEqualStrings(openai_parsed.value.base_url, "https://api.openai.com/v1");
    try testing.expectEqualStrings(openai_parsed.value.api_key, "sk-test123");
    try testing.expectEqualStrings(openai_parsed.value.model, "text-embedding-3-small");

    // Test Ollama config parsing
    const ollama_config_json =
        \\{
        \\  "host": "http://localhost:11434",
        \\  "model": "nomic-embed-text"
        \\}
    ;

    var ollama_parsed = try std.json.parseFromSlice(OllamaConfig, testing.allocator, ollama_config_json, .{});
    defer ollama_parsed.deinit();

    try testing.expectEqualStrings(ollama_parsed.value.host, "http://localhost:11434");
    try testing.expectEqualStrings(ollama_parsed.value.model, "nomic-embed-text");
}
