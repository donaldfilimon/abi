//! ABI Streaming Request Types
//!
//! Contains request type definitions and JSON parsing helpers for the
//! streaming inference server's custom ABI endpoint.

const std = @import("std");
const backends = @import("backends/mod.zig");

/// ABI streaming request format
pub const AbiStreamRequest = struct {
    prompt: []const u8,
    backend: ?backends.BackendType,
    config: backends.GenerationConfig,
    stream_id: ?[]const u8,

    pub fn deinit(self: *const AbiStreamRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt);
        if (self.stream_id) |id| allocator.free(id);
    }
};

/// Parse ABI stream request from JSON
pub fn parseAbiStreamRequest(allocator: std.mem.Allocator, body: []const u8) !AbiStreamRequest {
    // Simple JSON parsing - in production would use a proper JSON parser
    const prompt = extractJsonString(body, "prompt") orelse return error.InvalidRequest;
    const prompt_copy = try allocator.dupe(u8, prompt);
    errdefer allocator.free(prompt_copy);

    const backend_str = extractJsonString(body, "backend");
    const backend: ?backends.BackendType = if (backend_str) |b|
        backends.BackendType.fromString(b)
    else
        null;

    const max_tokens = extractJsonInt(body, "max_tokens") orelse 1024;
    const temperature = extractJsonFloat(body, "temperature") orelse 0.7;

    const stream_id = if (extractJsonString(body, "stream_id")) |id|
        try allocator.dupe(u8, id)
    else
        null;

    return .{
        .prompt = prompt_copy,
        .backend = backend,
        .config = .{
            .max_tokens = @intCast(max_tokens),
            .temperature = @floatCast(temperature),
        },
        .stream_id = stream_id,
    };
}

// JSON helper functions

pub fn extractJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    const value_start = key_pos + search_key.len;

    // Skip whitespace
    var pos = value_start;
    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1; // Skip opening quote

    const str_start = pos;
    while (pos < json.len and json[pos] != '"') : (pos += 1) {
        if (json[pos] == '\\' and pos + 1 < json.len) pos += 1; // Skip escaped char
    }

    return json[str_start..pos];
}

pub fn extractJsonInt(json: []const u8, key: []const u8) ?i64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and (json[pos] >= '0' and json[pos] <= '9')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseInt(i64, json[num_start..pos], 10) catch null;
}

pub fn extractJsonFloat(json: []const u8, key: []const u8) ?f64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and ((json[pos] >= '0' and json[pos] <= '9') or json[pos] == '.')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseFloat(f64, json[num_start..pos]) catch null;
}
