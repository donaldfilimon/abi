const std = @import("std");
const foundation_http = @import("abi").foundation.http;

pub const MAX_REQUEST_SIZE = foundation_http.MAX_REQUEST_SIZE;
/// Maximum nesting depth of `{`/`[` containers accepted by JSON-RPC parse paths.
/// Bounds CPU/stack pressure from adversarial deeply-nested payloads (TM-008).
pub const MAX_JSON_DEPTH: usize = 32;

pub const JsonRpcRequest = struct {
    jsonrpc: []const u8,
    method: []const u8,
    id: ?std.json.Value = null,
    params: ?std.json.Value = null,
};

pub const JsonRpcErrorObj = struct {
    code: i32,
    message: []const u8,
    data: ?std.json.Value = null,
};

pub const JsonRpcResponse = struct {
    jsonrpc: []const u8 = "2.0",
    id: ?std.json.Value,
    result: ?std.json.Value = null,
    @"error": ?JsonRpcErrorObj = null,
};

pub const McpMethod = enum {
    initialize,
    @"tools/list",
    @"tools/call",
    @"resources/list",
    @"prompts/list",
    ping,
    shutdown,
    unknown,

    pub fn fromString(s: []const u8) McpMethod {
        if (std.mem.eql(u8, s, "initialize")) return .initialize;
        if (std.mem.eql(u8, s, "tools/list")) return .@"tools/list";
        if (std.mem.eql(u8, s, "tools/call")) return .@"tools/call";
        if (std.mem.eql(u8, s, "resources/list")) return .@"resources/list";
        if (std.mem.eql(u8, s, "prompts/list")) return .@"prompts/list";
        if (std.mem.eql(u8, s, "ping")) return .ping;
        if (std.mem.eql(u8, s, "shutdown")) return .shutdown;
        return .unknown;
    }
};

/// Structural pre-check for JSON-RPC request lines (stdio + HTTP body).
/// Rejects empty/oversize input, non-object roots, and over-nested containers
/// before `std.json.parseFromSlice` so both transports share one bound.
pub fn validateRequest(line: []const u8) !void {
    if (line.len == 0) return error.EmptyRequest;
    if (line.len > MAX_REQUEST_SIZE) return error.RequestTooLarge;
    var i: usize = 0;
    while (i < line.len and (line[i] == ' ' or line[i] == '\t')) : (i += 1) {}
    if (i >= line.len or line[i] != '{') return error.InvalidJsonFormat;
    try checkJsonDepth(line, MAX_JSON_DEPTH);
}

/// Walk JSON text and reject when object/array nesting exceeds `max_depth`.
/// String contents are skipped so braces inside quoted values do not count.
pub fn checkJsonDepth(src: []const u8, max_depth: usize) !void {
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (src) |byte| {
        if (in_string) {
            if (escape) {
                escape = false;
                continue;
            }
            if (byte == '\\') {
                escape = true;
                continue;
            }
            if (byte == '"') in_string = false;
            continue;
        }
        switch (byte) {
            '"' => in_string = true,
            '{', '[' => {
                depth += 1;
                if (depth > max_depth) return error.JsonTooDeep;
            },
            '}', ']' => {
                if (depth == 0) return error.InvalidJsonFormat;
                depth -= 1;
            },
            else => {},
        }
    }
}

test "protocol: McpMethod.fromString maps known and unknown methods" {
    try std.testing.expectEqual(McpMethod.initialize, McpMethod.fromString("initialize"));
    try std.testing.expectEqual(McpMethod.@"tools/call", McpMethod.fromString("tools/call"));
    try std.testing.expectEqual(McpMethod.@"tools/list", McpMethod.fromString("tools/list"));
    try std.testing.expectEqual(McpMethod.ping, McpMethod.fromString("ping"));
    // Unrecognized and empty method names fall through to `.unknown`.
    try std.testing.expectEqual(McpMethod.unknown, McpMethod.fromString("tools/run"));
    try std.testing.expectEqual(McpMethod.unknown, McpMethod.fromString(""));
}

test "protocol: validateRequest accepts an object line and rejects malformed input" {
    // Happy path: a JSON object, optionally with leading whitespace.
    try validateRequest("{\"jsonrpc\":\"2.0\"}");
    try validateRequest("  \t{\"a\":1}");

    // Edge/malformed cases each map to a distinct, stable error.
    try std.testing.expectError(error.EmptyRequest, validateRequest(""));
    try std.testing.expectError(error.InvalidJsonFormat, validateRequest("not json"));
    try std.testing.expectError(error.InvalidJsonFormat, validateRequest("   "));

    var oversized: [MAX_REQUEST_SIZE + 1]u8 = undefined;
    @memset(&oversized, '{');
    try std.testing.expectError(error.RequestTooLarge, validateRequest(&oversized));
}

test "protocol: validateRequest rejects over-nested JSON containers" {
    // Depth-1 object is fine; build a chain of nested objects past MAX_JSON_DEPTH.
    const allocator = std.testing.allocator;
    var deep: std.ArrayListUnmanaged(u8) = .empty;
    defer deep.deinit(allocator);
    var i: usize = 0;
    while (i < MAX_JSON_DEPTH + 1) : (i += 1) {
        try deep.appendSlice(allocator, "{\"a\":");
    }
    try deep.appendSlice(allocator, "1");
    i = 0;
    while (i < MAX_JSON_DEPTH + 1) : (i += 1) {
        try deep.append(allocator, '}');
    }
    try std.testing.expectError(error.JsonTooDeep, validateRequest(deep.items));

    // Braces inside strings must not count toward depth.
    try validateRequest("{\"a\":\"{{{{{\"}");
}

test "protocol: checkJsonDepth allows shallow nests and rejects deeper ones" {
    try checkJsonDepth("{\"a\":[1,2,{\"b\":3}]}", 4);
    try std.testing.expectError(error.JsonTooDeep, checkJsonDepth("[[[[[]]]]]", 3));
}

test {
    std.testing.refAllDecls(@This());
}
