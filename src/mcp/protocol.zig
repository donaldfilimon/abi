const std = @import("std");

pub const MAX_REQUEST_SIZE = 64 * 1024; // 64KB request limit

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

pub fn validateRequest(line: []const u8) !void {
    if (line.len == 0) return error.EmptyRequest;
    if (line.len > MAX_REQUEST_SIZE) return error.RequestTooLarge;
    var i: usize = 0;
    while (i < line.len and (line[i] == ' ' or line[i] == '\t')) : (i += 1) {}
    if (i >= line.len or line[i] != '{') return error.InvalidJsonFormat;
}
