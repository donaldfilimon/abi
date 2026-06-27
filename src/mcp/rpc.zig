const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const json_helpers = @import("json_helpers.zig");
const shutdown = @import("shutdown.zig");

const JsonRpcRequest = protocol.JsonRpcRequest;
const McpMethod = protocol.McpMethod;
const valueToJson = json_helpers.valueToJson;

pub fn processJsonRpc(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    const request = std.json.parseFromSlice(JsonRpcRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch return error.ParseError;
    defer request.deinit();

    if (!std.mem.eql(u8, request.value.jsonrpc, "2.0")) return error.InvalidRequest;

    const method = McpMethod.fromString(request.value.method);
    const result_json = switch (method) {
        .initialize => try handlers.handleInitializeJson(allocator, request.value.params),
        .@"tools/list" => try handlers.handleToolsListJson(allocator),
        .@"tools/call" => try handlers.handleToolsCallJson(allocator, request.value.params),
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => blk: {
            // Only signal shutdown; `main` performs teardown after joining the
            // HTTP thread (avoids freeing shared state under a peer transport's
            // in-flight call). See src/mcp/main.zig.
            shutdown.request();
            break :blk try allocator.dupe(u8, "null");
        },
        .@"resources/list" => try allocator.dupe(u8, "{\"resources\":[]}"),
        .@"prompts/list" => try allocator.dupe(u8, "{\"prompts\":[]}"),
        .unknown => return error.MethodNotFound,
    };
    defer allocator.free(result_json);

    const id_json = if (request.value.id) |id_val| try valueToJson(id_val, allocator) else try allocator.dupe(u8, "null");
    defer allocator.free(id_json);

    return try std.fmt.allocPrint(allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"result\":{s}}}", .{ id_json, result_json });
}

test "processJsonRpc handles tools list and preserves ids" {
    const allocator = std.testing.allocator;

    const numeric = try processJsonRpc(allocator, "{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"tools/list\"}");
    defer allocator.free(numeric);
    try std.testing.expect(std.mem.indexOf(u8, numeric, "\"id\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, numeric, "\"tools\"") != null);

    const string_id = try processJsonRpc(allocator, "{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"method\":\"ping\"}");
    defer allocator.free(string_id);
    try std.testing.expect(std.mem.indexOf(u8, string_id, "\"id\":\"abc\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, string_id, "\"result\":{}") != null);
}

test "processJsonRpc rejects invalid requests" {
    try std.testing.expectError(error.ParseError, processJsonRpc(std.testing.allocator, "not json"));
    try std.testing.expectError(error.InvalidRequest, processJsonRpc(std.testing.allocator, "{\"jsonrpc\":\"1.0\",\"method\":\"ping\"}"));
    try std.testing.expectError(error.MethodNotFound, processJsonRpc(std.testing.allocator, "{\"jsonrpc\":\"2.0\",\"method\":\"unknown\"}"));
}

test {
    std.testing.refAllDecls(@This());
}
