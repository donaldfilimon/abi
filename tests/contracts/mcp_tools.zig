const std = @import("std");
const build_options = @import("build_options");
const handlers = @import("mcp_handlers");

fn expectObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => error.ExpectedObject,
    };
}

fn expectArray(value: std.json.Value) !std.json.Array {
    return switch (value) {
        .array => |arr| arr,
        else => error.ExpectedArray,
    };
}

fn expectToolResponseContains(allocator: std.mem.Allocator, response_json: []const u8, needle: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, response_json, .{});
    defer parsed.deinit();

    const root = try expectObject(parsed.value);
    const content = try expectArray(root.get("content") orelse return error.MissingContent);
    try std.testing.expect(content.items.len > 0);

    const first = try expectObject(content.items[0]);
    const kind = switch (first.get("type") orelse return error.MissingContentType) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expectEqualStrings("text", kind);

    const text = switch (first.get("text") orelse return error.MissingText) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expect(std.mem.indexOf(u8, text, needle) != null);
}

fn expectToolJsonContains(allocator: std.mem.Allocator, json_text: []const u8, needle: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    const response = try handlers.handleToolsCallJson(allocator, parsed.value);
    defer allocator.free(response);
    try expectToolResponseContains(allocator, response, needle);
}

fn expectToolJsonContainsEither(allocator: std.mem.Allocator, json_text: []const u8, a: []const u8, b: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    const response = try handlers.handleToolsCallJson(allocator, parsed.value);
    defer allocator.free(response);

    const parsed_response = try std.json.parseFromSlice(std.json.Value, allocator, response, .{});
    defer parsed_response.deinit();
    const root = try expectObject(parsed_response.value);
    const content = try expectArray(root.get("content") orelse return error.MissingContent);
    try std.testing.expect(content.items.len > 0);
    const first = try expectObject(content.items[0]);
    const kind = switch (first.get("type") orelse return error.MissingContentType) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expectEqualStrings("text", kind);
    const text = switch (first.get("text") orelse return error.MissingText) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expect(std.mem.indexOf(u8, text, a) != null or std.mem.indexOf(u8, text, b) != null);
}

fn expectToolError(json_text: []const u8, expected_error: anyerror) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_text, .{});
    defer parsed.deinit();
    try std.testing.expectError(expected_error, handlers.handleToolsCallJson(std.testing.allocator, parsed.value));
}

test "MCP ai_run tool contract" {
    const allocator = std.testing.allocator;
    const call =
        \\{"name":"ai_run","arguments":{"input":"execute deploy quickly"}}
    ;
    if (build_options.feat_ai) {
        try expectToolJsonContains(allocator, call, "Abi action");
    } else {
        try expectToolJsonContains(allocator, call, "AI feature is disabled");
    }
}

test "MCP ai_complete tool contract" {
    const allocator = std.testing.allocator;
    const call =
        \\{"name":"ai_complete","arguments":{"input":"hello","model":"abi-local"}}
    ;
    try expectToolJsonContains(allocator, call, "model=abi-local");
    if (build_options.feat_ai and build_options.feat_wdbx) {
        try expectToolJsonContains(allocator, call, "persisted=true");
        try expectToolJsonContains(allocator, call, "kv_entries=1");
        try expectToolJsonContains(allocator, call, "vectors=2");
        try expectToolJsonContains(allocator, call, "blocks=1");
        try expectToolJsonContains(allocator, call, "metadata_key=completion:");
        try expectToolJsonContains(allocator, call, "block_id=");
    } else {
        try expectToolJsonContains(allocator, call, "persisted=false");
        try expectToolJsonContains(allocator, call, "vectors=0");
        try expectToolJsonContains(allocator, call, "blocks=0");
    }
}

test "MCP ai_complete defaults model" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(
        allocator,
        \\{"name":"ai_complete","arguments":{"input":"hello"}}
    ,
        "model=abi-local",
    );
}

test "MCP ai_complete rejects empty input" {
    try expectToolError(
        \\{"name":"ai_complete","arguments":{"input":"","model":"abi-local"}}
    , error.InvalidCompletionInput);
}

test "MCP ai_train tool contract" {
    const allocator = std.testing.allocator;
    const call =
        \\{"name":"ai_train","arguments":{"profile":"abi","dataset":"data.jsonl"}}
    ;
    if (build_options.feat_ai) {
        try expectToolJsonContains(allocator, call, "training accepted");
    } else {
        try expectToolJsonContains(allocator, call, "training disabled");
        try expectToolJsonContains(allocator, call, "AI feature is disabled");
    }
}

test "MCP wdbx_query tool contract" {
    const allocator = std.testing.allocator;
    const call =
        \\{"name":"wdbx_query","arguments":{"query":"creative explore ideas"}}
    ;
    if (build_options.feat_wdbx) {
        try expectToolJsonContains(allocator, call, "wdbx local match profile=");
    } else {
        try expectToolJsonContains(allocator, call, "wdbx local match unavailable");
        try expectToolJsonContains(allocator, call, "wdbx feature is disabled");
    }
}

test "MCP tool call errors are stable" {
    try std.testing.expectError(error.MissingParams, handlers.handleToolsCallJson(std.testing.allocator, null));
    try expectToolError("{}", error.MissingToolName);
    try expectToolError("{\"name\":\"unknown\",\"arguments\":{}}", error.UnknownTool);
    try expectToolError("{\"name\":\"ai_run\"}", error.MissingArguments);
    try expectToolError("{\"name\":\"ai_run\",\"arguments\":{}}", error.MissingInput);
    try expectToolError("{\"name\":\"ai_train\",\"arguments\":{\"dataset\":\"data.jsonl\"}}", error.MissingProfile);
    try expectToolError("{\"name\":\"ai_train\",\"arguments\":{\"profile\":\"abi\"}}", error.MissingDataset);
    try expectToolError("{\"name\":\"wdbx_query\",\"arguments\":{}}", error.MissingQuery);
}
