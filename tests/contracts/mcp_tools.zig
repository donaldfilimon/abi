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
        try expectToolJsonContains(allocator, call, "ranking=hybrid");
        try expectToolJsonContains(allocator, call, "persona=");
    } else {
        try expectToolJsonContains(allocator, call, "wdbx local match unavailable");
        try expectToolJsonContains(allocator, call, "wdbx feature is disabled");
    }
}

test "MCP scheduler and status tool contracts" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(allocator, "{\"name\":\"scheduler_stats\",\"arguments\":{}}", "source=mcp-server");
    try expectToolJsonContains(allocator, "{\"name\":\"scheduler_info\",\"arguments\":{}}", "source=mcp-server");
    try expectToolJsonContains(allocator, "{\"name\":\"gpu_status\",\"arguments\":{}}", "backend=");
    try expectToolJsonContains(allocator, "{\"name\":\"wdbx_stats\",\"arguments\":{}}", "kv=");
}

test "MCP connector_test tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(allocator, "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"openai\",\"input\":\"hello\"}}", "connector=openai");
    try expectToolJsonContains(allocator, "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"anthropic\",\"input\":\"hello\"}}", "connector=anthropic");
    try expectToolJsonContains(allocator, "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"discord\",\"input\":\"hello\"}}", "connector=discord");
    try expectToolJsonContains(allocator, "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"twilio\",\"input\":\"hello\"}}", "connector=twilio");
    try expectToolJsonContains(allocator, "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"grok\",\"input\":\"hello\"}}", "connector=grok");
}

test "MCP plugin_list tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(allocator, "{\"name\":\"plugin_list\",\"arguments\":{}}", "plugins count=2");
    try expectToolJsonContains(allocator, "{\"name\":\"plugin_list\",\"arguments\":{}}", "example-plugin");
    try expectToolJsonContains(allocator, "{\"name\":\"plugin_list\",\"arguments\":{}}", "example-wdbx-plugin");
}

test "MCP plugin_run tool contract" {
    const allocator = std.testing.allocator;
    // Happy path: the bundled example-plugin echoes the input length.
    try expectToolJsonContains(
        allocator,
        \\{"name":"plugin_run","arguments":{"name":"example-plugin","input":"hello"}}
    ,
        "example-plugin received input (len=5)",
    );
    // `input` is optional and defaults to an empty string.
    try expectToolJsonContains(
        allocator,
        \\{"name":"plugin_run","arguments":{"name":"example-plugin"}}
    ,
        "example-plugin received input (len=0)",
    );
    // A missing plugin name is a stable typed error, not UnknownTool.
    try expectToolError(
        \\{"name":"plugin_run","arguments":{}}
    , error.MissingPluginName);
}

// Returns a minimal-but-valid `tools/call` payload for `tool_name`, exercising the
// dispatch arm without requiring tool-specific external state. Every catalog tool
// must be representable here; an unrepresented tool fails the parity test below.
fn minimalCallFor(tool_name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, tool_name, "ai_run"))
        return "{\"name\":\"ai_run\",\"arguments\":{\"input\":\"hello\"}}";
    if (std.mem.eql(u8, tool_name, "ai_complete"))
        return "{\"name\":\"ai_complete\",\"arguments\":{\"input\":\"hello\"}}";
    if (std.mem.eql(u8, tool_name, "ai_train"))
        return "{\"name\":\"ai_train\",\"arguments\":{\"profile\":\"abi\",\"dataset\":\"data.jsonl\"}}";
    if (std.mem.eql(u8, tool_name, "wdbx_query"))
        return "{\"name\":\"wdbx_query\",\"arguments\":{\"query\":\"hello\"}}";
    if (std.mem.eql(u8, tool_name, "scheduler_stats"))
        return "{\"name\":\"scheduler_stats\",\"arguments\":{}}";
    if (std.mem.eql(u8, tool_name, "scheduler_info"))
        return "{\"name\":\"scheduler_info\",\"arguments\":{}}";
    if (std.mem.eql(u8, tool_name, "connector_test"))
        return "{\"name\":\"connector_test\",\"arguments\":{\"service\":\"openai\",\"input\":\"hello\"}}";
    if (std.mem.eql(u8, tool_name, "gpu_status"))
        return "{\"name\":\"gpu_status\",\"arguments\":{}}";
    if (std.mem.eql(u8, tool_name, "plugin_list"))
        return "{\"name\":\"plugin_list\",\"arguments\":{}}";
    if (std.mem.eql(u8, tool_name, "wdbx_stats"))
        return "{\"name\":\"wdbx_stats\",\"arguments\":{}}";
    if (std.mem.eql(u8, tool_name, "plugin_run"))
        return "{\"name\":\"plugin_run\",\"arguments\":{\"name\":\"example-plugin\",\"input\":\"hello\"}}";
    return null;
}

// Every name advertised by the `tools/list` catalog must be dispatchable by
// `handleToolsCallJson` (i.e. it must not fall through to the UnknownTool arm).
// This pins catalog/dispatch parity so neither side can drift unnoticed.
test "MCP catalog and dispatch stay in parity" {
    const allocator = std.testing.allocator;

    const catalog_json = try handlers.handleToolsListJson(allocator);
    defer allocator.free(catalog_json);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, catalog_json, .{});
    defer parsed.deinit();

    const root = try expectObject(parsed.value);
    const tools = try expectArray(root.get("tools") orelse return error.MissingTools);
    try std.testing.expect(tools.items.len > 0);

    for (tools.items) |tool_val| {
        const tool = try expectObject(tool_val);
        const name = switch (tool.get("name") orelse return error.MissingToolName) {
            .string => |s| s,
            else => return error.ExpectedString,
        };

        const call = minimalCallFor(name) orelse {
            std.debug.print("catalog tool '{s}' has no minimal dispatch case in the parity test\n", .{name});
            return error.UnmappedCatalogTool;
        };

        const call_parsed = try std.json.parseFromSlice(std.json.Value, allocator, call, .{});
        defer call_parsed.deinit();

        // The call may fail for feature/state reasons, but it must never report
        // the tool as unknown — that would mean the catalog advertises a tool the
        // dispatcher cannot route.
        const response = handlers.handleToolsCallJson(allocator, call_parsed.value) catch |err| {
            try std.testing.expect(err != error.UnknownTool);
            continue;
        };
        allocator.free(response);
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
    try expectToolError("{\"name\":\"connector_test\",\"arguments\":{\"input\":\"hello\"}}", error.MissingConnectorService);
    try expectToolError("{\"name\":\"connector_test\",\"arguments\":{\"service\":\"unknown\",\"input\":\"hello\"}}", error.UnknownConnector);
}
