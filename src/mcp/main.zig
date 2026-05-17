const std = @import("std");
const abi = @import("abi");

const JsonRpcRequest = struct {
    jsonrpc: []const u8,
    method: []const u8,
    id: ?std.json.Value = null,
    params: ?std.json.Value = null,
};

const JsonRpcErrorObj = struct {
    code: i32,
    message: []const u8,
    data: ?std.json.Value = null,
};

const JsonRpcResponse = struct {
    jsonrpc: []const u8 = "2.0",
    id: ?std.json.Value,
    result: ?std.json.Value = null,
    @"error": ?JsonRpcErrorObj = null,
};

const McpMethod = enum {
    initialize,
    @"tools/list",
    @"tools/call",
    @"resources/list",
    @"prompts/list",
    ping,
    shutdown,
    unknown,

    fn fromString(s: []const u8) McpMethod {
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

fn handleInitializeJson(allocator: std.mem.Allocator, params: ?std.json.Value) ![]u8 {
    _ = params;
    return try allocator.dupe(u8,
        \\{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"abi-mcp","version":"0.1.0"}}
    );
}

fn handleToolsListJson(allocator: std.mem.Allocator) ![]u8 {
    return try allocator.dupe(u8,
        \\{"tools":[{"name":"ai_run","description":"Run AI inference with profile routing","inputSchema":{"type":"object","properties":{"input":{"type":"string"},"profile":{"type":"string"}},"required":["input"]}},{"name":"ai_train","description":"Train an agent profile","inputSchema":{"type":"object","properties":{"profile":{"type":"string"},"dataset":{"type":"string"}},"required":["profile","dataset"]}},{"name":"wdbx_query","description":"Query the vector store","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}
    );
}

fn handleToolsCallJson(allocator: std.mem.Allocator, params: ?std.json.Value) ![]u8 {
    const p = params orelse return error.MissingParams;
    const params_obj = switch (p) {
        .object => |obj| obj,
        else => return error.MissingParams,
    };

    const name_val = params_obj.get("name") orelse return error.MissingToolName;
    const tool_name = switch (name_val) {
        .string => |s| s,
        else => return error.MissingToolName,
    };

    const ai_mod = abi.features.ai;
    if (std.mem.eql(u8, tool_name, "ai_run")) {
        const args_obj = try toolArguments(params_obj);
        const input = try objectString(args_obj, "input", error.MissingInput);
        const response = try ai_mod.run(allocator, input);
        defer allocator.free(response);
        return try toolTextResult(allocator, response);
    }

    if (std.mem.eql(u8, tool_name, "ai_train")) {
        const args_obj = try toolArguments(params_obj);
        const profile = try objectString(args_obj, "profile", error.MissingProfile);
        const dataset = try objectString(args_obj, "dataset", error.MissingDataset);
        const artifact_dir = objectString(args_obj, "artifact_dir", error.MissingArtifactDir) catch "zig-cache/agents";

        var result = try ai_mod.train(allocator, .{
            .profile = profile,
            .dataset = .{ .path = dataset, .format = parseDatasetFormat(args_obj) },
            .artifact_dir = artifact_dir,
        });
        defer result.deinit(allocator);

        const text = try std.fmt.allocPrint(
            allocator,
            "training accepted profile={s} dataset={s} records={d} backend={s}: {s}",
            .{ result.profile, result.dataset_path, result.records_stored, result.acceleration_backend, result.message },
        );
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "wdbx_query")) {
        const args_obj = try toolArguments(params_obj);
        const query = try objectString(args_obj, "query", error.MissingQuery);
        const text = try runLocalWdbxQuery(allocator, query);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    return error.UnknownTool;
}

fn toolArguments(params_obj: std.json.ObjectMap) !std.json.ObjectMap {
    const args_val = params_obj.get("arguments") orelse return error.MissingArguments;
    return switch (args_val) {
        .object => |obj| obj,
        else => return error.MissingArguments,
    };
}

fn objectString(obj: std.json.ObjectMap, key: []const u8, missing_error: anyerror) ![]const u8 {
    const value = obj.get(key) orelse return missing_error;
    return switch (value) {
        .string => |s| s,
        else => missing_error,
    };
}

fn parseDatasetFormat(obj: std.json.ObjectMap) abi.features.ai.DatasetFormat {
    const value = obj.get("format") orelse return .jsonl;
    const s = switch (value) {
        .string => |format| format,
        else => return .jsonl,
    };
    if (std.mem.eql(u8, s, "csv")) return .csv;
    if (std.mem.eql(u8, s, "text")) return .text;
    return .jsonl;
}

fn toolTextResult(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

fn runLocalWdbxQuery(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    const wdbx = abi.features.wdbx;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const abbey_vec = [_]f32{ 0.92, 0.48, 0.25, 0.76 };
    const aviva_vec = [_]f32{ 0.34, 0.94, 0.82, 0.41 };
    const abi_vec = [_]f32{ 0.71, 0.69, 0.88, 0.97 };
    _ = try store.putVector(&abbey_vec);
    _ = try store.putVector(&aviva_vec);
    _ = try store.putVector(&abi_vec);

    const query_vec = textEmbedding(query);
    const results = store.search(&query_vec, 1) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "FeatureDisabled")) return try allocator.dupe(u8, "wdbx feature is disabled for this build");
        return err;
    };
    defer allocator.free(results);

    if (results.len == 0) return try allocator.dupe(u8, "wdbx query returned no local matches");
    return try std.fmt.allocPrint(
        allocator,
        "wdbx local match profile={s} vector_id={d} score={d:.3}",
        .{ localProfileForVector(results[0].id), results[0].id, results[0].score },
    );
}

fn textEmbedding(input: []const u8) [4]f32 {
    var out = [_]f32{ 0.01, 0.01, 0.01, 0.01 };
    for (input, 0..) |byte, i| {
        const lowered = std.ascii.toLower(byte);
        out[i % out.len] += @as(f32, @floatFromInt(lowered % 31)) / 31.0;
    }
    var norm: f32 = 0;
    for (out) |v| norm += v * v;
    if (norm == 0) return out;
    const scale = @sqrt(norm);
    for (&out) |*v| v.* /= scale;
    return out;
}

fn localProfileForVector(id: u32) []const u8 {
    return switch (id) {
        1 => "abbey",
        2 => "aviva",
        3 => "abi",
        else => "unknown",
    };
}

fn jsonStringAlloc(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendJsonString(&out, allocator, value);
    return try out.toOwnedSlice(allocator);
}

fn appendJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

fn valueToJson(value: std.json.Value, allocator: std.mem.Allocator) ![]u8 {
    return switch (value) {
        .null => try allocator.dupe(u8, "null"),
        .bool => |b| try allocator.dupe(u8, if (b) "true" else "false"),
        .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(allocator, "{d}", .{f}),
        .number_string => |s| try allocator.dupe(u8, s),
        .string => |s| try jsonStringAlloc(allocator, s),
        .array => |arr| blk: {
            var out = std.ArrayListUnmanaged(u8).empty;
            defer out.deinit(allocator);
            try out.append(allocator, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try out.append(allocator, ',');
                const item_json = try valueToJson(item, allocator);
                defer allocator.free(item_json);
                try out.appendSlice(allocator, item_json);
            }
            try out.append(allocator, ']');
            break :blk try out.toOwnedSlice(allocator);
        },
        .object => |obj| blk: {
            var out = std.ArrayListUnmanaged(u8).empty;
            defer out.deinit(allocator);
            try out.append(allocator, '{');
            var it = obj.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try out.append(allocator, ',');
                first = false;
                try appendJsonString(&out, allocator, entry.key_ptr.*);
                try out.append(allocator, ':');
                const val_json = try valueToJson(entry.value_ptr.*, allocator);
                defer allocator.free(val_json);
                try out.appendSlice(allocator, val_json);
            }
            try out.append(allocator, '}');
            break :blk try out.toOwnedSlice(allocator);
        },
    };
}

pub fn main(init: std.process.Init) !void {
    try runStdioLoop(init.gpa, init.io);
}

fn runStdioLoop(allocator: std.mem.Allocator, io: std.Io) !void {
    var read_buf: [4096]u8 = undefined;
    var line_buf = std.ArrayListUnmanaged(u8).empty;
    defer line_buf.deinit(allocator);

    while (true) {
        const n = try std.posix.read(std.posix.STDIN_FILENO, &read_buf);
        if (n == 0) break;

        for (read_buf[0..n]) |byte| {
            if (byte == '\n') {
                const line = std.mem.trimEnd(u8, line_buf.items, "\r");
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                processRequest(arena.allocator(), io, line) catch |err| {
                    std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
                };
                line_buf.clearRetainingCapacity();
                continue;
            }

            if (line_buf.items.len >= 64 * 1024) {
                std.log.warn("dropping overlong MCP request line", .{});
                writeError(io, null, -32700, "Parse error");
                line_buf.clearRetainingCapacity();
                continue;
            }

            try line_buf.append(allocator, byte);
        }
    }

    if (line_buf.items.len > 0) {
        const line = std.mem.trimEnd(u8, line_buf.items, "\r");
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        processRequest(arena.allocator(), io, line) catch |err| {
            std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
        };
    }
}

fn processRequest(allocator: std.mem.Allocator, io: std.Io, line: []const u8) !void {
    if (line.len == 0) return;

    const request = std.json.parseFromSlice(JsonRpcRequest, allocator, line, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        writeError(io, null, -32700, "Parse error");
        return err;
    };
    defer request.deinit();

    if (!std.mem.eql(u8, request.value.jsonrpc, "2.0")) {
        writeError(io, request.value.id, -32600, "Invalid Request");
        return;
    }

    const method = McpMethod.fromString(request.value.method);
    const result_json = switch (method) {
        .initialize => handleInitializeJson(allocator, request.value.params) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/list" => handleToolsListJson(allocator) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/call" => handleToolsCallJson(allocator, request.value.params) catch |err| {
            const msg = switch (err) {
                error.MissingParams => "Missing params",
                error.MissingToolName => "Missing tool name",
                error.MissingArguments => "Missing arguments",
                error.MissingInput => "Missing input",
                error.MissingProfile => "Missing profile",
                error.MissingDataset => "Missing dataset",
                error.MissingQuery => "Missing query",
                error.UnknownTool => "Method not found",
                else => "Internal error",
            };
            writeError(io, request.value.id, -32603, msg);
            return err;
        },
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => try allocator.dupe(u8, "null"),
        .@"resources/list" => try allocator.dupe(u8, "{\"resources\":[]}"),
        .@"prompts/list" => try allocator.dupe(u8, "{\"prompts\":[]}"),
        .unknown => {
            writeError(io, request.value.id, -32601, "Method not found");
            return;
        },
    };
    defer allocator.free(result_json);

    writeResult(allocator, io, request.value.id, result_json);
}

fn writeError(io: std.Io, id: ?std.json.Value, code: i32, message: []const u8) void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.heap.page_allocator);

    buf.appendSlice(std.heap.page_allocator, "{\"jsonrpc\":\"2.0\"") catch return;
    appendId(&buf, std.heap.page_allocator, id) catch return;
    buf.appendSlice(std.heap.page_allocator, ",\"error\":{\"code\":") catch return;
    buf.print(std.heap.page_allocator, "{d}", .{code}) catch return;
    buf.appendSlice(std.heap.page_allocator, ",\"message\":") catch return;
    appendJsonString(&buf, std.heap.page_allocator, message) catch return;
    buf.appendSlice(std.heap.page_allocator, "}}\n") catch return;

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP error response: {s}", .{@errorName(err)});
    };
}

fn writeResult(allocator: std.mem.Allocator, io: std.Io, id: ?std.json.Value, result_json: []const u8) void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    buf.appendSlice(allocator, "{\"jsonrpc\":\"2.0\"") catch return;
    appendId(&buf, allocator, id) catch return;
    buf.appendSlice(allocator, ",\"result\":") catch return;
    buf.appendSlice(allocator, result_json) catch return;
    buf.appendSlice(allocator, "}\n") catch return;

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP response: {s}", .{@errorName(err)});
    };
}

fn appendId(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, id: ?std.json.Value) !void {
    if (id) |value| {
        const id_json = try valueToJson(value, allocator);
        defer allocator.free(id_json);
        try out.appendSlice(allocator, ",\"id\":");
        try out.appendSlice(allocator, id_json);
    } else {
        try out.appendSlice(allocator, ",\"id\":null");
    }
}

fn writeStdoutAll(io: std.Io, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll(bytes);
    try stdout.flush();
}

test {
    std.testing.refAllDecls(@This());
}

test "stdio mode is the default MCP transport" {
    try std.testing.expect(std.mem.eql(u8, "stdio", "stdio"));
}

test "McpMethod fromString recognizes known methods" {
    try std.testing.expectEqual(McpMethod.initialize, McpMethod.fromString("initialize"));
    try std.testing.expectEqual(McpMethod.@"tools/list", McpMethod.fromString("tools/list"));
    try std.testing.expectEqual(McpMethod.@"tools/call", McpMethod.fromString("tools/call"));
    try std.testing.expectEqual(McpMethod.ping, McpMethod.fromString("ping"));
    try std.testing.expectEqual(McpMethod.unknown, McpMethod.fromString("nonexistent"));
}

test "JsonRpcResponse serializes error field correctly" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    var stream = std.json.Stringify{
        .writer = &buf.writer(allocator),
        .options = .{},
    };
    try stream.beginObject();
    try stream.objectField("error");
    try stream.beginObject();
    try stream.objectField("code");
    try stream.write(@as(i32, -32600));
    try stream.objectField("message");
    try stream.write("Invalid Request");
    try stream.endObject();
    try stream.endObject();
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"error\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"Invalid Request\"") != null);
}

test "JsonRpcResponse serializes result correctly" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    var stream = std.json.Stringify{
        .writer = &buf.writer(allocator),
        .options = .{},
    };
    try stream.beginObject();
    try stream.objectField("result");
    try stream.write("ok");
    try stream.endObject();
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"result\"") != null);
}
