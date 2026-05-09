const std = @import("std");
const types = @import("../types.zig");
const json_write = @import("json_write.zig");

pub fn handleToolsList(self: anytype, writer: anytype, id: ?types.RequestId) !void {
    const rid = id orelse return;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(self.allocator);

    try buf.appendSlice(self.allocator, "{\"tools\":[");

    for (self.tools.items, 0..) |tool, i| {
        if (i > 0) try buf.append(self.allocator, ',');
        try buf.appendSlice(self.allocator, "{\"name\":\"");
        try buf.appendSlice(self.allocator, tool.def.name);
        try buf.appendSlice(self.allocator, "\",\"description\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, tool.def.description);
        try buf.appendSlice(self.allocator, "\",\"inputSchema\":");
        try buf.appendSlice(self.allocator, tool.def.input_schema);
        try buf.append(self.allocator, '}');
    }

    try buf.appendSlice(self.allocator, "]}");
    try types.writeResponse(writer, rid, buf.items);
}

pub fn handleToolsCall(
    self: anytype,
    writer: anytype,
    id: ?types.RequestId,
    params: ?std.json.ObjectMap,
) !void {
    const rid = id orelse return;

    const p = params orelse {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing params");
        return;
    };

    const name_val = p.get("name") orelse {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing tool name");
        return;
    };
    if (name_val != .string) {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Tool name must be string");
        return;
    }
    const tool_name = name_val.string;

    const args: ?std.json.ObjectMap = if (p.get("arguments")) |a|
        (if (a == .object) a.object else null)
    else
        null;

    for (self.tools.items) |tool| {
        if (std.mem.eql(u8, tool.def.name, tool_name)) {
            var result_buf = std.ArrayListUnmanaged(u8).empty;
            defer result_buf.deinit(self.allocator);

            tool.handler(self.allocator, args, &result_buf) catch |err| {
                var err_buf = std.ArrayListUnmanaged(u8).empty;
                defer err_buf.deinit(self.allocator);

                err_buf.appendSlice(self.allocator, "{\"content\":[{\"type\":\"text\",\"text\":\"Error: ") catch |alloc_err| {
                    std.log.err("MCP: OOM formatting tool error: {t}", .{alloc_err});
                    return;
                };
                var err_msg_buf: [128]u8 = undefined;
                const err_msg = std.fmt.bufPrint(&err_msg_buf, "{t}", .{err}) catch "unknown error";
                json_write.appendJsonEscaped(self.allocator, &err_buf, err_msg) catch |alloc_err| {
                    std.log.err("MCP: OOM escaping tool error message: {t}", .{alloc_err});
                    return;
                };
                err_buf.appendSlice(self.allocator, "\"}],\"isError\":true}") catch |alloc_err| {
                    std.log.err("MCP: OOM formatting tool error suffix: {t}", .{alloc_err});
                    return;
                };
                types.writeResponse(writer, rid, err_buf.items) catch |write_err| {
                    std.log.err("MCP: failed to write tool error response: {t}", .{write_err});
                    return;
                };
                return;
            };

            var out = std.ArrayListUnmanaged(u8).empty;
            defer out.deinit(self.allocator);

            try out.appendSlice(self.allocator, "{\"content\":[{\"type\":\"text\",\"text\":\"");
            try json_write.appendJsonEscaped(self.allocator, &out, result_buf.items);
            try out.appendSlice(self.allocator, "\"}]}");
            try types.writeResponse(writer, rid, out.items);
            return;
        }
    }

    try types.writeError(writer, rid, types.ErrorCode.method_not_found, "Unknown tool");
}

test {
    std.testing.refAllDecls(@This());
}
