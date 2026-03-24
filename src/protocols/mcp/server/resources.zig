const std = @import("std");
const types = @import("../types.zig");
const json_write = @import("json_write.zig");

pub fn handleResourcesList(self: anytype, writer: anytype, id: ?types.RequestId) !void {
    const rid = id orelse return;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(self.allocator);

    try buf.appendSlice(self.allocator, "{\"resources\":[");

    for (self.resources.items, 0..) |resource, i| {
        if (i > 0) try buf.append(self.allocator, ',');
        try buf.appendSlice(self.allocator, "{\"uri\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, resource.def.uri);
        try buf.appendSlice(self.allocator, "\",\"name\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, resource.def.name);
        try buf.appendSlice(self.allocator, "\",\"description\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, resource.def.description);
        try buf.appendSlice(self.allocator, "\",\"mimeType\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, resource.def.mime_type);
        try buf.appendSlice(self.allocator, "\"}");
    }

    try buf.appendSlice(self.allocator, "]}");
    try types.writeResponse(writer, rid, buf.items);
}

pub fn handleResourcesRead(
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

    const uri_val = p.get("uri") orelse {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing resource URI");
        return;
    };
    if (uri_val != .string) {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource URI must be string");
        return;
    }
    const uri = uri_val.string;

    for (self.resources.items) |resource| {
        if (std.mem.eql(u8, resource.def.uri, uri)) {
            var result_buf = std.ArrayListUnmanaged(u8).empty;
            defer result_buf.deinit(self.allocator);

            resource.handler(self.allocator, uri, &result_buf) catch |err| {
                var err_buf = std.ArrayListUnmanaged(u8).empty;
                defer err_buf.deinit(self.allocator);

                err_buf.appendSlice(self.allocator, "{\"contents\":[{\"uri\":\"") catch |alloc_err| {
                    std.log.err("MCP: OOM formatting resource error: {t}", .{alloc_err});
                    return;
                };
                json_write.appendJsonEscaped(self.allocator, &err_buf, uri) catch |alloc_err| {
                    std.log.err("MCP: OOM escaping resource URI: {t}", .{alloc_err});
                    return;
                };
                err_buf.appendSlice(self.allocator, "\",\"mimeType\":\"text/plain\",\"text\":\"Error: ") catch |alloc_err| {
                    std.log.err("MCP: OOM formatting resource error mid: {t}", .{alloc_err});
                    return;
                };
                var err_msg_buf: [128]u8 = undefined;
                const err_msg = std.fmt.bufPrint(&err_msg_buf, "{t}", .{err}) catch "unknown error";
                json_write.appendJsonEscaped(self.allocator, &err_buf, err_msg) catch |alloc_err| {
                    std.log.err("MCP: OOM escaping resource error message: {t}", .{alloc_err});
                    return;
                };
                err_buf.appendSlice(self.allocator, "\"}]}") catch |alloc_err| {
                    std.log.err("MCP: OOM formatting resource error suffix: {t}", .{alloc_err});
                    return;
                };
                types.writeResponse(writer, rid, err_buf.items) catch |write_err| {
                    std.log.err("MCP: failed to write resource error response: {t}", .{write_err});
                    return;
                };
                return;
            };

            var out = std.ArrayListUnmanaged(u8).empty;
            defer out.deinit(self.allocator);

            try out.appendSlice(self.allocator, "{\"contents\":[{\"uri\":\"");
            try json_write.appendJsonEscaped(self.allocator, &out, uri);
            try out.appendSlice(self.allocator, "\",\"mimeType\":\"");
            try json_write.appendJsonEscaped(self.allocator, &out, resource.def.mime_type);
            try out.appendSlice(self.allocator, "\",\"text\":\"");
            try json_write.appendJsonEscaped(self.allocator, &out, result_buf.items);
            try out.appendSlice(self.allocator, "\"}]}");
            try types.writeResponse(writer, rid, out.items);
            return;
        }
    }

    try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource not found");
}
