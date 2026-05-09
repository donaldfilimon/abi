const std = @import("std");
const types = @import("../types.zig");
const json_write = @import("json_write.zig");
const resources = @import("resources.zig");
const tools = @import("tools.zig");

pub fn handleMessage(self: anytype, line: []const u8, writer: anytype) !void {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        self.allocator,
        line,
        .{},
    ) catch {
        try types.writeError(writer, null, types.ErrorCode.parse_error, "Parse error");
        return;
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) {
        try types.writeError(writer, null, types.ErrorCode.invalid_request, "Expected JSON object");
        return;
    }

    const obj = root.object;
    const id: ?types.RequestId = if (obj.get("id")) |id_val|
        types.RequestId.fromJson(id_val)
    else
        null;

    const ver = obj.get("jsonrpc") orelse {
        try types.writeError(writer, id, types.ErrorCode.invalid_request, "Missing required jsonrpc field");
        return;
    };
    if (ver != .string or !std.mem.eql(u8, ver.string, "2.0")) {
        try types.writeError(writer, id, types.ErrorCode.invalid_request, "Invalid JSON-RPC version");
        return;
    }

    const method_val = obj.get("method") orelse {
        try types.writeError(writer, id, types.ErrorCode.invalid_request, "Missing method");
        return;
    };
    if (method_val != .string) {
        try types.writeError(writer, id, types.ErrorCode.invalid_request, "Method must be string");
        return;
    }
    const method = method_val.string;

    const params: ?std.json.ObjectMap = if (obj.get("params")) |p|
        (if (p == .object) p.object else null)
    else
        null;

    if (std.mem.eql(u8, method, "initialize")) {
        try handleInitialize(self, writer, id);
    } else if (std.mem.eql(u8, method, "notifications/initialized")) {
        self.initialized = true;
    } else if (std.mem.eql(u8, method, "tools/list")) {
        if (!checkAuth(self, params, writer, id)) return;
        try tools.handleToolsList(self, writer, id);
    } else if (std.mem.eql(u8, method, "tools/call")) {
        if (!checkAuth(self, params, writer, id)) return;
        try tools.handleToolsCall(self, writer, id, params);
    } else if (std.mem.eql(u8, method, "resources/list")) {
        if (!checkAuth(self, params, writer, id)) return;
        try resources.handleResourcesList(self, writer, id);
    } else if (std.mem.eql(u8, method, "resources/read")) {
        if (!checkAuth(self, params, writer, id)) return;
        try resources.handleResourcesRead(self, writer, id, params);
    } else if (std.mem.eql(u8, method, "resources/subscribe")) {
        if (!checkAuth(self, params, writer, id)) return;
        try handleResourceSubscribe(self, writer, id, params);
    } else if (std.mem.eql(u8, method, "resources/unsubscribe")) {
        if (!checkAuth(self, params, writer, id)) return;
        try handleResourceUnsubscribe(self, writer, id, params);
    } else if (std.mem.eql(u8, method, "ping")) {
        try handlePing(writer, id);
    } else {
        try types.writeError(writer, id, types.ErrorCode.method_not_found, "Method not found");
    }
}

pub fn handleInitialize(self: anytype, writer: anytype, id: ?types.RequestId) !void {
    const rid = id orelse return;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(self.allocator);

    try buf.appendSlice(self.allocator, "{\"protocolVersion\":\"");
    try buf.appendSlice(self.allocator, types.PROTOCOL_VERSION);
    try buf.appendSlice(self.allocator, "\",\"capabilities\":{\"tools\":{\"listChanged\":false}");
    if (self.resources.items.len > 0) {
        try buf.appendSlice(self.allocator, ",\"resources\":{\"subscribe\":true,\"listChanged\":false}");
    }
    try buf.appendSlice(self.allocator, "}");
    try buf.appendSlice(self.allocator, ",\"serverInfo\":{\"name\":\"");
    try json_write.appendJsonEscaped(self.allocator, &buf, self.server_name);
    try buf.appendSlice(self.allocator, "\",\"version\":\"");
    try json_write.appendJsonEscaped(self.allocator, &buf, self.server_version);
    try buf.appendSlice(self.allocator, "\"}}");

    try types.writeResponse(writer, rid, buf.items);
}

pub fn handlePing(writer: anytype, id: ?types.RequestId) !void {
    const rid = id orelse return;
    try types.writeResponse(writer, rid, "{}");
}

/// Validate the request against the server's auth_token, if configured.
/// Returns `true` if the request is authorized (or auth is disabled).
/// Returns `false` and writes a JSON-RPC error if the token is missing or wrong.
fn checkAuth(self: anytype, params: ?std.json.ObjectMap, writer: anytype, id: ?types.RequestId) bool {
    const expected = self.auth_token orelse return true; // auth disabled
    if (params) |p| {
        if (p.get("_auth_token")) |token_val| {
            if (token_val == .string and std.mem.eql(u8, token_val.string, expected)) {
                return true;
            }
        }
    }
    types.writeError(writer, id, types.ErrorCode.unauthorized, "Unauthorized") catch |err| std.log.warn("failed to write auth error: {}", .{err});
    return false;
}

fn extractUri(params: ?std.json.ObjectMap) ?[]const u8 {
    const p = params orelse return null;
    const v = p.get("uri") orelse return null;
    return if (v == .string) v.string else null;
}

fn handleResourceSubscribe(self: anytype, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
    const rid = id orelse return;
    const uri = extractUri(params) orelse {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing uri parameter");
        return;
    };
    _ = try self.subscribeResource(uri);
    try types.writeResponse(writer, rid, "{}");
}

fn handleResourceUnsubscribe(self: anytype, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
    const rid = id orelse return;
    const uri = extractUri(params) orelse {
        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing uri parameter");
        return;
    };
    _ = self.unsubscribeResource(uri);
    try types.writeResponse(writer, rid, "{}");
}

test "resources/subscribe round-trip" {
    const lifecycle = @import("lifecycle.zig");
    var server = lifecycle.Server.init(std.testing.allocator, "test", "0.1");
    defer server.deinit();

    // Subscribe
    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"resources/subscribe\",\"params\":{\"uri\":\"abi://status\"}}";
    try handleMessage(&server, req, &writer);
    const out = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"result\":{}") != null);
    try std.testing.expect(server.isSubscribed("abi://status"));

    // Unsubscribe
    writer = std.Io.Writer.fixed(&buf);
    const req2 = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"resources/unsubscribe\",\"params\":{\"uri\":\"abi://status\"}}";
    try handleMessage(&server, req2, &writer);
    try std.testing.expect(!server.isSubscribed("abi://status"));
}

fn testResourceHandler(_: std.mem.Allocator, _: []const u8, _: *std.ArrayListUnmanaged(u8)) anyerror!void {}

test "resources/subscribe capability advertised" {
    const lifecycle = @import("lifecycle.zig");
    var server = lifecycle.Server.init(std.testing.allocator, "test", "0.1");
    defer server.deinit();
    try server.addResource(.{
        .def = .{ .uri = "abi://test", .name = "test", .description = "test resource" },
        .handler = &testResourceHandler,
    });

    var buf: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    try handleMessage(&server, req, &writer);
    const out = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"subscribe\":true") != null);
}

test {
    std.testing.refAllDecls(@This());
}
