//! ACP HTTP route dispatch, request parsing, and path matching.

const std = @import("std");
const AgentCard = @import("agent_card.zig").AgentCard;
const Server = @import("mod.zig").Server;

pub const HttpError = std.mem.Allocator.Error || error{
    InvalidAddress,
    ListenFailed,
    ReadFailed,
    RequestTooLarge,
};

const max_http_body_bytes = 256 * 1024;

pub fn dispatchHttpRequest(
    allocator: std.mem.Allocator,
    acp_server: *Server,
    card: AgentCard,
    request: *std.http.Server.Request,
) !void {
    const target = request.head.target;
    const path = splitPath(target);

    if (std.mem.eql(u8, path, "/.well-known/agent.json")) {
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const body = try card.toJson(allocator);
        defer allocator.free(body);
        return respondJson(request, body, .ok);
    }

    if (std.mem.startsWith(u8, path, "/tasks")) {
        return handleTasksHttpRoute(allocator, acp_server, request, path);
    }

    if (std.mem.startsWith(u8, path, "/sessions")) {
        return handleSessionsHttpRoute(allocator, acp_server, request, path);
    }

    return respondJson(request, "{\"error\":\"not found\"}", .not_found);
}

pub fn splitPath(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| {
        return target[0..idx];
    }
    return target;
}

fn handleTasksHttpRoute(
    allocator: std.mem.Allocator,
    acp_server: *Server,
    request: *std.http.Server.Request,
    path: []const u8,
) !void {
    if (std.mem.eql(u8, path, "/tasks")) {
        return respondJson(request, "{\"error\":\"not found\"}", .not_found);
    }
    if (std.mem.eql(u8, path, "/tasks/send")) {
        if (request.head.method != .POST) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const body = readRequestBody(allocator, request) catch |err| switch (err) {
            HttpError.RequestTooLarge => {
                return respondJson(request, "{\"error\":\"payload too large\"}", .payload_too_large);
            },
            HttpError.ReadFailed => {
                return respondJson(request, "{\"error\":\"invalid body\"}", .bad_request);
            },
            else => return err,
        };
        defer allocator.free(body);
        const message = extractMessage(allocator, body) catch body;
        defer if (message.ptr != body.ptr) allocator.free(message);
        const id = try acp_server.createTask(message);
        var buf: [128]u8 = undefined;
        const json_body = std.fmt.bufPrint(&buf, "{{\"id\":\"{s}\"}}", .{id}) catch return error.OutOfMemory;
        return respondJson(request, json_body, .created);
    }
    if (std.mem.startsWith(u8, path, "/tasks/")) {
        const id = path["/tasks/".len..];
        if (id.len == 0) return respondJson(request, "{\"error\":\"not found\"}", .not_found);
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const task = acp_server.getTask(id) orelse {
            return respondJson(request, "{\"error\":\"not found\"}", .not_found);
        };
        const body = try task.toJson(allocator);
        defer allocator.free(body);
        return respondJson(request, body, .ok);
    }
    return respondJson(request, "{\"error\":\"not found\"}", .not_found);
}

fn handleSessionsHttpRoute(
    allocator: std.mem.Allocator,
    acp_server: *Server,
    request: *std.http.Server.Request,
    path: []const u8,
) !void {
    if (std.mem.eql(u8, path, "/sessions")) {
        if (request.head.method != .POST) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const body = readRequestBody(allocator, request) catch |err| switch (err) {
            HttpError.RequestTooLarge => {
                return respondJson(request, "{\"error\":\"payload too large\"}", .payload_too_large);
            },
            HttpError.ReadFailed => {
                return respondJson(request, "{\"error\":\"invalid body\"}", .bad_request);
            },
            else => return err,
        };
        defer allocator.free(body);
        const metadata = extractMessage(allocator, body) catch null;
        defer if (metadata) |m| if (m.ptr != body.ptr) allocator.free(m);
        const id = try acp_server.createSession(metadata);
        var buf: [128]u8 = undefined;
        const json_body = std.fmt.bufPrint(&buf, "{{\"id\":\"{s}\"}}", .{id}) catch return error.OutOfMemory;
        return respondJson(request, json_body, .created);
    }
    if (std.mem.startsWith(u8, path, "/sessions/")) {
        const remainder = path["/sessions/".len..];
        if (remainder.len == 0) return respondJson(request, "{\"error\":\"not found\"}", .not_found);

        // Check for /sessions/{id}/tasks sub-route
        if (std.mem.indexOf(u8, remainder, "/tasks")) |idx| {
            const session_id = remainder[0..idx];
            if (request.head.method == .POST) {
                const body = readRequestBody(allocator, request) catch |err| switch (err) {
                    HttpError.RequestTooLarge => {
                        return respondJson(request, "{\"error\":\"payload too large\"}", .payload_too_large);
                    },
                    HttpError.ReadFailed => {
                        return respondJson(request, "{\"error\":\"invalid body\"}", .bad_request);
                    },
                    else => return err,
                };
                defer allocator.free(body);
                const task_id = extractMessage(allocator, body) catch body;
                defer if (task_id.ptr != body.ptr) allocator.free(task_id);
                acp_server.addTaskToSession(session_id, task_id) catch |err| switch (err) {
                    error.SessionNotFound => {
                        return respondJson(request, "{\"error\":\"session not found\"}", .not_found);
                    },
                    else => return err,
                };
                return respondJson(request, "{\"ok\":true}", .ok);
            }
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }

        // /sessions/{id} — get session
        const session_id = remainder;
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const session = acp_server.getSession(session_id) orelse {
            return respondJson(request, "{\"error\":\"not found\"}", .not_found);
        };
        const body = try session.toJson(allocator);
        defer allocator.free(body);
        return respondJson(request, body, .ok);
    }
    return respondJson(request, "{\"error\":\"not found\"}", .not_found);
}

pub fn extractMessage(allocator: std.mem.Allocator, body: []const u8) ![]const u8 {
    const trimmed = std.mem.trim(u8, body, " \t\r\n");
    if (trimmed.len == 0) return body;
    if (trimmed[0] == '{') {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return body;
        defer parsed.deinit();
        if (parsed.value == .object) {
            if (parsed.value.object.get("message")) |v| {
                if (v == .string) return allocator.dupe(u8, v.string);
            }
        }
    }
    return body;
}

pub fn readRequestBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) HttpError![]u8 {
    var buffer: [4096]u8 = undefined;
    const reader = request.readerExpectContinue(&buffer) catch return HttpError.ReadFailed;
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch return HttpError.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > max_http_body_bytes) return HttpError.RequestTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
        if (n < chunk.len) break;
    }
    return list.toOwnedSlice(allocator);
}

pub fn respondJson(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
) !void {
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "application/json" },
    };
    try request.respond(body, .{
        .status = status,
        .extra_headers = &headers,
    });
}
