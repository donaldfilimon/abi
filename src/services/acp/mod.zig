//! ACP (Agent Communication Protocol) Service
//!
//! Provides an HTTP server implementing the Agent Communication Protocol
//! for agent-to-agent communication. Exposes an agent card at
//! `/.well-known/agent.json` and task management endpoints.
//!
//! ## Usage
//! ```bash
//! abi acp serve --port 8080
//! curl http://localhost:8080/.well-known/agent.json
//! ```

const std = @import("std");
const net_utils = @import("../shared/utils.zig").net;

/// ACP Agent Card — describes this agent's capabilities
pub const AgentCard = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8,
    url: []const u8,
    capabilities: Capabilities,

    pub const Capabilities = struct {
        streaming: bool = false,
        pushNotifications: bool = false,
    };

    /// Serialize to JSON (escapes all string fields for safety)
    pub fn toJson(self: AgentCard, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"name\":\"");
        try appendEscaped(allocator, &buf, self.name);
        try buf.appendSlice(allocator, "\",\"description\":\"");
        try appendEscaped(allocator, &buf, self.description);
        try buf.appendSlice(allocator, "\",\"version\":\"");
        try appendEscaped(allocator, &buf, self.version);
        try buf.appendSlice(allocator, "\",\"url\":\"");
        try appendEscaped(allocator, &buf, self.url);
        try buf.appendSlice(allocator, "\",\"capabilities\":{\"streaming\":");
        try buf.appendSlice(allocator, if (self.capabilities.streaming) "true" else "false");
        try buf.appendSlice(allocator, ",\"pushNotifications\":");
        try buf.appendSlice(allocator, if (self.capabilities.pushNotifications) "true" else "false");
        try buf.appendSlice(allocator, "},\"skills\":[{\"id\":\"db_query\",\"name\":\"Vector Search\",\"description\":\"Search the WDBX vector database\"},{\"id\":\"db_insert\",\"name\":\"Vector Insert\",\"description\":\"Insert vectors with metadata\"},{\"id\":\"agent_chat\",\"name\":\"Chat\",\"description\":\"Conversational interaction\"}]}");

        return buf.toOwnedSlice(allocator);
    }
};

/// Task status in the ACP lifecycle
pub const TaskStatus = enum {
    submitted,
    working,
    input_required,
    completed,
    failed,
    canceled,

    pub fn toString(self: TaskStatus) []const u8 {
        return switch (self) {
            .submitted => "submitted",
            .working => "working",
            .input_required => "input-required",
            .completed => "completed",
            .failed => "failed",
            .canceled => "canceled",
        };
    }
};

/// ACP Task
pub const Task = struct {
    id: []const u8,
    status: TaskStatus,
    messages: std.ArrayListUnmanaged(Message),

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn deinit(self: *Task, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        for (self.messages.items) |msg| {
            allocator.free(msg.role);
            allocator.free(msg.content);
        }
        self.messages.deinit(allocator);
    }

    /// Serialize task to JSON
    pub fn toJson(self: *const Task, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"id\":\"");
        try appendEscaped(allocator, &buf, self.id);
        try buf.appendSlice(allocator, "\",\"status\":\"");
        try buf.appendSlice(allocator, self.status.toString());
        try buf.appendSlice(allocator, "\",\"messages\":[");

        for (self.messages.items, 0..) |msg, i| {
            if (i > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"role\":\"");
            try appendEscaped(allocator, &buf, msg.role);
            try buf.appendSlice(allocator, "\",\"parts\":[{\"type\":\"text\",\"text\":\"");
            try appendEscaped(allocator, &buf, msg.content);
            try buf.appendSlice(allocator, "\"}]}");
        }

        try buf.appendSlice(allocator, "]}");
        return buf.toOwnedSlice(allocator);
    }
};

/// ACP Server that manages tasks
pub const Server = struct {
    allocator: std.mem.Allocator,
    card: AgentCard,
    tasks: std.StringHashMapUnmanaged(Task),
    next_task_id: u64,

    pub fn init(allocator: std.mem.Allocator, card: AgentCard) Server {
        return .{
            .allocator = allocator,
            .card = card,
            .tasks = .empty,
            .next_task_id = 1,
        };
    }

    pub fn deinit(self: *Server) void {
        var it = self.tasks.iterator();
        while (it.next()) |entry| {
            var task = entry.value_ptr;
            task.deinit(self.allocator);
        }
        self.tasks.deinit(self.allocator);
    }

    /// Create a new task from a message
    pub fn createTask(self: *Server, message: []const u8) ![]const u8 {
        var id_buf: [32]u8 = undefined;
        const id_str = std.fmt.bufPrint(&id_buf, "task-{d}", .{self.next_task_id}) catch "task-0";
        self.next_task_id += 1;

        const id = try self.allocator.dupe(u8, id_str);

        var task = Task{
            .id = id,
            .status = .submitted,
            .messages = .empty,
        };
        // Single errdefer for the whole task — frees id, messages list, and each message's role+content
        errdefer task.deinit(self.allocator);

        // Build message inline to avoid double-free: once append succeeds,
        // task.deinit owns the strings. No separate errdefer needed.
        try task.messages.ensureUnusedCapacity(self.allocator, 1);
        const role = try self.allocator.dupe(u8, "user");
        const content = self.allocator.dupe(u8, message) catch |err| {
            self.allocator.free(role);
            return err;
        };
        // ensureUnusedCapacity guarantees append won't fail
        task.messages.appendAssumeCapacity(.{ .role = role, .content = content });

        try self.tasks.put(self.allocator, id, task);
        return id;
    }

    /// Get a task by ID
    pub fn getTask(self: *Server, id: []const u8) ?*Task {
        return self.tasks.getPtr(id);
    }

    /// Get the number of tasks
    pub fn taskCount(self: *const Server) u32 {
        return self.tasks.count();
    }
};

fn appendEscaped(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var hex_buf: [6]u8 = undefined;
                    const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch continue;
                    try buf.appendSlice(allocator, hex);
                } else {
                    try buf.append(allocator, c);
                }
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// HTTP Server
// ═══════════════════════════════════════════════════════════════

pub const HttpError = std.mem.Allocator.Error || error{
    InvalidAddress,
    ListenFailed,
    ReadFailed,
    RequestTooLarge,
};

const max_http_body_bytes = 256 * 1024;

/// Run the ACP HTTP server loop. Blocks until the process exits.
/// Caller must provide an I/O backend (e.g. from std.Io.Threaded).
pub fn serveHttp(
    allocator: std.mem.Allocator,
    io: std.Io,
    address: []const u8,
    card: AgentCard,
) HttpError!void {
    var acp_server = Server.init(allocator, card);
    defer acp_server.deinit();

    const listen_addr = try resolveHttpAddress(io, allocator, address);
    var listener = listen_addr.listen(io, .{ .reuse_address = true }) catch return error.ListenFailed;
    defer listener.deinit(io);

    std.log.info("ACP HTTP server listening on {s}", .{address});

    while (true) {
        var stream = listener.accept(io) catch |err| {
            std.log.err("ACP accept error: {t}", .{err});
            continue;
        };
        defer stream.close(io);
        handleHttpConnection(allocator, io, &stream, &acp_server, card) catch |err| {
            std.log.err("ACP connection error: {t}", .{err});
        };
    }
}

fn resolveHttpAddress(
    io: std.Io,
    allocator: std.mem.Allocator,
    address: []const u8,
) HttpError!std.Io.net.IpAddress {
    var host_port = net_utils.parseHostPort(allocator, address) catch
        return HttpError.InvalidAddress;
    defer host_port.deinit(allocator);
    return std.Io.net.IpAddress.resolve(io, host_port.host, host_port.port) catch
        return HttpError.InvalidAddress;
}

fn handleHttpConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    stream: *std.Io.net.Stream,
    acp_server: *Server,
    card: AgentCard,
) !void {
    var recv_buf: [8192]u8 = undefined;
    var send_buf: [8192]u8 = undefined;
    var reader = stream.reader(io, &recv_buf);
    var writer = stream.writer(io, &send_buf);
    var server: std.http.Server = .init(
        &reader.interface,
        &writer.interface,
    );

    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        dispatchHttpRequest(allocator, acp_server, card, &request) catch |err| {
            std.log.err("ACP request error: {t}", .{err});
            acpRespondJson(&request, "{\"error\":\"internal server error\"}", .internal_server_error) catch |response_err| {
                std.log.err("ACP: failed to send error response: {t}", .{response_err});
            };
        };
    }
}

fn dispatchHttpRequest(
    allocator: std.mem.Allocator,
    acp_server: *Server,
    card: AgentCard,
    request: *std.http.Server.Request,
) !void {
    const target = request.head.target;
    const path = acpSplitPath(target);

    if (std.mem.eql(u8, path, "/.well-known/agent.json")) {
        if (request.head.method != .GET) {
            return acpRespondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const body = try card.toJson(allocator);
        defer allocator.free(body);
        return acpRespondJson(request, body, .ok);
    }

    if (std.mem.startsWith(u8, path, "/tasks")) {
        return handleTasksHttpRoute(allocator, acp_server, request, path);
    }

    return acpRespondJson(request, "{\"error\":\"not found\"}", .not_found);
}

fn acpSplitPath(target: []const u8) []const u8 {
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
        return acpRespondJson(request, "{\"error\":\"not found\"}", .not_found);
    }
    if (std.mem.eql(u8, path, "/tasks/send")) {
        if (request.head.method != .POST) {
            return acpRespondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const body = acpReadRequestBody(allocator, request) catch |err| switch (err) {
            HttpError.RequestTooLarge => {
                return acpRespondJson(request, "{\"error\":\"payload too large\"}", .payload_too_large);
            },
            HttpError.ReadFailed => {
                return acpRespondJson(request, "{\"error\":\"invalid body\"}", .bad_request);
            },
            else => return err,
        };
        defer allocator.free(body);
        const message = acpExtractMessage(allocator, body) catch body;
        defer if (message.ptr != body.ptr) allocator.free(message);
        const id = try acp_server.createTask(message);
        var buf: [128]u8 = undefined;
        const json_body = std.fmt.bufPrint(&buf, "{{\"id\":\"{s}\"}}", .{id}) catch return error.OutOfMemory;
        return acpRespondJson(request, json_body, .created);
    }
    if (std.mem.startsWith(u8, path, "/tasks/")) {
        const id = path["/tasks/".len..];
        if (id.len == 0) return acpRespondJson(request, "{\"error\":\"not found\"}", .not_found);
        if (request.head.method != .GET) {
            return acpRespondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const task = acp_server.getTask(id) orelse {
            return acpRespondJson(request, "{\"error\":\"not found\"}", .not_found);
        };
        const body = try task.toJson(allocator);
        defer allocator.free(body);
        return acpRespondJson(request, body, .ok);
    }
    return acpRespondJson(request, "{\"error\":\"not found\"}", .not_found);
}

fn acpExtractMessage(allocator: std.mem.Allocator, body: []const u8) ![]const u8 {
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

fn acpReadRequestBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) HttpError![]u8 {
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

fn acpRespondJson(
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

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "AgentCard toJson" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test-agent",
        .description = "A test agent",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{},
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "test-agent") != null);
}

test "Server createTask" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("Hello, agent!");
    try std.testing.expectEqual(@as(u32, 1), server.taskCount());

    const task = server.getTask(id);
    try std.testing.expect(task != null);
    try std.testing.expectEqual(TaskStatus.submitted, task.?.status);
}

test "Task toJson" {
    const allocator = std.testing.allocator;
    var task = Task{
        .id = try allocator.dupe(u8, "task-1"),
        .status = .working,
        .messages = .empty,
    };
    defer task.deinit(allocator);

    const role = try allocator.dupe(u8, "user");
    const content = try allocator.dupe(u8, "test message");
    try task.messages.append(allocator, .{ .role = role, .content = content });

    const json = try task.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "task-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "working") != null);
}

test "TaskStatus toString" {
    try std.testing.expectEqualStrings("submitted", TaskStatus.submitted.toString());
    try std.testing.expectEqualStrings("completed", TaskStatus.completed.toString());
    try std.testing.expectEqualStrings("input-required", TaskStatus.input_required.toString());
}

test "AgentCard toJson escapes special characters" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test\"agent",
        .description = "line1\nline2\\end",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{ .streaming = true },
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);

    // Verify special chars are escaped
    try std.testing.expect(std.mem.indexOf(u8, json, "test\\\"agent") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "line1\\nline2\\\\end") != null);
    // Verify streaming capability
    try std.testing.expect(std.mem.indexOf(u8, json, "\"streaming\":true") != null);
}

test "appendEscaped handles all special chars" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try appendEscaped(allocator, &buf, "a\"b\\c\nd\re");
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd\\re", buf.items);
}

test "Server multiple tasks" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "multi",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    _ = try server.createTask("First task");
    _ = try server.createTask("Second task");
    try std.testing.expectEqual(@as(u32, 2), server.taskCount());
}

test "getTask returns null for unknown ID" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    try std.testing.expect(server.getTask("nonexistent") == null);
}

test "createTask assigns sequential IDs" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id1 = try server.createTask("First");
    const id2 = try server.createTask("Second");
    try std.testing.expectEqualStrings("task-1", id1);
    try std.testing.expectEqualStrings("task-2", id2);
}

test "Task message content preserved" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("Hello, agent!");
    const task = server.getTask(id).?;
    try std.testing.expectEqual(@as(usize, 1), task.messages.items.len);
    try std.testing.expectEqualStrings("user", task.messages.items[0].role);
    try std.testing.expectEqualStrings("Hello, agent!", task.messages.items[0].content);
}

test "appendEscaped handles control characters" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    // Test tab and control char below 0x20
    try appendEscaped(allocator, &buf, "a\tb\x01c");
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\t") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\u") != null);
}

test {
    std.testing.refAllDecls(@This());
}
