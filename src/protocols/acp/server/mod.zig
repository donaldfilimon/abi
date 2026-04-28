//! ACP (Agent Communication Protocol) Server
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
const net_utils = @import("../../../foundation/mod.zig").utils.net;
const foundation_time = @import("../../../foundation/time.zig");
const parity_gate = @import("../../../common/parity_gate.zig");

pub const agent_card = @import("agent_card.zig");
pub const tasks = @import("tasks.zig");
pub const sessions = @import("sessions.zig");
pub const routing = @import("routing.zig");
pub const json_utils = @import("json_utils.zig");
pub const discord_routes = @import("discord_routes.zig");
pub const openapi = @import("openapi.zig");

pub const AgentCard = agent_card.AgentCard;
pub const TaskStatus = tasks.TaskStatus;
pub const Task = tasks.Task;
pub const TransitionError = tasks.TransitionError;
pub const Session = sessions.Session;
pub const HttpError = routing.HttpError;

/// ACP Server that manages tasks and sessions
pub const Server = struct {
    allocator: std.mem.Allocator,
    card: AgentCard,
    tasks_map: std.StringHashMapUnmanaged(Task),
    sessions_map: std.StringHashMapUnmanaged(Session),
    next_task_id: u64,
    next_session_id: u64,
    openapi_spec: ?[]u8 = null,

    pub fn init(allocator: std.mem.Allocator, card: AgentCard) Server {
        return .{
            .allocator = allocator,
            .card = card,
            .tasks_map = .empty,
            .sessions_map = .empty,
            .next_task_id = 1,
            .next_session_id = 1,
        };
    }

    /// Lazily generate and cache the OpenAPI spec.
    pub fn getOrBuildOpenApiSpec(self: *Server) ![]const u8 {
        if (self.openapi_spec) |spec| return spec;
        self.openapi_spec = try openapi.generate(self.allocator, self.card);
        return self.openapi_spec.?;
    }

    pub fn deinit(self: *Server) void {
        var task_it = self.tasks_map.iterator();
        while (task_it.next()) |entry| {
            var task = entry.value_ptr;
            task.deinit(self.allocator);
        }
        self.tasks_map.deinit(self.allocator);

        var session_it = self.sessions_map.iterator();
        while (session_it.next()) |entry| {
            var session = entry.value_ptr;
            session.deinit(self.allocator);
        }
        self.sessions_map.deinit(self.allocator);
        if (self.openapi_spec) |spec| self.allocator.free(spec);
    }

    /// Create a new task from a message
    pub fn createTask(self: *Server, message: []const u8) ![]const u8 {
        var id_buf: [32]u8 = undefined;
        const id_str = std.fmt.bufPrint(&id_buf, "task-{d}", .{self.next_task_id}) catch
            return error.OutOfMemory;
        self.next_task_id += 1;

        const id = try self.allocator.dupe(u8, id_str);

        const now = foundation_time.unixMs();
        var task = Task{
            .id = id,
            .status = .submitted,
            .messages = .empty,
            .created_at_ms = now,
            .updated_at_ms = now,
            .history = .empty,
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

        try self.tasks_map.put(self.allocator, id, task);
        return id;
    }

    /// Get a task by ID
    pub fn getTask(self: *Server, id: []const u8) ?*Task {
        return self.tasks_map.getPtr(id);
    }

    /// Update a task's status, enforcing valid state transitions.
    /// Returns error.TaskNotFound if the ID doesn't exist, or
    /// error.InvalidTransition if the transition is not allowed.
    pub fn updateTaskStatus(self: *Server, id: []const u8, new_status: TaskStatus) (TransitionError || error{TaskNotFound})!void {
        const task = self.tasks_map.getPtr(id) orelse return error.TaskNotFound;
        try task.transitionTo(self.allocator, new_status);
    }

    /// Get the number of tasks
    pub fn taskCount(self: *const Server) u32 {
        return self.tasks_map.count();
    }

    /// Create a new session
    pub fn createSession(self: *Server, metadata: ?[]const u8) ![]const u8 {
        var id_buf: [32]u8 = undefined;
        const id_str = std.fmt.bufPrint(&id_buf, "session-{d}", .{self.next_session_id}) catch "session-0";
        self.next_session_id += 1;

        const id = try self.allocator.dupe(u8, id_str);
        var session = Session{
            .id = id,
            .created_at = 0,
            .metadata = null,
            .task_ids = .empty,
        };
        errdefer session.deinit(self.allocator);

        if (metadata) |m| {
            session.metadata = try self.allocator.dupe(u8, m);
        }

        try self.sessions_map.put(self.allocator, id, session);
        return id;
    }

    /// Get a session by ID
    pub fn getSession(self: *Server, id: []const u8) ?*Session {
        return self.sessions_map.getPtr(id);
    }

    /// Get the number of sessions
    pub fn sessionCount(self: *const Server) u32 {
        return self.sessions_map.count();
    }

    /// Return a JSON status summary (task/session counts, version).
    pub fn statusJson(self: *const Server, allocator: std.mem.Allocator) ![]const u8 {
        const build_options = @import("build_options");
        var buf: [512]u8 = undefined;
        const json = std.fmt.bufPrint(&buf,
            \\{{"status":"ok","version":"{s}","tasks":{d},"sessions":{d}}}
        , .{
            build_options.package_version,
            self.tasks_map.count(),
            self.sessions_map.count(),
        }) catch return error.OutOfMemory;
        return allocator.dupe(u8, json);
    }

    /// Add a task to a session
    pub fn addTaskToSession(self: *Server, session_id: []const u8, task_id: []const u8) !void {
        const session = self.sessions_map.getPtr(session_id) orelse return error.SessionNotFound;
        const owned_task_id = try self.allocator.dupe(u8, task_id);
        errdefer self.allocator.free(owned_task_id);
        try session.task_ids.append(self.allocator, owned_task_id);
    }
};

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
        routing.dispatchHttpRequest(allocator, acp_server, card, &request) catch |err| {
            std.log.err("ACP request error: {t}", .{err});
            routing.respondJson(&request, "{\"error\":\"internal server error\"}", .internal_server_error) catch |response_err| {
                std.log.err("ACP: failed to send error response: {t}", .{response_err});
            };
        };
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

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

test "Server createSession" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createSession(null);
    try std.testing.expectEqual(@as(u32, 1), server.sessionCount());

    const session = server.getSession(id);
    try std.testing.expect(session != null);
    try std.testing.expectEqualStrings(id, session.?.id);
}

test "Server addTaskToSession" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const session_id = try server.createSession(null);
    const task_id = try server.createTask("hello");

    try server.addTaskToSession(session_id, task_id);

    const session = server.getSession(session_id).?;
    try std.testing.expectEqual(@as(usize, 1), session.task_ids.items.len);
    try std.testing.expectEqualStrings(task_id, session.task_ids.items[0]);
}

test "Server updateTaskStatus valid transition" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("transition test");
    try server.updateTaskStatus(id, .working);
    try std.testing.expectEqual(TaskStatus.working, server.getTask(id).?.status);

    try server.updateTaskStatus(id, .completed);
    try std.testing.expectEqual(TaskStatus.completed, server.getTask(id).?.status);
}

test "Server updateTaskStatus invalid transition" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("transition test");
    // submitted -> completed is not valid
    try std.testing.expectError(error.InvalidTransition, server.updateTaskStatus(id, .completed));
    // Status should remain submitted
    try std.testing.expectEqual(TaskStatus.submitted, server.getTask(id).?.status);
}

test "Server updateTaskStatus task not found" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    try std.testing.expectError(error.TaskNotFound, server.updateTaskStatus("nonexistent", .working));
}

test {
    if (!parity_gate.canRunTest()) return;
    std.testing.refAllDecls(@This());
}
