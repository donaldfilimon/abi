//! Integration Tests: ACP (Agent Communication Protocol)
//!
//! Verifies the ACP module's public API from a consumer perspective:
//! Server lifecycle, AgentCard creation and serialization, Task management,
//! Session management, TaskStatus transitions, error handling, and
//! type availability.

const std = @import("std");
const abi = @import("abi");

const acp = abi.acp;

// ============================================================================
// Type availability
// ============================================================================

test "acp: Server type is available" {
    _ = acp.Server;
}

test "acp: AgentCard type is available" {
    _ = acp.AgentCard;
}

test "acp: Task type is available" {
    _ = acp.Task;
}

test "acp: Session type is available" {
    _ = acp.Session;
}

test "acp: HttpError type is available" {
    _ = acp.HttpError;
}

test "acp: TaskStatus type is available" {
    _ = acp.TaskStatus;
}

test "acp: AgentCard.Capabilities type is available" {
    _ = acp.AgentCard.Capabilities;
}

test "acp: Task.Message type is available" {
    _ = acp.Task.Message;
}

// ============================================================================
// AgentCard creation and field access
// ============================================================================

test "acp: AgentCard creation with default capabilities" {
    const card = acp.AgentCard{
        .name = "test-agent",
        .description = "An integration test agent",
        .version = "1.0.0",
        .url = "http://localhost:8080",
        .capabilities = .{},
    };
    try std.testing.expectEqualStrings("test-agent", card.name);
    try std.testing.expectEqualStrings("An integration test agent", card.description);
    try std.testing.expectEqualStrings("1.0.0", card.version);
    try std.testing.expectEqualStrings("http://localhost:8080", card.url);
    try std.testing.expect(!card.capabilities.streaming);
    try std.testing.expect(!card.capabilities.pushNotifications);
    try std.testing.expect(!card.capabilities.stateTransitionHistory);
    try std.testing.expect(!card.capabilities.extensions);
}

test "acp: AgentCard creation with all capabilities enabled" {
    const card = acp.AgentCard{
        .name = "full-agent",
        .description = "Full capabilities",
        .version = "2.0.0",
        .url = "http://example.com",
        .capabilities = .{
            .streaming = true,
            .pushNotifications = true,
            .stateTransitionHistory = true,
            .extensions = true,
        },
    };
    try std.testing.expect(card.capabilities.streaming);
    try std.testing.expect(card.capabilities.pushNotifications);
    try std.testing.expect(card.capabilities.stateTransitionHistory);
    try std.testing.expect(card.capabilities.extensions);
}

test "acp: AgentCard toJson produces valid JSON" {
    const allocator = std.testing.allocator;
    const card = acp.AgentCard{
        .name = "json-agent",
        .description = "JSON test",
        .version = "0.1.0",
        .url = "http://localhost:9090",
        .capabilities = .{ .streaming = true },
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "json-agent") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"streaming\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"pushNotifications\":false") != null);
}

test "acp: AgentCard toJson escapes special characters" {
    const allocator = std.testing.allocator;
    const card = acp.AgentCard{
        .name = "agent\"with\"quotes",
        .description = "line1\nline2\\end",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "agent\\\"with\\\"quotes") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "line1\\nline2\\\\end") != null);
}

// ============================================================================
// TaskStatus
// ============================================================================

test "acp: TaskStatus toString covers all variants" {
    try std.testing.expectEqualStrings("submitted", acp.TaskStatus.submitted.toString());
    try std.testing.expectEqualStrings("working", acp.TaskStatus.working.toString());
    try std.testing.expectEqualStrings("input-required", acp.TaskStatus.input_required.toString());
    try std.testing.expectEqualStrings("completed", acp.TaskStatus.completed.toString());
    try std.testing.expectEqualStrings("failed", acp.TaskStatus.failed.toString());
    try std.testing.expectEqualStrings("canceled", acp.TaskStatus.canceled.toString());
}

test "acp: TaskStatus enum has exactly 6 variants" {
    const fields = @typeInfo(acp.TaskStatus).@"enum".fields;
    try std.testing.expectEqual(@as(usize, 6), fields.len);
}

// ============================================================================
// Server init/deinit lifecycle
// ============================================================================

test "acp: Server init and deinit" {
    const allocator = std.testing.allocator;
    const card = acp.AgentCard{
        .name = "lifecycle-agent",
        .description = "Lifecycle test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    var server = acp.Server.init(allocator, card);
    defer server.deinit();

    try std.testing.expectEqual(@as(u32, 0), server.taskCount());
    try std.testing.expectEqual(@as(u32, 0), server.sessionCount());
}

test "acp: Server preserves agent card" {
    const allocator = std.testing.allocator;
    const card = acp.AgentCard{
        .name = "card-check",
        .description = "Card preservation test",
        .version = "3.0.0",
        .url = "http://example.com:1234",
        .capabilities = .{ .extensions = true },
    };
    var server = acp.Server.init(allocator, card);
    defer server.deinit();

    try std.testing.expectEqualStrings("card-check", server.card.name);
    try std.testing.expectEqualStrings("3.0.0", server.card.version);
    try std.testing.expect(server.card.capabilities.extensions);
}

// ============================================================================
// Task creation and management
// ============================================================================

test "acp: Server createTask returns task ID" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("Hello, agent!");
    try std.testing.expect(id.len > 0);
    try std.testing.expectEqual(@as(u32, 1), server.taskCount());
}

test "acp: Server createTask assigns sequential IDs" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
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

test "acp: Server getTask retrieves created task" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("Test message");
    const task = server.getTask(id);
    try std.testing.expect(task != null);
    try std.testing.expectEqual(acp.TaskStatus.submitted, task.?.status);
}

test "acp: Server getTask returns null for unknown ID" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    try std.testing.expect(server.getTask("nonexistent") == null);
}

test "acp: Task message content is preserved" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
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

test "acp: Server multiple tasks increment count" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "multi",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    _ = try server.createTask("Task A");
    _ = try server.createTask("Task B");
    _ = try server.createTask("Task C");
    try std.testing.expectEqual(@as(u32, 3), server.taskCount());
}

test "acp: Task toJson includes id and status" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createTask("Serialize me");
    const task = server.getTask(id).?;
    const json = try task.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "task-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "submitted") != null);
}

// ============================================================================
// Session management
// ============================================================================

test "acp: Server createSession returns session ID" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createSession(null);
    try std.testing.expect(id.len > 0);
    try std.testing.expectEqual(@as(u32, 1), server.sessionCount());
}

test "acp: Server createSession with metadata" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createSession("user-context-data");
    const session = server.getSession(id);
    try std.testing.expect(session != null);
    try std.testing.expect(session.?.metadata != null);
    try std.testing.expectEqualStrings("user-context-data", session.?.metadata.?);
}

test "acp: Server createSession without metadata" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const id = try server.createSession(null);
    const session = server.getSession(id);
    try std.testing.expect(session != null);
    try std.testing.expect(session.?.metadata == null);
}

test "acp: Server getSession returns null for unknown ID" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    try std.testing.expect(server.getSession("unknown") == null);
}

test "acp: Server addTaskToSession links task to session" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
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

test "acp: Server addTaskToSession with unknown session returns error" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const task_id = try server.createTask("hello");
    try std.testing.expectError(error.SessionNotFound, server.addTaskToSession("no-such-session", task_id));
}

test "acp: Session toJson includes session fields" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    const session_id = try server.createSession("meta-info");
    const session = server.getSession(session_id).?;
    const json = try session.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "session-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "meta-info") != null);
}

test "acp: Server multiple sessions increment count" {
    const allocator = std.testing.allocator;
    var server = acp.Server.init(allocator, .{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "http://localhost",
        .capabilities = .{},
    });
    defer server.deinit();

    _ = try server.createSession(null);
    _ = try server.createSession("s2");
    _ = try server.createSession("s3");
    try std.testing.expectEqual(@as(u32, 3), server.sessionCount());
}

// ============================================================================
// Error types and error set completeness
// ============================================================================

test "acp: HttpError includes expected error variants" {
    // Verify HttpError is a valid error set by checking it contains expected variants
    const err_info = @typeInfo(acp.HttpError);
    try std.testing.expect(err_info == .error_set);
    // HttpError should include OutOfMemory, InvalidAddress, ListenFailed, ReadFailed, RequestTooLarge
    const fields = err_info.error_set.?;
    try std.testing.expect(fields.len >= 4);
}

// ============================================================================
// serveHttp function availability
// ============================================================================

test "acp: serveHttp function is available" {
    // Verify the function exists and has the expected signature
    const T = @TypeOf(acp.serveHttp);
    _ = T;
}

// ============================================================================
// Full lifecycle: create server, tasks, sessions, link them
// ============================================================================

test "acp: full lifecycle -- server, tasks, sessions, linking" {
    const allocator = std.testing.allocator;

    // 1. Create server
    var server = acp.Server.init(allocator, .{
        .name = "lifecycle-agent",
        .description = "Full lifecycle integration test",
        .version = "1.0.0",
        .url = "http://localhost:8080",
        .capabilities = .{ .streaming = true, .stateTransitionHistory = true },
    });
    defer server.deinit();

    // 2. Create a session
    const session_id = try server.createSession("integration-test");
    try std.testing.expectEqual(@as(u32, 1), server.sessionCount());

    // 3. Create tasks
    const task1_id = try server.createTask("What is the meaning of life?");
    const task2_id = try server.createTask("Follow-up question");
    try std.testing.expectEqual(@as(u32, 2), server.taskCount());

    // 4. Link tasks to session
    try server.addTaskToSession(session_id, task1_id);
    try server.addTaskToSession(session_id, task2_id);

    // 5. Verify session contains both tasks
    const session = server.getSession(session_id).?;
    try std.testing.expectEqual(@as(usize, 2), session.task_ids.items.len);

    // 6. Verify tasks are retrievable
    const task1 = server.getTask(task1_id).?;
    try std.testing.expectEqual(acp.TaskStatus.submitted, task1.status);
    try std.testing.expectEqualStrings("What is the meaning of life?", task1.messages.items[0].content);

    // 7. Serialize agent card
    const json = try server.card.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "lifecycle-agent") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"streaming\":true") != null);
}

// Sibling test modules (pulled in via refAllDecls)
const _openapi = @import("acp_openapi_test.zig");

test {
    std.testing.refAllDecls(@This());
}
