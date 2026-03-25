//! ACP (Agent Communication Protocol) stub.
//!
//! Mirrors the full API of mod.zig / server.zig, returning error.FeatureDisabled
//! for all operations that would perform real work.

const std = @import("std");

// =============================================================================
// Sub-module namespace (empty — types re-exported at top level)
// =============================================================================

pub const server = struct {};

// =============================================================================
// Error set
// =============================================================================

const AcpError = error{
    FeatureDisabled,
    SessionNotFound,
    TaskNotFound,
    InvalidTransition,
    OutOfMemory,
};

pub const TransitionError = error{InvalidTransition};

// =============================================================================
// AgentCard
// =============================================================================

/// ACP Agent Card — describes this agent's capabilities.
pub const AgentCard = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8,
    url: []const u8,
    capabilities: Capabilities,

    pub const Capabilities = struct {
        streaming: bool = false,
        pushNotifications: bool = false,
        stateTransitionHistory: bool = false,
        extensions: bool = false,
    };

    /// Stub: always returns error.FeatureDisabled.
    pub fn toJson(self: AgentCard, allocator: std.mem.Allocator) AcpError![]u8 {
        _ = self;
        _ = allocator;
        return AcpError.FeatureDisabled;
    }
};

// =============================================================================
// TaskStatus
// =============================================================================

/// Task status in the ACP lifecycle.
pub const TaskStatus = enum {
    submitted,
    working,
    input_required,
    completed,
    failed,
    canceled,

    pub fn canTransitionTo(self: TaskStatus, target: TaskStatus) bool {
        if (target == .canceled) return true;
        return switch (self) {
            .submitted => target == .working,
            .working => target == .completed or target == .failed or target == .input_required,
            .input_required => target == .working,
            .completed, .failed, .canceled => false,
        };
    }

    pub fn transition(self: TaskStatus, target: TaskStatus) TransitionError!TaskStatus {
        if (self.canTransitionTo(target)) return target;
        return error.InvalidTransition;
    }

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

// =============================================================================
// Task
// =============================================================================

/// ACP Task stub.
pub const Task = struct {
    id: []const u8,
    status: TaskStatus,
    messages: std.ArrayListUnmanaged(Message),

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn deinit(self: *Task, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn toJson(self: *const Task, allocator: std.mem.Allocator) AcpError![]u8 {
        _ = self;
        _ = allocator;
        return AcpError.FeatureDisabled;
    }
};

// =============================================================================
// Session
// =============================================================================

/// ACP Session stub — groups related tasks together.
pub const Session = struct {
    id: []const u8,
    created_at: i64,
    metadata: ?[]const u8,
    task_ids: std.ArrayListUnmanaged([]const u8),

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn toJson(self: *const Session, allocator: std.mem.Allocator) AcpError![]u8 {
        _ = self;
        _ = allocator;
        return AcpError.FeatureDisabled;
    }
};

// =============================================================================
// HttpError
// =============================================================================

pub const HttpError = std.mem.Allocator.Error || error{
    InvalidAddress,
    ListenFailed,
    ReadFailed,
    RequestTooLarge,
    FeatureDisabled,
};

// =============================================================================
// Server
// =============================================================================

/// ACP Server stub.
pub const Server = struct {
    allocator: std.mem.Allocator,
    card: AgentCard,

    pub fn init(allocator: std.mem.Allocator, card: AgentCard) Server {
        return .{
            .allocator = allocator,
            .card = card,
        };
    }

    pub fn deinit(self: *Server) void {
        _ = self;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn createTask(self: *Server, message: []const u8) AcpError![]const u8 {
        _ = self;
        _ = message;
        return AcpError.FeatureDisabled;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn updateTaskStatus(self: *Server, id: []const u8, new_status: TaskStatus) AcpError!void {
        _ = self;
        _ = id;
        _ = new_status;
        return AcpError.FeatureDisabled;
    }

    /// Stub: always returns null.
    pub fn getTask(self: *Server, id: []const u8) ?*Task {
        _ = self;
        _ = id;
        return null;
    }

    /// Stub: always returns 0.
    pub fn taskCount(self: *const Server) u32 {
        _ = self;
        return 0;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn createSession(self: *Server, metadata: ?[]const u8) AcpError![]const u8 {
        _ = self;
        _ = metadata;
        return AcpError.FeatureDisabled;
    }

    /// Stub: always returns null.
    pub fn getSession(self: *Server, id: []const u8) ?*Session {
        _ = self;
        _ = id;
        return null;
    }

    /// Stub: always returns 0.
    pub fn sessionCount(self: *const Server) u32 {
        _ = self;
        return 0;
    }

    /// Stub: always returns error.FeatureDisabled.
    pub fn addTaskToSession(self: *Server, session_id: []const u8, task_id: []const u8) AcpError!void {
        _ = self;
        _ = session_id;
        _ = task_id;
        return AcpError.FeatureDisabled;
    }
};

// =============================================================================
// serveHttp
// =============================================================================

/// Stub: always returns error.FeatureDisabled.
pub fn serveHttp(
    allocator: std.mem.Allocator,
    io: std.Io,
    address: []const u8,
    card: AgentCard,
) HttpError!void {
    _ = allocator;
    _ = io;
    _ = address;
    _ = card;
    return HttpError.FeatureDisabled;
}

// =============================================================================
// Module lifecycle
// =============================================================================

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

// =============================================================================
// Tests
// =============================================================================

test "Server stub init and deinit" {
    const card = AgentCard{
        .name = "stub-agent",
        .description = "stub",
        .version = "0.0.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    var srv = Server.init(std.testing.allocator, card);
    defer srv.deinit();

    try std.testing.expectEqual(@as(u32, 0), srv.taskCount());
    try std.testing.expectEqual(@as(u32, 0), srv.sessionCount());
    try std.testing.expect(srv.getTask("any") == null);
    try std.testing.expect(srv.getSession("any") == null);
}

test "Server stub createTask returns FeatureDisabled" {
    const card = AgentCard{
        .name = "stub-agent",
        .description = "stub",
        .version = "0.0.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    var srv = Server.init(std.testing.allocator, card);
    defer srv.deinit();

    try std.testing.expectError(AcpError.FeatureDisabled, srv.createTask("hello"));
}

test "Server stub createSession returns FeatureDisabled" {
    const card = AgentCard{
        .name = "stub-agent",
        .description = "stub",
        .version = "0.0.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    var srv = Server.init(std.testing.allocator, card);
    defer srv.deinit();

    try std.testing.expectError(AcpError.FeatureDisabled, srv.createSession(null));
}

test "Server stub addTaskToSession returns FeatureDisabled" {
    const card = AgentCard{
        .name = "stub-agent",
        .description = "stub",
        .version = "0.0.0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    var srv = Server.init(std.testing.allocator, card);
    defer srv.deinit();

    try std.testing.expectError(AcpError.FeatureDisabled, srv.addTaskToSession("s1", "t1"));
}

test "TaskStatus toString stub" {
    try std.testing.expectEqualStrings("submitted", TaskStatus.submitted.toString());
    try std.testing.expectEqualStrings("completed", TaskStatus.completed.toString());
    try std.testing.expectEqualStrings("input-required", TaskStatus.input_required.toString());
}

test "AgentCard toJson stub returns FeatureDisabled" {
    const card = AgentCard{
        .name = "x",
        .description = "y",
        .version = "0",
        .url = "http://localhost",
        .capabilities = .{},
    };
    try std.testing.expectError(AcpError.FeatureDisabled, card.toJson(std.testing.allocator));
}

test "isEnabled returns false" {
    try std.testing.expect(!isEnabled());
}

test "isInitialized returns false" {
    try std.testing.expect(!isInitialized());
}

test {
    std.testing.refAllDecls(@This());
}
