//! Session persistence for conversation memory.
//!
//! Provides save/load functionality for agent sessions including:
//! - JSON serialization of messages and configuration
//! - Session listing and metadata
//! - Path validation (no directory traversal)
//!
//! Session files are stored in ~/.abi/sessions/ by default.

const std = @import("std");
const time = @import("../../../shared/utils/time.zig");
const mod = @import("mod.zig");

const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryType = mod.MemoryType;

/// Session persistence errors
pub const PersistenceError = error{
    SessionNotFound,
    InvalidSessionData,
    PathTraversal,
    InvalidPath,
    SerializationFailed,
    DeserializationFailed,
    DiskFull,
    PermissionDenied,
    IoError,
    OutOfMemory,
};

/// Session metadata (lightweight info without full message content)
pub const SessionMeta = struct {
    id: []const u8,
    name: []const u8,
    created_at: i64,
    updated_at: i64,
    message_count: usize,
    total_tokens: usize,

    pub fn deinit(self: *SessionMeta, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        self.* = undefined;
    }
};

/// Full session data for save/load
pub const SessionData = struct {
    id: []const u8,
    name: []const u8,
    created_at: i64,
    updated_at: i64,
    messages: []Message,
    config: SessionConfig,

    pub fn deinit(self: *SessionData, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        for (self.messages) |*msg| {
            msg.deinit(allocator);
        }
        allocator.free(self.messages);
        self.* = undefined;
    }
};

/// Session configuration stored with session data
pub const SessionConfig = struct {
    memory_type: MemoryType = .sliding_window,
    max_tokens: usize = 4000,
    temperature: f32 = 0.7,
    model: []const u8 = "default",
    system_prompt: ?[]const u8 = null,
};

/// Session store for file-based persistence
pub const SessionStore = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,

    /// Initialize session store with base directory
    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) SessionStore {
        return .{
            .allocator = allocator,
            .base_dir = base_dir,
        };
    }

    /// Save session to file
    pub fn saveSession(
        self: *SessionStore,
        session: SessionData,
    ) PersistenceError!void {
        // Validate session ID (no path traversal)
        if (!isValidSessionId(session.id)) {
            return PersistenceError.PathTraversal;
        }

        // Build path
        const path = self.buildPath(session.id) catch return PersistenceError.OutOfMemory;
        defer self.allocator.free(path);

        // Serialize to JSON
        const json = serializeSession(self.allocator, session) catch return PersistenceError.SerializationFailed;
        defer self.allocator.free(json);

        // Write to file using Zig 0.16 I/O
        writeFile(self.allocator, path, json) catch return PersistenceError.IoError;
    }

    /// Load session from file
    pub fn loadSession(
        self: *SessionStore,
        id: []const u8,
    ) PersistenceError!SessionData {
        // Validate session ID
        if (!isValidSessionId(id)) {
            return PersistenceError.PathTraversal;
        }

        // Build path
        const path = self.buildPath(id) catch return PersistenceError.OutOfMemory;
        defer self.allocator.free(path);

        // Read file
        const content = readFile(self.allocator, path) catch return PersistenceError.SessionNotFound;
        defer self.allocator.free(content);

        // Deserialize from JSON
        return deserializeSession(self.allocator, content) catch return PersistenceError.InvalidSessionData;
    }

    /// Delete session file
    pub fn deleteSession(
        self: *SessionStore,
        id: []const u8,
    ) PersistenceError!void {
        // Validate session ID
        if (!isValidSessionId(id)) {
            return PersistenceError.PathTraversal;
        }

        const path = self.buildPath(id) catch return PersistenceError.OutOfMemory;
        defer self.allocator.free(path);

        deleteFile(self.allocator, path) catch return PersistenceError.IoError;
    }

    /// List all sessions in the store
    pub fn listSessions(
        self: *SessionStore,
    ) PersistenceError![]SessionMeta {
        var sessions: std.ArrayListUnmanaged(SessionMeta) = .empty;
        errdefer {
            for (sessions.items) |*s| s.deinit(self.allocator);
            sessions.deinit(self.allocator);
        }

        // List .json files in base_dir
        const files = listSessionFiles(self.allocator, self.base_dir) catch return PersistenceError.IoError;
        defer {
            for (files) |f| self.allocator.free(f);
            self.allocator.free(files);
        }

        for (files) |filename| {
            // Extract session ID from filename (remove .json)
            if (std.mem.endsWith(u8, filename, ".json")) {
                const id = filename[0 .. filename.len - 5];
                const session = self.loadSession(id) catch continue;
                defer {
                    var s = session;
                    // Don't free messages, just the session struct
                    self.allocator.free(s.id);
                    self.allocator.free(s.name);
                    for (s.messages) |*msg| msg.deinit(self.allocator);
                    self.allocator.free(s.messages);
                }

                sessions.append(self.allocator, .{
                    .id = self.allocator.dupe(u8, session.id) catch continue,
                    .name = self.allocator.dupe(u8, session.name) catch continue,
                    .created_at = session.created_at,
                    .updated_at = session.updated_at,
                    .message_count = session.messages.len,
                    .total_tokens = countTokens(session.messages),
                }) catch continue;
            }
        }

        return sessions.toOwnedSlice(self.allocator);
    }

    /// Check if session exists
    pub fn sessionExists(
        self: *SessionStore,
        id: []const u8,
    ) bool {
        if (!isValidSessionId(id)) return false;

        const path = self.buildPath(id) catch return false;
        defer self.allocator.free(path);

        return fileExists(self.allocator, path);
    }

    fn buildPath(self: *SessionStore, id: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}/{s}.json", .{ self.base_dir, id });
    }
};

// =============================================================================
// Serialization
// =============================================================================

fn serializeSession(allocator: std.mem.Allocator, session: SessionData) ![]u8 {
    var json: std.ArrayListUnmanaged(u8) = .empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\n");
    try json.appendSlice(allocator, "  \"id\": \"");
    try appendEscaped(allocator, &json, session.id);
    try json.appendSlice(allocator, "\",\n");

    try json.appendSlice(allocator, "  \"name\": \"");
    try appendEscaped(allocator, &json, session.name);
    try json.appendSlice(allocator, "\",\n");

    var buf: [64]u8 = undefined;
    const created_str = std.fmt.bufPrint(&buf, "{d}", .{session.created_at}) catch "0";
    try json.appendSlice(allocator, "  \"created_at\": ");
    try json.appendSlice(allocator, created_str);
    try json.appendSlice(allocator, ",\n");

    const updated_str = std.fmt.bufPrint(&buf, "{d}", .{session.updated_at}) catch "0";
    try json.appendSlice(allocator, "  \"updated_at\": ");
    try json.appendSlice(allocator, updated_str);
    try json.appendSlice(allocator, ",\n");

    // Config
    try json.appendSlice(allocator, "  \"config\": {\n");
    try json.appendSlice(allocator, "    \"memory_type\": \"");
    try json.appendSlice(allocator, memoryTypeToString(session.config.memory_type));
    try json.appendSlice(allocator, "\",\n");

    const max_tokens_str = std.fmt.bufPrint(&buf, "{d}", .{session.config.max_tokens}) catch "0";
    try json.appendSlice(allocator, "    \"max_tokens\": ");
    try json.appendSlice(allocator, max_tokens_str);
    try json.appendSlice(allocator, ",\n");

    const temp_str = std.fmt.bufPrint(&buf, "{d:.2}", .{session.config.temperature}) catch "0";
    try json.appendSlice(allocator, "    \"temperature\": ");
    try json.appendSlice(allocator, temp_str);
    try json.appendSlice(allocator, ",\n");

    try json.appendSlice(allocator, "    \"model\": \"");
    try appendEscaped(allocator, &json, session.config.model);
    try json.appendSlice(allocator, "\"");

    if (session.config.system_prompt) |prompt| {
        try json.appendSlice(allocator, ",\n    \"system_prompt\": \"");
        try appendEscaped(allocator, &json, prompt);
        try json.appendSlice(allocator, "\"");
    }
    try json.appendSlice(allocator, "\n  },\n");

    // Messages
    try json.appendSlice(allocator, "  \"messages\": [\n");
    for (session.messages, 0..) |msg, i| {
        if (i > 0) try json.appendSlice(allocator, ",\n");
        try json.appendSlice(allocator, "    {\n");
        try json.appendSlice(allocator, "      \"role\": \"");
        try json.appendSlice(allocator, msg.role.toString());
        try json.appendSlice(allocator, "\",\n");
        try json.appendSlice(allocator, "      \"content\": \"");
        try appendEscaped(allocator, &json, msg.content);
        try json.appendSlice(allocator, "\"");

        if (msg.timestamp != 0) {
            const ts_str = std.fmt.bufPrint(&buf, "{d}", .{msg.timestamp}) catch "0";
            try json.appendSlice(allocator, ",\n      \"timestamp\": ");
            try json.appendSlice(allocator, ts_str);
        }

        try json.appendSlice(allocator, "\n    }");
    }
    try json.appendSlice(allocator, "\n  ]\n");

    try json.appendSlice(allocator, "}\n");

    return json.toOwnedSlice(allocator);
}

fn deserializeSession(allocator: std.mem.Allocator, json_content: []const u8) !SessionData {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_content,
        .{},
    ) catch return error.InvalidSessionData;
    defer parsed.deinit();

    const root = parsed.value.object;

    // Extract required fields
    const id = root.get("id") orelse return error.InvalidSessionData;
    const name = root.get("name") orelse return error.InvalidSessionData;
    const created_at = root.get("created_at") orelse return error.InvalidSessionData;
    const updated_at = root.get("updated_at") orelse return error.InvalidSessionData;
    const messages_val = root.get("messages") orelse return error.InvalidSessionData;

    // Parse config
    var config = SessionConfig{};
    if (root.get("config")) |config_val| {
        if (config_val.object.get("memory_type")) |mt| {
            config.memory_type = stringToMemoryType(mt.string);
        }
        if (config_val.object.get("max_tokens")) |mt| {
            config.max_tokens = @intCast(mt.integer);
        }
        if (config_val.object.get("temperature")) |t| {
            config.temperature = @floatCast(t.float);
        }
        if (config_val.object.get("model")) |m| {
            config.model = try allocator.dupe(u8, m.string);
        }
        if (config_val.object.get("system_prompt")) |sp| {
            config.system_prompt = try allocator.dupe(u8, sp.string);
        }
    }

    // Parse messages
    var messages: std.ArrayListUnmanaged(Message) = .empty;
    errdefer {
        for (messages.items) |*m| m.deinit(allocator);
        messages.deinit(allocator);
    }

    for (messages_val.array.items) |msg_val| {
        const role_str = msg_val.object.get("role") orelse continue;
        const content = msg_val.object.get("content") orelse continue;

        var msg = Message{
            .role = stringToRole(role_str.string),
            .content = try allocator.dupe(u8, content.string),
            .timestamp = 0,
        };

        if (msg_val.object.get("timestamp")) |ts| {
            msg.timestamp = ts.integer;
        }

        try messages.append(allocator, msg);
    }

    return .{
        .id = try allocator.dupe(u8, id.string),
        .name = try allocator.dupe(u8, name.string),
        .created_at = created_at.integer,
        .updated_at = updated_at.integer,
        .messages = try messages.toOwnedSlice(allocator),
        .config = config,
    };
}

fn appendEscaped(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(u8), str: []const u8) !void {
    for (str) |c| {
        switch (c) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => try list.append(allocator, c),
        }
    }
}

fn memoryTypeToString(mt: MemoryType) []const u8 {
    return switch (mt) {
        .short_term => "short_term",
        .sliding_window => "sliding_window",
        .summarizing => "summarizing",
        .long_term => "long_term",
        .hybrid => "hybrid",
    };
}

fn stringToMemoryType(str: []const u8) MemoryType {
    if (std.mem.eql(u8, str, "short_term")) return .short_term;
    if (std.mem.eql(u8, str, "sliding_window")) return .sliding_window;
    if (std.mem.eql(u8, str, "summarizing")) return .summarizing;
    if (std.mem.eql(u8, str, "long_term")) return .long_term;
    if (std.mem.eql(u8, str, "hybrid")) return .hybrid;
    return .sliding_window;
}

fn stringToRole(str: []const u8) MessageRole {
    if (std.mem.eql(u8, str, "system")) return .system;
    if (std.mem.eql(u8, str, "user")) return .user;
    if (std.mem.eql(u8, str, "assistant")) return .assistant;
    if (std.mem.eql(u8, str, "tool")) return .tool;
    return .user;
}

fn countTokens(messages: []const Message) usize {
    var total: usize = 0;
    for (messages) |msg| {
        total += msg.estimateTokens();
    }
    return total;
}

// =============================================================================
// Validation
// =============================================================================

/// Validate session ID to prevent path traversal
fn isValidSessionId(id: []const u8) bool {
    if (id.len == 0 or id.len > 128) return false;

    // Disallow .. path traversal
    if (std.mem.indexOf(u8, id, "..") != null) return false;

    // Disallow absolute paths (/, \, C:)
    if (id[0] == '/' or id[0] == '\\') return false;
    if (id.len >= 2 and id[1] == ':') return false;

    // Only allow alphanumeric, dash, underscore
    for (id) |c| {
        const valid = (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '-' or c == '_';
        if (!valid) return false;
    }

    return true;
}

// =============================================================================
// File I/O (Zig 0.16 compatible)
// =============================================================================

fn writeFile(allocator: std.mem.Allocator, path: []const u8, content: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Ensure directory exists (by attempting to create parent)
    // Note: In production, would need proper directory creation

    const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
        std.log.err("Failed to create file {s}: {t}", .{ path, err });
        return err;
    };
    defer file.close(io);

    var write_buf: [8192]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    _ = writer.interface.write(content) catch |err| {
        std.log.err("Failed to write to file: {t}", .{err});
        return err;
    };
    writer.flush() catch |err| {
        std.log.err("Failed to flush file: {t}", .{err});
        return err;
    };
}

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024)) catch |err| {
        return err;
    };

    return content;
}

fn deleteFile(allocator: std.mem.Allocator, path: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().deleteFile(io, path) catch |err| {
        std.log.warn("Failed to delete file {s}: {t}", .{ path, err });
        return err;
    };
}

fn fileExists(allocator: std.mem.Allocator, path: []const u8) bool {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Try to open the file - if it succeeds, it exists
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn listSessionFiles(allocator: std.mem.Allocator, dir_path: []const u8) ![][]const u8 {
    var files: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer {
        for (files.items) |f| allocator.free(f);
        files.deinit(allocator);
    }

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch {
        // Directory doesn't exist, return empty list
        return files.toOwnedSlice(allocator);
    };
    defer dir.close(io);

    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".json")) {
                const name_copy = try allocator.dupe(u8, entry.name);
                try files.append(allocator, name_copy);
            }
        } else break;
    }

    return files.toOwnedSlice(allocator);
}

// =============================================================================
// Public API
// =============================================================================

/// Create a new session with the given name
pub fn createSession(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: SessionConfig,
) !SessionData {
    // Generate ID from name + timestamp
    const timestamp = time.unixSeconds();
    const id = try std.fmt.allocPrint(allocator, "{s}-{d}", .{ name, timestamp });

    return .{
        .id = id,
        .name = try allocator.dupe(u8, name),
        .created_at = timestamp,
        .updated_at = timestamp,
        .messages = &[_]Message{},
        .config = config,
    };
}

/// Get default session directory path
pub fn getDefaultSessionDir(allocator: std.mem.Allocator) ![]u8 {
    // Use ~/.abi/sessions on Unix, %APPDATA%\abi\sessions on Windows
    // For simplicity, use .abi/sessions relative to current directory
    return try allocator.dupe(u8, ".abi/sessions");
}

// =============================================================================
// Tests
// =============================================================================

test "session id validation" {
    try std.testing.expect(isValidSessionId("my-session"));
    try std.testing.expect(isValidSessionId("session_123"));
    try std.testing.expect(isValidSessionId("MySession"));

    try std.testing.expect(!isValidSessionId(""));
    try std.testing.expect(!isValidSessionId("../etc/passwd"));
    try std.testing.expect(!isValidSessionId("/root/session"));
    try std.testing.expect(!isValidSessionId("C:\\Windows"));
    try std.testing.expect(!isValidSessionId("session with spaces"));
    try std.testing.expect(!isValidSessionId("session.json"));
}

test "serialize and deserialize session" {
    const allocator = std.testing.allocator;

    const original = SessionData{
        .id = "test-session-123",
        .name = "Test Session",
        .created_at = 1704067200,
        .updated_at = 1704067300,
        .messages = &[_]Message{
            .{ .role = .user, .content = "Hello!", .timestamp = 1704067200 },
            .{ .role = .assistant, .content = "Hi there!", .timestamp = 1704067250 },
        },
        .config = .{
            .memory_type = .sliding_window,
            .max_tokens = 4000,
            .temperature = 0.7,
            .model = "gpt-4",
            .system_prompt = "You are helpful.",
        },
    };

    const json = try serializeSession(allocator, original);
    defer allocator.free(json);

    var loaded = try deserializeSession(allocator, json);
    defer loaded.deinit(allocator);

    try std.testing.expectEqualStrings(original.id, loaded.id);
    try std.testing.expectEqualStrings(original.name, loaded.name);
    try std.testing.expectEqual(original.created_at, loaded.created_at);
    try std.testing.expectEqual(original.messages.len, loaded.messages.len);
}

test "memory type conversion" {
    try std.testing.expectEqual(MemoryType.hybrid, stringToMemoryType("hybrid"));
    try std.testing.expectEqual(MemoryType.sliding_window, stringToMemoryType("sliding_window"));
    try std.testing.expectEqual(MemoryType.sliding_window, stringToMemoryType("unknown"));

    try std.testing.expectEqualStrings("hybrid", memoryTypeToString(.hybrid));
    try std.testing.expectEqualStrings("short_term", memoryTypeToString(.short_term));
}

