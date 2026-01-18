//! Stub implementation for chat memory when AI features are disabled.

const std = @import("std");
const stub_root = @This();

pub const persistence = struct {
    pub const SessionStore = stub_root.SessionStore;
    pub const SessionData = stub_root.SessionData;
    pub const SessionMeta = stub_root.SessionMeta;
    pub const SessionConfig = stub_root.SessionConfig;
    pub const PersistenceError = stub_root.PersistenceError;
};

/// Stub message role.
pub const MessageRole = enum {
    system,
    user,
    assistant,
    tool,

    pub fn toString(self: MessageRole) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

/// Stub message.
pub const Message = struct {
    role: MessageRole = .user,
    content: []const u8 = "",
    name: ?[]const u8 = null,
    timestamp: i64 = 0,
    token_count: usize = 0,
    metadata: ?[]const u8 = null,

    pub fn user(content: []const u8) Message {
        return .{ .role = .user, .content = content };
    }

    pub fn assistant(content: []const u8) Message {
        return .{ .role = .assistant, .content = content };
    }

    pub fn system(content: []const u8) Message {
        return .{ .role = .system, .content = content };
    }

    pub fn tool(name: []const u8, content: []const u8) Message {
        return .{ .role = .tool, .content = content, .name = name };
    }

    pub fn estimateTokens(self: Message) usize {
        return (self.content.len + 3) / 4;
    }

    pub fn clone(self: Message, allocator: std.mem.Allocator) !Message {
        _ = allocator;
        return self;
    }

    pub fn deinit(self: *Message, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub memory type.
pub const MemoryType = enum {
    short_term,
    sliding_window,
    summarizing,
    long_term,
    hybrid,
};

/// Stub memory statistics.
pub const MemoryStats = struct {
    message_count: usize = 0,
    total_tokens: usize = 0,
    memory_type: MemoryType = .short_term,
    capacity: usize = 0,
    utilization: f64 = 0,
};

/// Stub conversation context.
pub const ConversationContext = struct {
    messages: []const Message = &[_]Message{},
    total_tokens: usize = 0,
    truncated: bool = false,
    original_count: usize = 0,
};

/// Stub short-term memory.
pub const ShortTermMemory = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) ShortTermMemory {
        _ = capacity;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ShortTermMemory) void {
        self.* = undefined;
    }

    pub fn add(self: *ShortTermMemory, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn getMessages(self: *const ShortTermMemory) []const Message {
        _ = self;
        return &[_]Message{};
    }

    pub fn clear(self: *ShortTermMemory) void {
        _ = self;
    }

    pub fn getStats(self: *const ShortTermMemory) MemoryStats {
        _ = self;
        return .{};
    }
};

/// Stub sliding window memory.
pub const SlidingWindowMemory = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_tokens: usize) SlidingWindowMemory {
        _ = max_tokens;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SlidingWindowMemory) void {
        self.* = undefined;
    }

    pub fn add(self: *SlidingWindowMemory, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn setSystemMessage(self: *SlidingWindowMemory, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn getMessages(self: *const SlidingWindowMemory, allocator: std.mem.Allocator) ![]Message {
        _ = self;
        _ = allocator;
        return error.MemoryDisabled;
    }

    pub fn clear(self: *SlidingWindowMemory) void {
        _ = self;
    }

    pub fn getStats(self: *const SlidingWindowMemory) MemoryStats {
        _ = self;
        return .{};
    }
};

/// Stub summarizing memory.
pub const SummarizingMemory = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SummaryConfig) SummarizingMemory {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SummarizingMemory) void {
        self.* = undefined;
    }

    pub fn add(self: *SummarizingMemory, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn getContext(self: *const SummarizingMemory, allocator: std.mem.Allocator) ![]Message {
        _ = self;
        _ = allocator;
        return error.MemoryDisabled;
    }

    pub fn clear(self: *SummarizingMemory) void {
        _ = self;
    }

    pub fn getStats(self: *const SummarizingMemory) MemoryStats {
        _ = self;
        return .{};
    }
};

/// Stub summary config.
const SummaryConfig = struct {
    max_messages: usize = 20,
    keep_recent: usize = 5,
    max_summary_tokens: usize = 500,
};

/// Stub long-term memory.
pub const LongTermMemory = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LongTermConfig) LongTermMemory {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LongTermMemory) void {
        self.* = undefined;
    }

    pub fn store(self: *LongTermMemory, message: Message, embedding: ?[]const f32, importance: f32) !void {
        _ = self;
        _ = message;
        _ = embedding;
        _ = importance;
        return error.MemoryDisabled;
    }

    pub fn retrieve(self: *LongTermMemory, query: []const u8, top_k: ?usize) ![]RetrievalResult {
        _ = self;
        _ = query;
        _ = top_k;
        return error.MemoryDisabled;
    }

    pub fn clear(self: *LongTermMemory) void {
        _ = self;
    }

    pub fn getStats(self: *const LongTermMemory) MemoryStats {
        _ = self;
        return .{};
    }
};

/// Stub long-term config.
const LongTermConfig = struct {
    max_memories: usize = 1000,
    embedding_dim: usize = 384,
    top_k: usize = 5,
    min_similarity: f32 = 0.5,
};

/// Stub retrieval result.
const RetrievalResult = struct {
    message: Message = .{},
    similarity: f32 = 0,
    importance: f32 = 0,
};

/// Stub memory config.
pub const MemoryConfig = struct {
    primary_type: MemoryType = .sliding_window,
    short_term_capacity: usize = 50,
    sliding_window_tokens: usize = 4000,
    system_reserve: usize = 500,
    summary_config: SummaryConfig = .{},
    long_term_config: LongTermConfig = .{},
    enable_long_term: bool = false,
    auto_store_long_term: bool = true,
    long_term_importance_threshold: f32 = 0.6,
};

/// Stub memory manager.
pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    config: MemoryConfig,

    pub fn init(allocator: std.mem.Allocator, config: MemoryConfig) MemoryManager {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *MemoryManager) void {
        self.* = undefined;
    }

    pub fn setSystemMessage(self: *MemoryManager, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn add(self: *MemoryManager, message: Message) !void {
        _ = self;
        _ = message;
        return error.MemoryDisabled;
    }

    pub fn addUserMessage(self: *MemoryManager, content: []const u8) !void {
        _ = self;
        _ = content;
        return error.MemoryDisabled;
    }

    pub fn addAssistantMessage(self: *MemoryManager, content: []const u8) !void {
        _ = self;
        _ = content;
        return error.MemoryDisabled;
    }

    pub fn getContext(self: *MemoryManager, max_tokens: ?usize) ![]Message {
        _ = self;
        _ = max_tokens;
        return error.MemoryDisabled;
    }

    pub fn clear(self: *MemoryManager) void {
        _ = self;
    }
};

/// Stub persistence errors.
pub const PersistenceError = error{
    PersistenceDisabled,
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

/// Stub session metadata.
pub const SessionMeta = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    created_at: i64 = 0,
    updated_at: i64 = 0,
    message_count: usize = 0,
    total_tokens: usize = 0,

    pub fn deinit(self: *SessionMeta, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub session data.
pub const SessionData = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    created_at: i64 = 0,
    updated_at: i64 = 0,
    messages: []Message = &.{},
    config: SessionConfig = .{},

    pub fn deinit(self: *SessionData, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub session config.
pub const SessionConfig = struct {
    memory_type: MemoryType = .sliding_window,
    max_tokens: usize = 4000,
    temperature: f32 = 0.7,
    model: []const u8 = "default",
    system_prompt: ?[]const u8 = null,
};

/// Stub session store.
pub const SessionStore = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,

    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) SessionStore {
        return .{
            .allocator = allocator,
            .base_dir = base_dir,
        };
    }

    pub fn saveSession(self: *SessionStore, session: SessionData) PersistenceError!void {
        _ = self;
        _ = session;
        return error.PersistenceDisabled;
    }

    pub fn loadSession(self: *SessionStore, id: []const u8) PersistenceError!SessionData {
        _ = self;
        _ = id;
        return error.PersistenceDisabled;
    }

    pub fn deleteSession(self: *SessionStore, id: []const u8) PersistenceError!void {
        _ = self;
        _ = id;
        return error.PersistenceDisabled;
    }

    pub fn listSessions(self: *SessionStore) PersistenceError![]SessionMeta {
        _ = self;
        return error.PersistenceDisabled;
    }

    pub fn sessionExists(self: *SessionStore, id: []const u8) bool {
        _ = self;
        _ = id;
        return false;
    }
};

/// Stub factory functions.
pub fn createShortTermMemory(allocator: std.mem.Allocator, capacity: usize) ShortTermMemory {
    return ShortTermMemory.init(allocator, capacity);
}

pub fn createSlidingWindowMemory(allocator: std.mem.Allocator, max_tokens: usize) SlidingWindowMemory {
    return SlidingWindowMemory.init(allocator, max_tokens);
}

pub fn createSummarizingMemory(allocator: std.mem.Allocator, config: SummaryConfig) SummarizingMemory {
    return SummarizingMemory.init(allocator, config);
}

pub fn createLongTermMemory(allocator: std.mem.Allocator, config: LongTermConfig) LongTermMemory {
    return LongTermMemory.init(allocator, config);
}

pub fn createMemoryManager(allocator: std.mem.Allocator) MemoryManager {
    return MemoryManager.init(allocator, .{});
}

pub fn createMemoryManagerWithConfig(allocator: std.mem.Allocator, config: MemoryConfig) MemoryManager {
    return MemoryManager.init(allocator, config);
}
