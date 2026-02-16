//! Chat memory stub — disabled at compile time.

const std = @import("std");
const stub_root = @This();

pub const persistence = struct {
    pub const SessionStore = stub_root.SessionStore;
    pub const SessionData = stub_root.SessionData;
    pub const SessionMeta = stub_root.SessionMeta;
    pub const SessionConfig = stub_root.SessionConfig;
    pub const PersistenceError = stub_root.PersistenceError;
};

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
    pub fn clone(self: Message, _: std.mem.Allocator) !Message {
        return self;
    }
    pub fn deinit(self: *Message, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const MemoryType = enum { short_term, sliding_window, summarizing, long_term, hybrid };

pub const MemoryStats = struct {
    message_count: usize = 0,
    total_tokens: usize = 0,
    memory_type: MemoryType = .short_term,
    capacity: usize = 0,
    utilization: f64 = 0,
};

pub const ConversationContext = struct {
    messages: []const Message = &[_]Message{},
    total_tokens: usize = 0,
    truncated: bool = false,
    original_count: usize = 0,
};

pub const ShortTermMemory = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: usize) ShortTermMemory {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *ShortTermMemory) void {
        self.* = undefined;
    }
    pub fn add(_: *ShortTermMemory, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn getMessages(_: *const ShortTermMemory) []const Message {
        return &[_]Message{};
    }
    pub fn clear(_: *ShortTermMemory) void {}
    pub fn getStats(_: *const ShortTermMemory) MemoryStats {
        return .{};
    }
};

pub const SlidingWindowMemory = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: usize) SlidingWindowMemory {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *SlidingWindowMemory) void {
        self.* = undefined;
    }
    pub fn add(_: *SlidingWindowMemory, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn setSystemMessage(_: *SlidingWindowMemory, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn getMessages(_: *const SlidingWindowMemory, _: std.mem.Allocator) ![]Message {
        return error.MemoryDisabled;
    }
    pub fn clear(_: *SlidingWindowMemory) void {}
    pub fn getStats(_: *const SlidingWindowMemory) MemoryStats {
        return .{};
    }
};

pub const SummarizingMemory = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: SummaryConfig) SummarizingMemory {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *SummarizingMemory) void {
        self.* = undefined;
    }
    pub fn add(_: *SummarizingMemory, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn getContext(_: *const SummarizingMemory, _: std.mem.Allocator) ![]Message {
        return error.MemoryDisabled;
    }
    pub fn clear(_: *SummarizingMemory) void {}
    pub fn getStats(_: *const SummarizingMemory) MemoryStats {
        return .{};
    }
};

const SummaryConfig = struct {
    max_messages: usize = 20,
    keep_recent: usize = 5,
    max_summary_tokens: usize = 500,
};

pub const LongTermMemory = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: LongTermConfig) LongTermMemory {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *LongTermMemory) void {
        self.* = undefined;
    }
    pub fn store(_: *LongTermMemory, _: Message, _: ?[]const f32, _: f32) !void {
        return error.MemoryDisabled;
    }
    pub fn retrieve(_: *LongTermMemory, _: []const u8, _: ?usize) ![]RetrievalResult {
        return error.MemoryDisabled;
    }
    pub fn clear(_: *LongTermMemory) void {}
    pub fn getStats(_: *const LongTermMemory) MemoryStats {
        return .{};
    }
};

const LongTermConfig = struct {
    max_memories: usize = 1000,
    embedding_dim: usize = 384,
    top_k: usize = 5,
    min_similarity: f32 = 0.5,
};

const RetrievalResult = struct {
    message: Message = .{},
    similarity: f32 = 0,
    importance: f32 = 0,
};

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

pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    config: MemoryConfig,
    pub fn init(allocator: std.mem.Allocator, config: MemoryConfig) MemoryManager {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn deinit(self: *MemoryManager) void {
        self.* = undefined;
    }
    pub fn setSystemMessage(_: *MemoryManager, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn add(_: *MemoryManager, _: Message) !void {
        return error.MemoryDisabled;
    }
    pub fn addUserMessage(_: *MemoryManager, _: []const u8) !void {
        return error.MemoryDisabled;
    }
    pub fn addAssistantMessage(_: *MemoryManager, _: []const u8) !void {
        return error.MemoryDisabled;
    }
    pub fn getContext(_: *MemoryManager, _: ?usize) ![]Message {
        return error.MemoryDisabled;
    }
    pub fn clear(_: *MemoryManager) void {}
};

// --- Persistence Types ---

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

pub const SessionMeta = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    created_at: i64 = 0,
    updated_at: i64 = 0,
    message_count: usize = 0,
    total_tokens: usize = 0,
    pub fn deinit(self: *SessionMeta, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const SessionData = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    created_at: i64 = 0,
    updated_at: i64 = 0,
    messages: []Message = &.{},
    config: SessionConfig = .{},
    owns_model: bool = false,
    owns_system_prompt: bool = false,
    pub fn deinit(self: *SessionData, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const SessionConfig = struct {
    memory_type: MemoryType = .sliding_window,
    max_tokens: usize = 4000,
    temperature: f32 = 0.7,
    model: []const u8 = "default",
    system_prompt: ?[]const u8 = null,
};

pub const SessionStore = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) SessionStore {
        return .{ .allocator = allocator, .base_dir = base_dir };
    }
    pub fn saveSession(_: *SessionStore, _: SessionData) PersistenceError!void {
        return error.PersistenceDisabled;
    }
    pub fn loadSession(_: *SessionStore, _: []const u8) PersistenceError!SessionData {
        return error.PersistenceDisabled;
    }
    pub fn deleteSession(_: *SessionStore, _: []const u8) PersistenceError!void {
        return error.PersistenceDisabled;
    }
    pub fn listSessions(_: *SessionStore) PersistenceError![]SessionMeta {
        return error.PersistenceDisabled;
    }
    pub fn sessionExists(_: *SessionStore, _: []const u8) bool {
        return false;
    }
};

// --- Factory Functions ---

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
