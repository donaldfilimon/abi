//! Chat memory stub types — extracted from stub.zig.

const std = @import("std");
const semantic_store = @import("../../database/stub.zig").semantic_store;

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

pub const SummaryConfig = struct {
    max_messages: usize = 20,
    keep_recent: usize = 5,
    max_summary_tokens: usize = 500,
};

pub const LongTermConfig = struct {
    max_memories: usize = 1000,
    embedding_dim: usize = 384,
    top_k: usize = 5,
    min_similarity: f32 = 0.5,
};

pub const RetrievalResult = struct {
    message: Message = .{},
    similarity: f32 = 0,
    importance: f32 = 0,
    hit: semantic_store.RetrievalHit = .{},
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

// --- Persistence Types ---

pub const PersistenceError = error{
    FeatureDisabled,
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
