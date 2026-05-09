//! Chat memory stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const MessageRole = types.MessageRole;
pub const Message = types.Message;
pub const MemoryType = types.MemoryType;
pub const MemoryStats = types.MemoryStats;
pub const ConversationContext = types.ConversationContext;
pub const SummaryConfig = types.SummaryConfig;
pub const LongTermConfig = types.LongTermConfig;
pub const RetrievalResult = types.RetrievalResult;
pub const MemoryConfig = types.MemoryConfig;
pub const PersistenceError = types.PersistenceError;
pub const SessionMeta = types.SessionMeta;
pub const SessionData = types.SessionData;
pub const SessionConfig = types.SessionConfig;

const stub_root = @This();

pub const persistence = struct {
    pub const SessionStore = stub_root.SessionStore;
    pub const SessionData_ = stub_root.SessionData;
    pub const SessionMeta_ = stub_root.SessionMeta;
    pub const SessionConfig_ = stub_root.SessionConfig;
    pub const PersistenceError_ = stub_root.PersistenceError;
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
        return error.FeatureDisabled;
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
        return error.FeatureDisabled;
    }
    pub fn setSystemMessage(_: *SlidingWindowMemory, _: Message) !void {
        return error.FeatureDisabled;
    }
    pub fn getMessages(_: *const SlidingWindowMemory, _: std.mem.Allocator) ![]Message {
        return error.FeatureDisabled;
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
        return error.FeatureDisabled;
    }
    pub fn getContext(_: *const SummarizingMemory, _: std.mem.Allocator) ![]Message {
        return error.FeatureDisabled;
    }
    pub fn clear(_: *SummarizingMemory) void {}
    pub fn getStats(_: *const SummarizingMemory) MemoryStats {
        return .{};
    }
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
        return error.FeatureDisabled;
    }
    pub fn retrieve(_: *LongTermMemory, _: []const u8, _: ?usize) ![]RetrievalResult {
        return error.FeatureDisabled;
    }
    pub fn clear(_: *LongTermMemory) void {}
    pub fn getStats(_: *const LongTermMemory) MemoryStats {
        return .{};
    }
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
        return error.FeatureDisabled;
    }
    pub fn add(_: *MemoryManager, _: Message) !void {
        return error.FeatureDisabled;
    }
    pub fn addUserMessage(_: *MemoryManager, _: []const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn addAssistantMessage(_: *MemoryManager, _: []const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn getContext(_: *MemoryManager, _: ?usize) ![]Message {
        return error.FeatureDisabled;
    }
    pub fn clear(_: *MemoryManager) void {}
};

pub const SessionStore = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) SessionStore {
        return .{ .allocator = allocator, .base_dir = base_dir };
    }
    pub fn saveSession(_: *SessionStore, _: SessionData) PersistenceError!void {
        return error.FeatureDisabled;
    }
    pub fn loadSession(_: *SessionStore, _: []const u8) PersistenceError!SessionData {
        return error.FeatureDisabled;
    }
    pub fn deleteSession(_: *SessionStore, _: []const u8) PersistenceError!void {
        return error.FeatureDisabled;
    }
    pub fn listSessions(_: *SessionStore) PersistenceError![]SessionMeta {
        return error.FeatureDisabled;
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

test {
    std.testing.refAllDecls(@This());
}
