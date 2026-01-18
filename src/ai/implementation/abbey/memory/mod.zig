//! Abbey Memory Module
//!
//! Comprehensive memory architecture with three tiers:
//! - Episodic: Event-based memory with temporal organization
//! - Semantic: Long-term knowledge storage with vector retrieval
//! - Working: Active context with attention-based prioritization
//!
//! Also includes memory consolidation and retrieval strategies.

const std = @import("std");
const types = @import("../../../core/types.zig");
const config = @import("../../../core/config.zig");

pub const episodic = @import("episodic.zig");
pub const semantic = @import("semantic.zig");
pub const working = @import("working.zig");

// Re-exports
pub const Episode = episodic.Episode;
pub const EpisodicMemory = episodic.EpisodicMemory;
pub const EpisodicStats = episodic.EpisodicMemory.EpisodicStats;

pub const Knowledge = semantic.Knowledge;
pub const SemanticMemory = semantic.SemanticMemory;
pub const SemanticStats = semantic.SemanticMemory.SemanticStats;
pub const KnowledgeMatch = semantic.SemanticMemory.KnowledgeMatch;

pub const WorkingItem = working.WorkingItem;
pub const WorkingMemory = working.WorkingMemory;
pub const WorkingStats = working.WorkingMemory.WorkingStats;

// ============================================================================
// Unified Memory Manager
// ============================================================================

/// Unified interface to all memory systems
pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    config: config.MemoryConfig,

    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    working_memory: WorkingMemory,

    // Cross-memory associations
    episode_to_knowledge: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(u64)),

    // Statistics
    total_stores: usize = 0,
    total_retrievals: usize = 0,
    consolidation_count: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, mem_config: config.MemoryConfig) !Self {
        var episode = EpisodicMemory.init(allocator, mem_config.max_episodes, mem_config.embedding_dim);
        errdefer episode.deinit();

        var semantic_mem = SemanticMemory.init(allocator, mem_config.embedding_dim);
        errdefer semantic_mem.deinit();

        const work = WorkingMemory.init(
            allocator,
            mem_config.short_term_capacity,
            mem_config.max_context_tokens,
        );

        return Self{
            .allocator = allocator,
            .config = mem_config,
            .episodic_memory = episode,
            .semantic_memory = semantic_mem,
            .working_memory = work,
            .episode_to_knowledge = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.episodic_memory.deinit();
        self.semantic_memory.deinit();
        self.working_memory.deinit();

        var it = self.episode_to_knowledge.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.episode_to_knowledge.deinit(self.allocator);
    }

    // ========================================================================
    // Unified Store Operations
    // ========================================================================

    /// Store a message (goes to working and episodic)
    pub fn storeMessage(self: *Self, message: types.Message) !void {
        // Add to working memory
        const item_type: WorkingItem.ItemType = switch (message.role) {
            .user => .user_input,
            .assistant => .assistant_output,
            .system => .context,
            .tool => .context,
            .internal => .context,
        };
        _ = try self.working_memory.add(message.content, item_type, message.importance);

        // Add to current episode
        try self.episodic_memory.addMessage(message);

        self.total_stores += 1;
    }

    /// Store knowledge (goes to semantic memory)
    pub fn storeKnowledge(
        self: *Self,
        content: []const u8,
        category: Knowledge.KnowledgeCategory,
        source: Knowledge.KnowledgeSource,
    ) !u64 {
        const id = try self.semantic_memory.store(content, category, source);
        self.total_stores += 1;
        return id;
    }

    /// Store knowledge with embedding
    pub fn storeKnowledgeWithEmbedding(
        self: *Self,
        content: []const u8,
        embedding: []const f32,
        category: Knowledge.KnowledgeCategory,
        source: Knowledge.KnowledgeSource,
    ) !u64 {
        const id = try self.semantic_memory.storeWithEmbedding(content, embedding, category, source);
        self.total_stores += 1;
        return id;
    }

    // ========================================================================
    // Unified Retrieve Operations
    // ========================================================================

    /// Get context for current conversation
    pub fn getContext(self: *Self, max_tokens: usize) ![]u8 {
        self.total_retrievals += 1;
        return self.working_memory.buildContext(max_tokens);
    }

    /// Get relevant knowledge
    pub fn getRelevantKnowledge(
        self: *Self,
        query_embedding: []const f32,
        top_k: usize,
    ) ![]const KnowledgeMatch {
        self.total_retrievals += 1;
        return self.semantic_memory.search(query_embedding, top_k);
    }

    /// Get recent episodes
    pub fn getRecentEpisodes(self: *Self, count: usize) []const *Episode {
        self.total_retrievals += 1;
        return self.episodic_memory.getRecent(count);
    }

    /// Hybrid retrieval: recent + relevant
    pub fn getHybridContext(
        self: *Self,
        query_embedding: ?[]const f32,
        max_tokens: usize,
        knowledge_slots: usize,
    ) !HybridContext {
        self.total_retrievals += 1;

        // Get working memory context
        const working_context = try self.working_memory.buildContext(max_tokens / 2);

        // Get relevant knowledge if embedding provided
        var knowledge_context: ?[]u8 = null;
        if (query_embedding) |emb| {
            const matches = try self.semantic_memory.search(emb, knowledge_slots);
            defer self.allocator.free(matches);

            if (matches.len > 0) {
                var kb = std.ArrayListUnmanaged(u8){};
                for (matches) |match| {
                    if (self.semantic_memory.get(match.id)) |k| {
                        try kb.appendSlice(self.allocator, "[Knowledge] ");
                        try kb.appendSlice(self.allocator, k.content);
                        try kb.appendSlice(self.allocator, "\n");
                    }
                }
                knowledge_context = try kb.toOwnedSlice(self.allocator);
            }
        }

        // Get recent episode summary
        const recent = self.episodic_memory.getRecent(3);
        var episode_context: ?[]u8 = null;
        if (recent.len > 0) {
            var eb = std.ArrayListUnmanaged(u8){};
            try eb.appendSlice(self.allocator, "[Recent conversation context available]\n");
            episode_context = try eb.toOwnedSlice(self.allocator);
        }

        return HybridContext{
            .working = working_context,
            .knowledge = knowledge_context,
            .episodic = episode_context,
        };
    }

    pub const HybridContext = struct {
        working: []u8,
        knowledge: ?[]u8,
        episodic: ?[]u8,

        pub fn deinit(self: *HybridContext, allocator: std.mem.Allocator) void {
            allocator.free(self.working);
            if (self.knowledge) |k| allocator.free(k);
            if (self.episodic) |e| allocator.free(e);
        }

        pub fn totalSize(self: *const HybridContext) usize {
            var size = self.working.len;
            if (self.knowledge) |k| size += k.len;
            if (self.episodic) |e| size += e.len;
            return size;
        }
    };

    // ========================================================================
    // Memory Management
    // ========================================================================

    /// Begin a new conversation episode
    pub fn beginConversation(self: *Self) !void {
        _ = try self.episodic_memory.beginEpisode();
    }

    /// End current conversation
    pub fn endConversation(self: *Self) !void {
        try self.episodic_memory.endEpisode();
    }

    /// Consolidate memories
    pub fn consolidate(self: *Self) !void {
        // Consolidate old episodes
        _ = try self.episodic_memory.consolidate(24); // 24 hours

        // Prune low-value knowledge
        _ = self.semantic_memory.prune(0.3, 2);

        // Decay working memory
        self.working_memory.decayAll(0.1);

        self.consolidation_count += 1;
    }

    /// Clear working memory (between conversations)
    pub fn clearWorking(self: *Self) void {
        self.working_memory.clear();
    }

    /// Full reset
    pub fn reset(self: *Self) void {
        self.working_memory.clear();

        for (self.episodic_memory.episodes.items) |*ep| {
            ep.deinit(self.allocator);
        }
        self.episodic_memory.episodes.clearRetainingCapacity();

        for (self.semantic_memory.knowledge.items) |*k| {
            k.deinit(self.allocator);
        }
        self.semantic_memory.knowledge.clearRetainingCapacity();

        self.total_stores = 0;
        self.total_retrievals = 0;
        self.consolidation_count = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    pub fn getStats(self: *const Self) MemoryStats {
        return .{
            .episodic = self.episodic_memory.getStats(),
            .semantic = self.semantic_memory.getStats(),
            .working = self.working_memory.getStats(),
            .total_stores = self.total_stores,
            .total_retrievals = self.total_retrievals,
            .consolidation_count = self.consolidation_count,
        };
    }

    pub const MemoryStats = struct {
        episodic: EpisodicStats,
        semantic: SemanticStats,
        working: WorkingStats,
        total_stores: usize,
        total_retrievals: usize,
        consolidation_count: usize,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "memory manager basic" {
    const allocator = std.testing.allocator;

    var manager = try MemoryManager.init(allocator, .{});
    defer manager.deinit();

    // Store a message
    try manager.storeMessage(types.Message.user("Hello!"));

    // Store knowledge
    const kid = try manager.storeKnowledge("Zig is great", .fact, .user_stated);
    try std.testing.expect(kid > 0);

    // Get context
    const ctx = try manager.getContext(1000);
    defer allocator.free(ctx);
    try std.testing.expect(ctx.len > 0);
}

test "memory manager hybrid context" {
    const allocator = std.testing.allocator;

    var manager = try MemoryManager.init(allocator, .{});
    defer manager.deinit();

    try manager.storeMessage(types.Message.user("Test message"));

    var hybrid = try manager.getHybridContext(null, 1000, 5);
    defer hybrid.deinit(allocator);

    try std.testing.expect(hybrid.working.len > 0);
}

test "memory manager consolidation" {
    const allocator = std.testing.allocator;

    var manager = try MemoryManager.init(allocator, .{});
    defer manager.deinit();

    try manager.consolidate();
    try std.testing.expectEqual(@as(usize, 1), manager.consolidation_count);
}
