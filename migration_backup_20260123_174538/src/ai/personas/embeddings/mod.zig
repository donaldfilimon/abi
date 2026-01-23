//! Persona Embeddings Sub-module
//!
//! Provides vector-based persona search and adaptive learning capabilities.
//! This module coordinates between WDBX storage and persona selection logic.
//!
//! Features:
//! - HNSW-based persona similarity search
//! - Persona characteristic embeddings with seed data
//! - Adaptive learning from user interactions and corrections
//! - Domain and user-specific weight management

const std = @import("std");
const index = @import("persona_index.zig");
const learning = @import("learning.zig");
const seed = @import("seed_data.zig");

// Re-export core types from persona_index
pub const PersonaEmbeddingIndex = index.PersonaEmbeddingIndex;
pub const PersonaMatch = index.PersonaMatch;
pub const IndexConfig = index.IndexConfig;

// Re-export types from learning
pub const AdaptiveLearner = learning.AdaptiveLearner;
pub const InteractionResult = learning.InteractionResult;
pub const LearnerConfig = learning.LearnerConfig;
pub const TrendDirection = learning.TrendDirection;
pub const TrendAnalysis = learning.TrendAnalysis;
pub const LearnerStats = learning.LearnerStats;
pub const DomainWeights = learning.DomainWeights;

// Re-export seed data types and functions
pub const PersonaCharacteristics = seed.PersonaCharacteristics;
pub const DomainPersonaMapping = seed.DomainPersonaMapping;
pub const ABBEY_CHARACTERISTICS = seed.ABBEY_CHARACTERISTICS;
pub const AVIVA_CHARACTERISTICS = seed.AVIVA_CHARACTERISTICS;
pub const ABI_CHARACTERISTICS = seed.ABI_CHARACTERISTICS;
pub const DOMAIN_MAPPINGS = seed.DOMAIN_MAPPINGS;
pub const getCharacteristics = seed.getCharacteristics;
pub const getCombinedCharacteristics = seed.getCombinedCharacteristics;
pub const getAllPersonas = seed.getAllPersonas;
pub const findDomainMapping = seed.findDomainMapping;

/// Main module state for the embeddings sub-feature.
pub const EmbeddingsModule = struct {
    allocator: std.mem.Allocator,
    index: *PersonaEmbeddingIndex,
    learner: AdaptiveLearner,

    const Self = @This();

    /// Initialize the embeddings module with required dependencies.
    pub fn init(
        allocator: std.mem.Allocator,
        db_ctx: anytype, // *database.Context
        emb_ctx: anytype, // *embeddings.Context
    ) !*Self {
        return initWithConfig(allocator, db_ctx, emb_ctx, .{}, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        db_ctx: anytype,
        emb_ctx: anytype,
        index_config: IndexConfig,
        learner_config: LearnerConfig,
    ) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const idx = try PersonaEmbeddingIndex.initWithConfig(allocator, db_ctx, emb_ctx, index_config);
        errdefer idx.deinit();

        self.* = .{
            .allocator = allocator,
            .index = idx,
            .learner = AdaptiveLearner.initWithConfig(allocator, learner_config),
        };

        return self;
    }

    /// Shutdown the module and release all resources.
    pub fn deinit(self: *Self) void {
        self.index.deinit();
        self.learner.deinit();
        self.allocator.destroy(self);
    }

    /// Find the best persona for a query, considering learned weights.
    pub fn findBestPersonaWithLearning(
        self: *Self,
        allocator: std.mem.Allocator,
        query: []const u8,
        domain: ?[]const u8,
        user_id: ?[]const u8,
        top_k: usize,
    ) ![]PersonaMatch {
        // Get base matches from embedding similarity
        const matches = try self.index.findBestPersona(allocator, query, top_k);
        errdefer allocator.free(matches);

        // Apply learned weights
        for (matches) |*match| {
            const learned_weight = self.learner.getCombinedWeight(match.persona, domain, user_id);
            match.similarity *= learned_weight;
        }

        // Re-sort by adjusted scores
        std.mem.sort(PersonaMatch, matches, {}, struct {
            fn lessThan(_: void, a: PersonaMatch, b: PersonaMatch) bool {
                return a.similarity > b.similarity; // Descending
            }
        }.lessThan);

        return matches;
    }

    /// Record an interaction result for learning.
    pub fn recordInteraction(self: *Self, result: InteractionResult) !void {
        try self.learner.recordInteraction(result);

        // Also store in index if conversation ID is provided
        if (result.domain) |_| {
            // Use timestamp as conversation ID for simplicity
            try self.index.storeConversationEmbedding(
                @intCast(@as(i64, result.timestamp)),
                "", // Would need content to embed
                result.persona,
                result.success_score,
            );
        }
    }

    /// Record a user correction for improved learning.
    pub fn recordCorrection(
        self: *Self,
        query: []const u8,
        was_persona: types.PersonaType,
        correct_persona: types.PersonaType,
        domain: ?[]const u8,
        user_id: ?[]const u8,
    ) !void {
        const timestamp = std.time.timestamp();

        // Record in learner
        try self.learner.recordInteraction(.{
            .persona = was_persona,
            .success_score = 0.0, // Failure
            .timestamp = timestamp,
            .domain = domain,
            .user_id = user_id,
            .is_correction = true,
            .corrected_to = correct_persona,
        });

        // Store feedback embedding
        try self.index.storeFeedbackEmbedding(
            @intCast(timestamp),
            query,
            correct_persona,
            was_persona,
        );
    }

    /// Get current weight for a persona in context.
    pub fn getPersonaWeight(
        self: *const Self,
        persona: types.PersonaType,
        domain: ?[]const u8,
        user_id: ?[]const u8,
    ) f32 {
        return self.learner.getCombinedWeight(persona, domain, user_id);
    }

    /// Analyze performance trend for a persona.
    pub fn analyzeTrend(self: *const Self, persona: types.PersonaType) TrendAnalysis {
        return self.learner.analyzeTrend(persona, 100);
    }

    /// Get module statistics.
    pub fn getStats(self: *const Self) ModuleStats {
        return .{
            .index_stats = self.index.getStats(),
            .learner_stats = self.learner.getStats(),
        };
    }

    /// Reset learned data (for testing or fresh start).
    pub fn reset(self: *Self) void {
        self.learner.reset();
    }
};

/// Import types for public API
const types = @import("../types.zig");

/// Combined statistics for the module.
pub const ModuleStats = struct {
    index_stats: PersonaEmbeddingIndex.IndexStats,
    learner_stats: LearnerStats,
};

// Tests
test {
    std.testing.refAllDecls(@This());
    _ = index;
    _ = learning;
    _ = seed;
}
