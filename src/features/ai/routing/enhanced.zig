//! Enhanced Profile Routing Framework
//!
//! Implements the multi-profile routing and blending algorithms described in
//! the Abbey-Aviva-Abi research document. Works with existing profile system
//! while adding WDBX memory integration and mathematical blending.
//!
//! Key features:
//! - Intent classification with machine learning (future extension)
//! - Risk scoring for policy compliance
//! - Dynamic profile blending (Abbey+Aviva mixing)
//! - WDBX block chaining for session continuity
//! - Hysteresis to prevent oscillation
//!
//! Mathematical models:
//! - Routing probability: P(p | I, C) = softmax(score_p + h_p)
//! - Profile blending: R_final = α·R_Abbey + (1-α)·R_Aviva
//! - Little's Law optimization: λ = N_concurrent / L_latency

const std = @import("std");
const time = @import("../../../services/shared/mod.zig").time;
const profiles = @import("../mod.zig");
const types = profiles.types;
const block_chain = @import("../../../core/database/block_chain.zig");
const embeddings = @import("../embeddings/mod.zig");

/// Enhanced routing result with mathematical blending and WDBX integration
pub const EnhancedRoutingDecision = struct {
    /// Base routing decision
    base_decision: types.RoutingDecision,
    /// Blending coefficient (0.0 = pure Aviva, 1.0 = pure Abbey)
    blend_coefficient: f32 = 0.0,
    /// Block chain ID for this conversation turn
    block_chain_id: ?u64 = null,
    /// Parent block ID for chain continuity
    parent_block_id: ?u64 = null,
    /// Skip pointer for efficient traversal
    skip_pointer: ?u64 = null,
    /// Mathematical confidence including hysteresis
    enhanced_confidence: f32,

    /// Create a new decision with WDBX integration
    pub fn create(allocator: std.mem.Allocator, base: types.RoutingDecision, blend_coeff: f32, parent_id: ?u64, block_id: ?u64) !EnhancedRoutingDecision {
        _ = allocator; // Not used in this implementation
        const enhanced_conf = @min(base.confidence + 0.1, 1.0); // Boost confidence

        return EnhancedRoutingDecision{
            .base_decision = base,
            .blend_coefficient = blend_coeff,
            .block_chain_id = block_id,
            .parent_block_id = parent_id,
            .enhanced_confidence = enhanced_conf,
        };
    }

    pub fn deinit(self: *EnhancedRoutingDecision, allocator: std.mem.Allocator) void {
        self.base_decision.deinit(allocator);
    }
};

/// Profile scores for mathematical selection
pub const ProfileScores = struct {
    abbey_score: f32 = 0.0,
    aviva_score: f32 = 0.0,
    abi_score: f32 = 0.0,

    /// Calculate normalized probabilities via softmax
    pub fn getProbabilities(self: ProfileScores) struct { p_abbey: f32, p_aviva: f32, p_abi: f32 } {
        // Apply softmax: exp(score) / sum(exp(scores))
        const exp_abbey = std.math.exp(self.abbey_score);
        const exp_aviva = std.math.exp(self.aviva_score);
        const exp_abi = std.math.exp(self.abi_score);
        const sum = exp_abbey + exp_aviva + exp_abi;

        return .{
            .p_abbey = exp_abbey / sum,
            .p_aviva = exp_aviva / sum,
            .p_abi = exp_abi / sum,
        };
    }

    /// Get best profile with threshold for blending
    pub fn getBestProfile(self: ProfileScores, threshold: f32) struct { profile: types.ProfileType, should_blend: bool } {
        const probs = self.getProbabilities();

        // Check if dominance exceeds threshold
        if (probs.p_abbey > threshold and probs.p_abbey > probs.p_aviva and probs.p_abbey > probs.p_abi) {
            return .{ .profile = .abbey, .should_blend = false };
        } else if (probs.p_aviva > threshold and probs.p_aviva > probs.p_abbey and probs.p_aviva > probs.p_abi) {
            return .{ .profile = .aviva, .should_blend = false };
        } else if (probs.p_abi > threshold) {
            return .{ .profile = .abi, .should_blend = false };
        } else {
            // No clear winner - blend Abbey and Aviva
            return .{ .profile = if (probs.p_abbey > probs.p_aviva) .abbey else .aviva, .should_blend = true };
        }
    }
};

/// Intent classifier for enhanced routing
pub const IntentClassifier = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize classifier
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Classify intent into research-defined categories
    pub fn classify(self: *const Self, text: []const u8) IntentCategory {
        // Simple rule-based classification (can be extended with ML)
        const lower = std.ascii.allocLowerString(self.allocator, text) catch return .general;
        defer self.allocator.free(lower);

        // Empathy-seeking patterns
        const empathy_patterns = [_][]const u8{ "feel", "emotional", "stressed", "anxious", "frustrated", "sad", "help me", "support", "comfort", "reassure", "vent", "listen" };

        for (empathy_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                return .empathy_seeking;
            }
        }

        // Technical problem-solving
        const technical_patterns = [_][]const u8{ "error", "bug", "fix", "debug", "code", "function", "compile", "implement", "algorithm", "optimize", "performance", "memory" };

        for (technical_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                return .technical_problem;
            }
        }

        // Factual inquiry
        const fact_patterns = [_][]const u8{ "what is", "definition", "explain", "how does", "why does", "fact", "statistic", "data", "research", "study" };

        for (fact_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                return .factual_inquiry;
            }
        }

        return .general;
    }
};

/// Intent categories for enhanced routing
pub const IntentCategory = enum {
    general,
    empathy_seeking,
    technical_problem,
    factual_inquiry,
    creative_generation,
    policy_check,
    safety_critical,
};

/// Enhanced router with WDBX block chain integration
pub const EnhancedRouter = struct {
    allocator: std.mem.Allocator,
    block_chain: ?*block_chain.BlockChain,
    intent_classifier: IntentClassifier,
    previous_profile: ?types.ProfileType = null,
    hysteresis_weight: f32 = 0.2, // Bias toward previous profile
    parent_block_id: ?u64 = null, // Current parent for chain continuity

    const Self = @This();

    /// Initialize enhanced router with optional block chain
    pub fn init(allocator: std.mem.Allocator, session_id: ?[]const u8) !Self {
        var bc: ?*block_chain.BlockChain = null;

        if (session_id) |sid| {
            const chain = try allocator.create(block_chain.BlockChain);
            chain.* = block_chain.BlockChain.init(allocator, sid);
            bc = chain;
        }

        return Self{
            .allocator = allocator,
            .block_chain = bc,
            .intent_classifier = IntentClassifier.init(allocator),
        };
    }

    /// Deinitialize router
    pub fn deinit(self: *Self) void {
        if (self.block_chain) |chain| {
            chain.deinit();
            self.allocator.destroy(chain);
        }
    }

    /// Calculate profile scores based on intent and sentiment
    pub fn calculateScores(self: *Self, request: types.ProfileRequest, sentiment: profiles.abi.SentimentResult) ProfileScores {
        var scores = ProfileScores{};

        // Base score from sentiment analyzer
        if (sentiment.requires_empathy) {
            scores.abbey_score += 2.0;
            scores.aviva_score -= 1.0;
        }

        if (sentiment.is_technical and !sentiment.requires_empathy) {
            scores.aviva_score += 2.0;
            scores.abbey_score -= 0.5;
        }

        // Intent-based adjustments
        const intent = self.intent_classifier.classify(request.content);
        switch (intent) {
            .empathy_seeking => {
                scores.abbey_score += 1.5;
                scores.aviva_score -= 1.0;
            },
            .technical_problem => {
                scores.aviva_score += 1.5;
                scores.abbey_score += 0.5; // Abbey can still help with technical empathy
            },
            .factual_inquiry => {
                scores.aviva_score += 1.0;
                scores.abbey_score += 0.3; // Abbey for contextual framing
            },
            .safety_critical, .policy_check => {
                scores.abi_score += 3.0; // Force Abi for safety
            },
            else => {},
        }

        // Apply hysteresis: bias toward previous profile
        if (self.previous_profile) |prev| {
            switch (prev) {
                .abbey => scores.abbey_score += self.hysteresis_weight,
                .aviva => scores.aviva_score += self.hysteresis_weight,
                .abi => scores.abi_score += self.hysteresis_weight,
                else => {},
            }
        }

        // Urgency boosts Aviva for speed
        if (sentiment.urgency_score > 0.7) {
            scores.aviva_score += sentiment.urgency_score;
        }

        return scores;
    }

    /// Route with enhanced algorithm and WDBX integration
    pub fn routeEnhanced(self: *Self, request: types.ProfileRequest, sentiment: profiles.abi.SentimentResult) !EnhancedRoutingDecision {
        // Calculate mathematical scores
        const scores = self.calculateScores(request, sentiment);

        // Get best profile decision
        const best = scores.getBestProfile(0.7); // 70% dominance threshold

        // Create base routing decision using existing Abi logic
        // (In practice, would integrate with existing AbiRouter)
        const base_reason = try std.fmt.allocPrint(self.allocator, "Enhanced routing: scores(Abbey={d:.2}, Aviva={d:.2}, Abi={d:.2})", .{ scores.abbey_score, scores.aviva_score, scores.abi_score });
        defer self.allocator.free(base_reason);

        const base_decision = types.RoutingDecision{
            .selected_profile = best.profile,
            .confidence = @min(scores.getProbabilities().p_abbey +
                scores.getProbabilities().p_aviva +
                scores.getProbabilities().p_abi, 1.0),
            .emotional_context = sentiment.toEmotionalState(),
            .policy_flags = .{},
            .routing_reason = try self.allocator.dupe(u8, base_reason),
        };

        // Calculate blending coefficient
        const blend_coeff = if (best.should_blend) blk: {
            const probs = scores.getProbabilities();
            // α = p_abbey / (p_abbey + p_aviva)
            break :blk probs.p_abbey / (probs.p_abbey + probs.p_aviva);
        } else 0.0;

        // Generate WDBX block chain entry if configured
        var block_id: ?u64 = null;
        if (self.block_chain) |chain| {
            // Create block with current parent
            block_id = try self.createBlockChainEntry(chain, request, base_decision, blend_coeff);
            // Update parent for next iteration
            self.parent_block_id = block_id;
        }

        // Store for hysteresis
        self.previous_profile = best.profile;

        // Return decision with block ID and parent relationship
        return try EnhancedRoutingDecision.create(self.allocator, base_decision, blend_coeff, null, block_id);
    }

    /// Store routing decision in WDBX block chain
    fn createBlockChainEntry(self: *Self, chain: *block_chain.BlockChain, request: types.ProfileRequest, decision: types.RoutingDecision, blend_coeff: f32) !u64 {
        // Generate embedding of request content
        // Create embeddings model temporarily
        const emb_model = embeddings.EmbeddingModel.init(self.allocator, .{ .dimension = 384 });
        const query_embedding = try emb_model.embed(request.content);
        defer {
            emb_model.deinit();
            self.allocator.free(query_embedding);
        }

        // Create profile tag
        const profile_tag = block_chain.ProfileTag{
            .primary_profile = switch (decision.selected_profile) {
                .abbey => .abbey,
                .aviva => .aviva,
                .abi => .abi,
                else => .blended,
            },
            .blend_coefficient = blend_coeff,
            .secondary_profile = if (blend_coeff > 0.0 and blend_coeff < 1.0) blk: {
                const secondary = if (decision.selected_profile == .abbey) .aviva else .abbey;
                break :blk switch (secondary) {
                    .abbey => .abbey,
                    .aviva => .aviva,
                    .abi => .abi,
                    else => .blended,
                };
            } else null,
        };

        // Create routing weights
        const routing_weights = block_chain.RoutingWeights{
            .abbey_weight = if (decision.selected_profile == .abbey) 1.0 - blend_coeff else blend_coeff,
            .aviva_weight = if (decision.selected_profile == .aviva) 1.0 - blend_coeff else blend_coeff,
            .abi_weight = if (decision.selected_profile == .abi) 1.0 else 0.0,
        };

        // Determine intent from classifier
        const intent = self.intent_classifier.classify(request.content);

        // Create block configuration
        const config = block_chain.BlockConfig{
            .query_embedding = query_embedding,
            .profile_tag = profile_tag,
            .routing_weights = routing_weights,
            .intent = switch (intent) {
                .empathy_seeking => .empathy_seeking,
                .technical_problem => .technical_problem,
                .factual_inquiry => .factual_inquiry,
                .safety_critical => .safety_critical,
                .policy_check => .policy_check,
                .creative_generation => .creative_generation,
                else => .general,
            },
            .parent_block_id = self.parent_block_id,
        };

        // Add block to chain
        const block_id = try chain.addBlock(config);

        return block_id;
    }
};

// Tests
test "ProfileScores softmax calculation" {
    const scores = ProfileScores{
        .abbey_score = 1.0,
        .aviva_score = 0.5,
        .abi_score = 0.1,
    };

    const probs = scores.getProbabilities();

    // Probabilities should sum to ~1.0
    const total = probs.p_abbey + probs.p_aviva + probs.p_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);

    // Abbey should have highest probability
    try std.testing.expect(probs.p_abbey > probs.p_aviva);
    try std.testing.expect(probs.p_abbey > probs.p_abi);
}

test "ProfileScores threshold decisions" {
    // Clear Abbey dominance
    const clear_abbey = ProfileScores{
        .abbey_score = 2.0,
        .aviva_score = 0.1,
        .abi_score = 0.1,
    };

    const clear_result = clear_abbey.getBestProfile(0.7);
    try std.testing.expect(clear_result.profile == .abbey);
    try std.testing.expect(!clear_result.should_blend);

    // Close race -> should blend
    const close_race = ProfileScores{
        .abbey_score = 0.6,
        .aviva_score = 0.5,
        .abi_score = 0.1,
    };

    const close_result = close_race.getBestProfile(0.7);
    try std.testing.expect(close_result.should_blend);
}

test "EnhancedRouter initialization" {
    const allocator = std.testing.allocator;

    var router = try EnhancedRouter.init(allocator, null);
    defer router.deinit();

    try std.testing.expect(router.block_chain == null);
    try std.testing.expect(router.previous_profile == null);
}

test "IntentClassifier basic classification" {
    const allocator = std.testing.allocator;
    var classifier = IntentClassifier.init(allocator);

    const empathy_text = "I feel really stressed about this project";
    try std.testing.expect(classifier.classify(empathy_text) == .empathy_seeking);

    const tech_text = "How do I debug this memory leak in Zig?";
    try std.testing.expect(classifier.classify(tech_text) == .technical_problem);

    const fact_text = "What is the capital of France?";
    try std.testing.expect(classifier.classify(fact_text) == .factual_inquiry);
}

test {
    std.testing.refAllDecls(@This());
}
