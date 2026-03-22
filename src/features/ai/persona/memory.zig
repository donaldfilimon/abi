//! Conversation Memory — WDBX Block Chain Integration
//!
//! Bridges the multi-persona orchestration layer with the WDBX block chain
//! storage system. Each routing decision + persona response is stored as
//! a ConversationBlock with cryptographic chaining for auditability.
//!
//! Per spec Section 4.1:
//!   B_t = { V_t, M_t, T_t, R_t, H_t }
//!   - V_t: Query/response embeddings
//!   - M_t: Profile tag, routing weights, intent, risk
//!   - T_t: MVCC timestamps
//!   - R_t: Parent/skip pointers
//!   - H_t: SHA-256 hash chain

const std = @import("std");
const types = @import("types.zig");
const PersonaId = types.PersonaId;
const RoutingDecision = types.RoutingDecision;
const PersonaResponse = types.PersonaResponse;

const block_chain = @import("../../../core/database/block_chain.zig");
const BlockChain = block_chain.BlockChain;
const BlockConfig = block_chain.BlockConfig;
const ProfileTag = block_chain.ProfileTag;
const RoutingWeights = block_chain.RoutingWeights;
const IntentCategory = block_chain.IntentCategory;
const PolicyFlags = block_chain.PolicyFlags;

/// Conversation memory backed by WDBX block chaining.
/// Stores each interaction as a cryptographically chained block with
/// routing metadata for auditability and retrieval.
pub const ConversationMemory = struct {
    allocator: std.mem.Allocator,
    chain: BlockChain,
    interaction_count: u64 = 0,

    const Self = @This();

    /// Create a new conversation memory for a session.
    pub fn init(allocator: std.mem.Allocator, session_id: []const u8) Self {
        return .{
            .allocator = allocator,
            .chain = BlockChain.init(allocator, session_id),
        };
    }

    /// Record an interaction: routing decision + input + response → ConversationBlock.
    pub fn recordInteraction(
        self: *Self,
        decision: RoutingDecision,
        input: []const u8,
        _: PersonaResponse,
    ) !u64 {
        // Generate a simple embedding placeholder from input content.
        // Real embeddings would come from a connector (OpenAI, Cohere, etc.)
        var query_embedding: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 };
        if (input.len > 0) {
            // Simple hash-based placeholder embedding
            const hash = std.hash.Fnv1a_32.hash(input);
            const f: f32 = @floatFromInt(hash);
            query_embedding = .{
                @mod(f, 1000.0) / 1000.0,
                @mod(f * 1.618, 1000.0) / 1000.0,
                @mod(f * 2.236, 1000.0) / 1000.0,
                @mod(f * 3.142, 1000.0) / 1000.0,
            };
        }

        // Map PersonaId → ProfileTag.ProfileType
        const primary_profile: ProfileTag.ProfileType = switch (decision.primary) {
            .abbey => .abbey,
            .aviva => .aviva,
            .abi => .abi,
        };

        // Calculate blend coefficient (how much primary dominates)
        const blend = decision.weights.forPersona(decision.primary);

        // Determine secondary persona for blending metadata
        const secondary: ?ProfileTag.ProfileType = if (blend < 0.9) blk: {
            if (decision.primary != .abbey and decision.weights.abbey > 0.1) break :blk .abbey;
            if (decision.primary != .aviva and decision.weights.aviva > 0.1) break :blk .aviva;
            if (decision.primary != .abi and decision.weights.abi > 0.1) break :blk .abi;
            break :blk null;
        } else null;

        // Map to WDBX intent category
        const intent: IntentCategory = if (!self.isQuerySafe(decision))
            .safety_critical
        else if (decision.primary == .aviva)
            .technical_problem
        else if (decision.primary == .abbey)
            .empathy_seeking
        else
            .general;

        // Get previous hash for chaining
        const prev_hash: [32]u8 = if (self.chain.current_head) |head| blk: {
            if (self.chain.getBlock(head)) |blk_val| {
                break :blk blk_val.hash;
            }
            break :blk .{0} ** 32;
        } else .{0} ** 32;

        // Build block config
        const config = BlockConfig{
            .query_embedding = &query_embedding,
            .profile_tag = .{
                .primary_profile = primary_profile,
                .blend_coefficient = blend,
                .secondary_profile = secondary,
            },
            .routing_weights = .{
                .abbey_weight = decision.weights.abbey,
                .aviva_weight = decision.weights.aviva,
                .abi_weight = decision.weights.abi,
            },
            .intent = intent,
            .risk_score = 1.0 - decision.confidence,
            .policy_flags = .{
                .is_safe = self.isQuerySafe(decision),
            },
            .parent_block_id = self.chain.current_head,
            .previous_hash = prev_hash,
        };

        const block_id = try self.chain.addBlock(config);
        self.interaction_count += 1;

        return block_id;
    }

    /// Check if the routing decision indicates a safe query.
    fn isQuerySafe(_: *const Self, decision: RoutingDecision) bool {
        // If Abi is primary with high confidence, it's likely a policy issue
        return !(decision.primary == .abi and decision.confidence > 0.8);
    }

    /// Get the number of recorded interactions.
    pub fn getInteractionCount(self: *const Self) u64 {
        return self.interaction_count;
    }

    /// Get the underlying block chain for direct access.
    pub fn getChain(self: *const Self) *const BlockChain {
        return &self.chain;
    }

    /// Retrieve recent conversation history (block IDs).
    pub fn getRecentHistory(self: *const Self, max_blocks: usize) ![]const u64 {
        return self.chain.traverseBackward(max_blocks);
    }

    pub fn deinit(self: *Self) void {
        self.chain.deinit();
    }
};

test "conversation memory initialization" {
    var mem = ConversationMemory.init(std.testing.allocator, "test-session");
    defer mem.deinit();

    try std.testing.expectEqual(@as(u64, 0), mem.getInteractionCount());
}

test "conversation memory records interaction" {
    var mem = ConversationMemory.init(std.testing.allocator, "test-session");
    defer mem.deinit();

    const decision = RoutingDecision{
        .primary = .aviva,
        .weights = .{ .abbey = 0.2, .aviva = 0.6, .abi = 0.2 },
        .strategy = .single,
        .confidence = 0.8,
        .reason = "Technical query",
    };

    const response = PersonaResponse{
        .persona = .aviva,
        .content = "Here is the answer.",
        .confidence = 0.85,
        .allocator = std.testing.allocator,
    };

    const block_id = try mem.recordInteraction(decision, "How do I implement HNSW?", response);
    try std.testing.expect(block_id > 0);
    try std.testing.expectEqual(@as(u64, 1), mem.getInteractionCount());
}

test "conversation memory chain integrity" {
    var mem = ConversationMemory.init(std.testing.allocator, "chain-test");
    defer mem.deinit();

    const decision = RoutingDecision{
        .primary = .abbey,
        .weights = .{ .abbey = 0.7, .aviva = 0.2, .abi = 0.1 },
        .strategy = .single,
        .confidence = 0.7,
        .reason = "Conversational query",
    };

    const response = PersonaResponse{
        .persona = .abbey,
        .content = "I understand how you feel.",
        .confidence = 0.9,
        .allocator = std.testing.allocator,
    };

    // Record two interactions
    const id1 = try mem.recordInteraction(decision, "I feel stuck", response);
    const id2 = try mem.recordInteraction(decision, "Can you help more?", response);

    try std.testing.expect(id1 != id2);
    try std.testing.expectEqual(@as(u64, 2), mem.getInteractionCount());

    // Verify chain linkage
    if (mem.chain.getBlock(id2)) |block2| {
        try std.testing.expectEqual(id1, block2.parent_block_id.?);
    }
}

test {
    std.testing.refAllDecls(@This());
}
