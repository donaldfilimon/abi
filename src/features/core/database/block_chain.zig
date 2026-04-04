//! WDBX Block Chaining for Conversational Memory
//!
//! Implements the block-chained memory model described in the Abbey-Aviva-Abi
//! research document. Each conversational turn is stored as a Conversation Block
//! containing embeddings, metadata, temporal markers, references, and integrity
//! fields. Blocks are linked into chains per session.
//!
//! Block Data Model (Research Definition):
//! B_t = { V_t, M_t, T_t, R_t, H_t }
//! Where:
//! - V_t: Query and response embeddings (vector representations)
//! - M_t: Metadata (profile tag, routing weights, intent, risk score)
//! - T_t: Temporal markers (commit and end timestamps for MVCC)
//! - R_t: References (parent block pointer)
//! - H_t: Integrity fields (cryptographic hash chain)
//!
//! Features:
//! - Multi-version concurrency control (MVCC) with commit/end timestamps
//! - Cryptographic block chaining for auditability and integrity
//! - Session continuity with parent block references
//! - Two-stage retrieval: ANN index + reranking with recency decay

const std = @import("std");
const time = @import("../../../foundation/mod.zig").time;
const crypto = std.crypto;

/// Conversation block for WDBX memory system
pub const ConversationBlock = struct {
    // Core content (V_t)
    query_embedding: []const f32,
    response_embedding: ?[]const f32 = null,

    // Metadata (M_t)
    profile_tag: ProfileTag,
    routing_weights: RoutingWeights,
    intent: IntentCategory,
    risk_score: f32 = 0.0,
    policy_flags: PolicyFlags = .{},

    // Temporal markers (T_t) for MVCC
    commit_timestamp: i64, // When block becomes visible (write)
    end_timestamp: ?i64 = null, // When block becomes invisible (for MVCC)

    // References (R_t)
    parent_block_id: ?u64 = null,

    // Integrity (H_t)
    hash: [32]u8, // SHA-256 hash of block content
    previous_hash: [32]u8, // Cryptographic chain link

    // Timestamp for recency decay in retrieval
    timestamp: i64,

    /// Create a new conversation block
    pub fn create(allocator: std.mem.Allocator, config: BlockConfig) !ConversationBlock {
        const now = time.unixSeconds();

        // Calculate hash of block content
        const block_hash = try computeBlockHash(allocator, config);

        return ConversationBlock{
            .query_embedding = try allocator.dupe(f32, config.query_embedding),
            .response_embedding = if (config.response_embedding) |resp|
                try allocator.dupe(f32, resp)
            else
                null,
            .profile_tag = config.profile_tag,
            .routing_weights = config.routing_weights,
            .intent = config.intent,
            .risk_score = config.risk_score,
            .policy_flags = config.policy_flags,
            .commit_timestamp = now,
            .parent_block_id = config.parent_block_id,
            .hash = block_hash,
            .previous_hash = config.previous_hash,
            .timestamp = now,
        };
    }

    /// Deallocate block memory
    pub fn deinit(self: *ConversationBlock, allocator: std.mem.Allocator) void {
        allocator.free(self.query_embedding);
        if (self.response_embedding) |resp| {
            allocator.free(resp);
        }
    }

    /// Check if block is visible under MVCC at given timestamp
    pub fn isVisible(self: ConversationBlock, read_timestamp: i64) bool {
        // Block is visible if:
        // 1. Read timestamp >= commit timestamp (block has been committed)
        // 2. Read timestamp < end timestamp OR end_timestamp is null (block not deleted)
        return read_timestamp >= self.commit_timestamp and
            (self.end_timestamp == null or read_timestamp < self.end_timestamp.?);
    }

    /// Compute recency decay factor for retrieval scoring
    pub fn getRecencyDecay(self: ConversationBlock, current_time: i64) f32 {
        const age_seconds = @as(f64, @floatFromInt(current_time - self.timestamp));
        // Exponential decay: e^(-λ*t) where λ = 1/(7 days in seconds)
        const decay_rate = 1.0 / (7.0 * 24.0 * 60.0 * 60.0); // 7-day half-life
        return @floatCast(std.math.exp(-decay_rate * age_seconds));
    }
};

/// Configuration for creating a block
pub const BlockConfig = struct {
    query_embedding: []const f32,
    response_embedding: ?[]const f32 = null,
    profile_tag: ProfileTag,
    routing_weights: RoutingWeights,
    intent: IntentCategory,
    risk_score: f32 = 0.0,
    policy_flags: PolicyFlags = .{},
    parent_block_id: ?u64 = null,
    previous_hash: [32]u8 = .{0} ** 32,
    /// Pipeline metadata — which step produced this block.
    pipeline_step: PipelineStepTag = .none,
    /// Groups blocks from a single pipeline execution.
    pipeline_id: ?u64 = null,
    /// Ordering within the pipeline (0-indexed).
    step_index: ?u16 = null,
};

/// Profile tag with blending coefficient
pub const ProfileTag = struct {
    primary_profile: ProfileType,
    blend_coefficient: f32 = 0.0, // 0.0 = pure secondary, 1.0 = pure primary
    secondary_profile: ?ProfileType = null,

    pub const ProfileType = enum {
        abbey,
        aviva,
        abi,
        blended,
    };
};

/// Routing weights for mathematical blending
pub const RoutingWeights = struct {
    abbey_weight: f32 = 0.0,
    aviva_weight: f32 = 0.0,
    abi_weight: f32 = 0.0,

    pub fn getPrimaryProfile(self: RoutingWeights) ProfileTag.ProfileType {
        if (self.abbey_weight >= self.aviva_weight and self.abbey_weight >= self.abi_weight) {
            return .abbey;
        } else if (self.aviva_weight >= self.abbey_weight and self.aviva_weight >= self.abi_weight) {
            return .aviva;
        } else {
            return .abi;
        }
    }

    pub fn getBlendCoefficient(self: RoutingWeights) f32 {
        const total = self.abbey_weight + self.aviva_weight;
        if (total == 0) return 0.0;
        return self.abbey_weight / total;
    }
};

/// Intent categories for memory organization
pub const IntentCategory = enum {
    general,
    empathy_seeking,
    technical_problem,
    factual_inquiry,
    creative_generation,
    policy_check,
    safety_critical,
};

/// Tags identifying which pipeline step produced a block.
pub const PipelineStepTag = enum {
    retrieve,
    template,
    route,
    modulate,
    generate,
    validate,
    store,
    transform,
    filter,
    reason,
    none,
};

/// Policy flags from Abi router
pub const PolicyFlags = struct {
    is_safe: bool = true,
    requires_moderation: bool = false,
    sensitive_topic: bool = false,
    pii_detected: bool = false,
    violation_details: ?[]const u8 = null,
};

/// Block chain manager for session memory
pub const BlockChain = struct {
    allocator: std.mem.Allocator,
    blocks: std.AutoHashMapUnmanaged(u64, ConversationBlock),
    session_id: []const u8,
    session_id_owned: bool = false,
    current_head: ?u64 = null,

    const Self = @This();

    /// Initialize a new block chain for a session
    pub fn init(allocator: std.mem.Allocator, session_id: []const u8) Self {
        var session_id_owned = false;
        const stored_session_id = blk: {
            const duped_session_id = allocator.dupe(u8, session_id) catch break :blk session_id;
            session_id_owned = true;
            break :blk duped_session_id;
        };

        return .{
            .allocator = allocator,
            .blocks = .empty,
            .session_id = stored_session_id,
            .session_id_owned = session_id_owned,
        };
    }

    /// Deinitialize block chain
    pub fn deinit(self: *Self) void {
        var iter = self.blocks.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.blocks.deinit(self.allocator);
        if (self.session_id_owned) {
            self.allocator.free(self.session_id);
        }
    }

    /// Add a new block to the chain
    pub fn addBlock(self: *Self, config: BlockConfig) !u64 {
        // Generate block ID from timestamp and hash
        const block_id = generateBlockId(config);

        // Create the block
        var block = try ConversationBlock.create(self.allocator, config);
        errdefer block.deinit(self.allocator);

        // Store block in memory
        const fetch_result = try self.blocks.fetchPut(self.allocator, block_id, block);
        if (fetch_result) |old| {
            var old_block = old.value;
            old_block.deinit(self.allocator);
        }

        // Update chain head
        self.current_head = block_id;

        return block_id;
    }

    /// Get block by ID
    pub fn getBlock(self: *const Self, block_id: u64) ?ConversationBlock {
        return self.blocks.get(block_id);
    }

    /// Traverse chain backward from current head
    pub fn traverseBackward(self: *const Self, max_blocks: usize) ![]const u64 {
        var result = std.ArrayListUnmanaged(u64).empty;
        errdefer result.deinit(self.allocator);

        var current = self.current_head;
        var count: usize = 0;

        while (current != null and count < max_blocks) {
            if (self.blocks.get(current.?)) |block| {
                try result.append(self.allocator, current.?);
                current = block.parent_block_id;
                count += 1;
            } else {
                break;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Create summarized block for long conversations
    pub fn createSummary(self: *Self, block_ids: []const u64) !u64 {
        // Average embeddings of selected blocks
        var query_sum = try self.allocator.alloc(f32, 384); // Assume 384-dim embeddings
        defer self.allocator.free(query_sum);
        @memset(query_sum, 0.0);

        var profile_weights = RoutingWeights{};
        var intent_counts: std.AutoHashMapUnmanaged(IntentCategory, usize) = .empty;
        defer intent_counts.deinit(self.allocator);

        for (block_ids) |block_id| {
            if (self.blocks.get(block_id)) |block| {
                // Accumulate query embedding
                for (block.query_embedding, 0..) |val, i| {
                    query_sum[i] += val;
                }

                // Accumulate routing weights
                profile_weights.abbey_weight += block.routing_weights.abbey_weight;
                profile_weights.aviva_weight += block.routing_weights.aviva_weight;
                profile_weights.abi_weight += block.routing_weights.abi_weight;

                // Count intents
                const count = intent_counts.get(block.intent) orelse 0;
                try intent_counts.put(self.allocator, block.intent, count + 1);
            }
        }

        // Calculate average embedding
        const query_avg = try self.allocator.dupe(f32, query_sum);
        defer self.allocator.free(query_avg);
        const scale = 1.0 / @as(f32, @floatFromInt(block_ids.len));
        for (query_avg) |*val| {
            val.* *= scale;
        }

        // Determine dominant intent
        var dominant_intent = IntentCategory.general;
        var max_count: usize = 0;
        var iter = intent_counts.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* > max_count) {
                max_count = entry.value_ptr.*;
                dominant_intent = entry.key_ptr.*;
            }
        }

        // Create summary block config
        const summary_config = BlockConfig{
            .query_embedding = query_avg,
            .profile_tag = .{
                .primary_profile = profile_weights.getPrimaryProfile(),
                .blend_coefficient = profile_weights.getBlendCoefficient(),
            },
            .routing_weights = profile_weights,
            .intent = dominant_intent,
            .parent_block_id = self.current_head,
            .previous_hash = .{0} ** 32, // Special summary hash
        };

        // Create and add summary block
        return try self.addBlock(summary_config);
    }
};

/// Generate block ID from config
fn generateBlockId(config: BlockConfig) u64 {
    const timestamp = time.unixSeconds();
    var hasher = std.hash.XxHash3.init(0);
    hasher.update(std.mem.asBytes(&timestamp));
    hasher.update(std.mem.sliceAsBytes(config.query_embedding));

    // Add randomness to prevent collisions for identical requests in the same second
    const static = struct {
        var counter: u64 = 0;
    };
    _ = @atomicRmw(u64, &static.counter, .Add, 1, .monotonic);
    hasher.update(std.mem.asBytes(&static.counter));

    const hash = hasher.final();
    return @as(u64, @intCast(timestamp)) ^ hash;
}

/// Compute cryptographic hash of block content
fn computeBlockHash(allocator: std.mem.Allocator, config: BlockConfig) ![32]u8 {
    _ = allocator; // Not used in this implementation
    var hasher = crypto.hash.sha2.Sha256.init(.{});

    // Hash embeddings
    hasher.update(std.mem.sliceAsBytes(config.query_embedding));
    if (config.response_embedding) |resp| {
        hasher.update(std.mem.sliceAsBytes(resp));
    }

    // Hash metadata
    hasher.update(std.mem.asBytes(&config.profile_tag));
    hasher.update(std.mem.asBytes(&config.routing_weights));
    hasher.update(std.mem.asBytes(&config.intent));
    hasher.update(std.mem.asBytes(&config.risk_score));
    hasher.update(std.mem.asBytes(&config.previous_hash));

    var hash: [32]u8 = undefined;
    hasher.final(&hash);
    return hash;
}

/// MVCC store for concurrent access to block chains
pub const MvccStore = struct {
    allocator: std.mem.Allocator,
    chains: std.StringHashMapUnmanaged(BlockChain), // Session ID -> BlockChain
    read_timestamps: std.StringHashMapUnmanaged(i64), // Session -> read timestamp

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .chains = .empty,
            .read_timestamps = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.chains.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.chains.deinit(self.allocator);

        var ts_iter = self.read_timestamps.iterator();
        while (ts_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.read_timestamps.deinit(self.allocator);
    }

    /// Get or create block chain for session
    pub fn getChain(self: *Self, session_id: []const u8) !*BlockChain {
        if (self.chains.getPtr(session_id)) |chain| {
            return chain;
        }

        const session_copy = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(session_copy);

        var chain = BlockChain.init(self.allocator, session_copy);
        errdefer chain.deinit();
        try self.chains.put(self.allocator, session_copy, chain);

        return self.chains.getPtr(session_copy).?;
    }

    /// Set read timestamp for session (MVCC snapshot)
    pub fn setReadTimestamp(self: *Self, session_id: []const u8, timestamp: i64) !void {
        const session_copy = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(session_copy);

        try self.read_timestamps.put(self.allocator, session_copy, timestamp);
    }

    /// Get blocks visible at current read timestamp
    pub fn getVisibleBlocks(self: *Self, session_id: []const u8) ![]const u64 {
        const chain = try self.getChain(session_id);
        const read_ts = self.read_timestamps.get(session_id) orelse time.unixSeconds();

        var visible = std.ArrayListUnmanaged(u64).empty;
        defer visible.deinit(self.allocator);

        var iter = chain.blocks.keyIterator();
        while (iter.next()) |block_id| {
            if (chain.blocks.get(block_id.*)) |block| {
                if (block.isVisible(read_ts)) {
                    try visible.append(self.allocator, block_id.*);
                }
            }
        }

        return visible.toOwnedSlice(self.allocator);
    }
};

// Tests
test "ConversationBlock creation and MVCC" {
    const allocator = std.testing.allocator;

    // Create test embedding
    const embedding = try allocator.alloc(f32, 384);
    defer allocator.free(embedding);
    @memset(embedding, 0.1);

    const config = BlockConfig{
        .query_embedding = embedding,
        .profile_tag = .{ .primary_profile = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
        .previous_hash = .{0} ** 32,
    };

    var block = try ConversationBlock.create(allocator, config);
    defer block.deinit(allocator);

    // Test MVCC visibility
    try std.testing.expect(block.isVisible(block.commit_timestamp));
    try std.testing.expect(block.isVisible(block.commit_timestamp + 1));

    // Test recency decay
    const decay = block.getRecencyDecay(block.timestamp);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), decay, 0.01);
}

test "BlockChain basic operations" {
    const allocator = std.testing.allocator;

    var chain = BlockChain.init(allocator, "test-session");
    defer chain.deinit();

    // Create test embedding
    const embedding = try allocator.alloc(f32, 384);
    defer allocator.free(embedding);
    @memset(embedding, 0.1);

    const config = BlockConfig{
        .query_embedding = embedding,
        .profile_tag = .{ .primary_profile = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
        .previous_hash = .{0} ** 32,
    };

    // Add first block
    const block1_id = try chain.addBlock(config);

    // Add second block with parent reference
    const config2 = BlockConfig{
        .query_embedding = embedding,
        .profile_tag = .{ .primary_profile = .aviva },
        .routing_weights = .{ .abbey_weight = 0.3, .aviva_weight = 0.7 },
        .intent = .technical_problem,
        .parent_block_id = block1_id,
        .previous_hash = .{0} ** 32,
    };

    const block2_id = try chain.addBlock(config2);

    // Verify chain structure
    try std.testing.expect(chain.current_head == block2_id);

    if (chain.getBlock(block2_id)) |block| {
        try std.testing.expect(block.parent_block_id == block1_id);
    } else {
        try std.testing.expect(false); // Block should exist
    }
}

test "MvccStore visibility control" {
    const allocator = std.testing.allocator;

    var store = MvccStore.init(allocator);
    defer store.deinit();

    const session_id = "test-session-mvcc";

    // Get chain and add a block
    const chain = try store.getChain(session_id);

    const embedding = try allocator.alloc(f32, 384);
    defer allocator.free(embedding);
    @memset(embedding, 0.1);

    const config = BlockConfig{
        .query_embedding = embedding,
        .profile_tag = .{ .primary_profile = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
        .previous_hash = .{0} ** 32,
    };

    const block_id = try chain.addBlock(config);

    // Set read timestamp after block creation
    try store.setReadTimestamp(session_id, time.unixSeconds() + 1);

    // Get visible blocks
    const visible = try store.getVisibleBlocks(session_id);
    defer allocator.free(visible);

    try std.testing.expect(visible.len == 1);
    try std.testing.expect(visible[0] == block_id);
}

fn testBlockChainInitBorrowedSessionId(allocator: std.mem.Allocator, session_id: []const u8) !void {
    var chain = BlockChain.init(allocator, session_id);
    defer chain.deinit();

    try std.testing.expect(std.mem.eql(u8, chain.session_id, session_id));

    // Force an allocator-visible failure after the init path so swallowed OOMs
    // still surface to checkAllAllocationFailures.
    const scratch = try allocator.alloc(u8, 1);
    defer allocator.free(scratch);
}

test "BlockChain init handles caller-owned session IDs" {
    const allocator = std.testing.allocator;
    const session_id = try allocator.dupe(u8, "caller-owned-session");
    defer allocator.free(session_id);

    try std.testing.checkAllAllocationFailures(allocator, testBlockChainInitBorrowedSessionId, .{session_id});
}

fn testMvccStoreGetChainInsertion(allocator: std.mem.Allocator, session_id: []const u8) !void {
    var store = MvccStore.init(allocator);
    defer store.deinit();

    const chain = try store.getChain(session_id);
    try std.testing.expect(chain.current_head == null);

    const scratch = try allocator.alloc(u8, 1);
    defer allocator.free(scratch);
}

test "MvccStore.getChain cleans up on insertion failure" {
    const allocator = std.testing.allocator;
    const session_id = try allocator.dupe(u8, "mvcc-oom-session");
    defer allocator.free(session_id);

    try std.testing.checkAllAllocationFailures(allocator, testMvccStoreGetChainInsertion, .{session_id});
}
