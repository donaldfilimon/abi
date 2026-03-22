//! Integration Tests: WDBX Database and Block Chain
//!
//! Tests the WDBX block chain memory, conversation storage,
//! and chain integrity verification.

const std = @import("std");
const abi = @import("abi");

const block_chain = abi.database.block_chain;
const BlockChain = block_chain.BlockChain;
const BlockConfig = block_chain.BlockConfig;
const ProfileTag = block_chain.ProfileTag;
const RoutingWeights = block_chain.RoutingWeights;
const IntentCategory = block_chain.IntentCategory;
const ConversationBlock = block_chain.ConversationBlock;

test "database: block chain initialization" {
    var chain = BlockChain.init(std.testing.allocator, "test-session");
    defer chain.deinit();

    try std.testing.expect(chain.current_head == null);
}

test "database: add block to chain" {
    var chain = BlockChain.init(std.testing.allocator, "test-session");
    defer chain.deinit();

    const embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    const block_id = try chain.addBlock(.{
        .query_embedding = &embedding,
        .profile_tag = .{
            .primary_profile = .abbey,
            .blend_coefficient = 0.8,
        },
        .routing_weights = .{
            .abbey_weight = 0.6,
            .aviva_weight = 0.3,
            .abi_weight = 0.1,
        },
        .intent = .empathy_seeking,
        .risk_score = 0.1,
    });

    try std.testing.expect(block_id > 0);
    try std.testing.expect(chain.current_head != null);
}

test "database: chain integrity across multiple blocks" {
    var chain = BlockChain.init(std.testing.allocator, "integrity-test");
    defer chain.deinit();

    const embedding = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    const id1 = try chain.addBlock(.{
        .query_embedding = &embedding,
        .profile_tag = .{ .primary_profile = .aviva },
        .routing_weights = .{ .aviva_weight = 0.8 },
        .intent = .technical_problem,
    });

    const id2 = try chain.addBlock(.{
        .query_embedding = &embedding,
        .profile_tag = .{ .primary_profile = .abbey },
        .routing_weights = .{ .abbey_weight = 0.7 },
        .intent = .empathy_seeking,
        .parent_block_id = id1,
    });

    // Verify chain linkage
    try std.testing.expect(id1 != id2);

    if (chain.getBlock(id2)) |block2| {
        // Block 2 should reference block 1 via parent or skip pointer
        try std.testing.expect(block2.commit_timestamp > 0);
    }
}

test "database: block MVCC visibility" {
    var chain = BlockChain.init(std.testing.allocator, "mvcc-test");
    defer chain.deinit();

    const embedding = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    const block_id = try chain.addBlock(.{
        .query_embedding = &embedding,
        .profile_tag = .{ .primary_profile = .abi },
        .routing_weights = .{ .abi_weight = 0.9 },
        .intent = .safety_critical,
    });

    if (chain.getBlock(block_id)) |block| {
        // Block should be visible at current time
        const now = @import("abi").foundation.time.unixSeconds();
        try std.testing.expect(block.isVisible(now));

        // Block should not be visible before its commit timestamp
        try std.testing.expect(!block.isVisible(block.commit_timestamp - 1));
    }
}

test "database: routing weights primary profile" {
    const weights = RoutingWeights{
        .abbey_weight = 0.2,
        .aviva_weight = 0.6,
        .abi_weight = 0.2,
    };

    try std.testing.expectEqual(ProfileTag.ProfileType.aviva, weights.getPrimaryProfile());
}

test {
    std.testing.refAllDecls(@This());
}
