const std = @import("std");
const testing = std.testing;
const aps = @import("advanced_persona_system.zig");

test "ActiveAgentPool allocation and deallocation" {
    var pool = aps.AgentCoordinationSystem.ActiveAgentPool.init();
    
    // Test allocation
    const slot1 = pool.allocateSlot();
    try testing.expect(slot1 != null);
    try testing.expect(slot1.? < aps.AgentCoordinationSystem.MaxConcurrentAgents);
    
    const slot2 = pool.allocateSlot();
    try testing.expect(slot2 != null);
    try testing.expect(slot2.? != slot1.?);
    
    // Test release
    pool.releaseSlot(slot1.?);
    
    // Should be able to allocate the released slot again
    const slot3 = pool.allocateSlot();
    try testing.expect(slot3 != null);
}

test "ActiveAgentPool concurrent allocation stress test" {
    var pool = aps.AgentCoordinationSystem.ActiveAgentPool.init();
    const num_threads = 4;
    const allocations_per_thread = 5;
    
    const ThreadContext = struct {
        pool: *aps.AgentCoordinationSystem.ActiveAgentPool,
        results: []?usize,
        thread_id: usize,
    };
    
    var contexts: [num_threads]ThreadContext = undefined;
    var results: [num_threads][allocations_per_thread]?usize = undefined;
    
    // Initialize contexts
    for (&contexts, 0..) |*ctx, i| {
        ctx.* = .{
            .pool = &pool,
            .results = &results[i],
            .thread_id = i,
        };
    }
    
    // Run concurrent allocations
    const allocFn = struct {
        fn run(ctx: *ThreadContext) void {
            for (ctx.results) |*result| {
                result.* = ctx.pool.allocateSlot();
                std.time.sleep(1000); // 1 microsecond
            }
        }
    }.run;
    
    var threads: [num_threads]std.Thread = undefined;
    for (&threads, &contexts) |*thread, *ctx| {
        thread.* = try std.Thread.spawn(.{}, allocFn, .{ctx});
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all allocated slots are unique
    var seen = std.bit_set.IntegerBitSet(25).initEmpty();
    for (results) |thread_results| {
        for (thread_results) |slot_opt| {
            if (slot_opt) |slot| {
                try testing.expect(!seen.isSet(slot));
                seen.set(slot);
            }
        }
    }
}

test "AgentRegistry basic operations" {
    const allocator = testing.allocator;
    var registry = aps.AgentCoordinationSystem.AgentRegistry.init(allocator);
    defer registry.deinit();
    
    // Create and register an agent
    const expertise_domains = [_]aps.ExpertiseDomain{
        .{ .name = "programming", .proficiency = 0.9 },
        .{ .name = "debugging", .proficiency = 0.8 },
    };
    
    const def = aps.AgentDefinition{
        .expertise_domains = &expertise_domains,
    };
    
    const agent_id = try registry.registerAgent(def);
    try testing.expect(agent_id.value > 0);
    
    // Find agents by expertise
    const search_domains = [_]aps.ExpertiseDomain{
        .{ .name = "programming", .proficiency = 0.5 },
    };
    
    const found = try registry.findAgentsByExpertise(&search_domains, 10);
    defer allocator.free(found);
    
    try testing.expect(found.len > 0);
    try testing.expect(found[0].value == agent_id.value);
}

test "Agent processing" {
    const allocator = testing.allocator;
    
    const expertise_domains = [_]aps.ExpertiseDomain{
        .{ .name = "natural_language", .proficiency = 0.9 },
        .{ .name = "sentiment_analysis", .proficiency = 0.7 },
    };
    
    const def = aps.AgentDefinition{
        .expertise_domains = &expertise_domains,
    };
    
    const agent = try aps.Agent.fromDefinition(allocator, def);
    defer agent.deinit();
    
    const query = "Analyze this text";
    const context = aps.QueryContext{};
    
    const response = try agent.process(query, context);
    defer allocator.free(response);
    
    try testing.expect(response.len > 0);
    try testing.expect(std.mem.indexOf(u8, response, "Processing query:") != null);
    try testing.expect(std.mem.indexOf(u8, response, "natural_language") != null);
}

test "EmbeddingVector operations" {
    const allocator = testing.allocator;
    
    var vec1 = try aps.EmbeddingVector.init(allocator, 4);
    defer vec1.deinit();
    vec1.data[0] = 1.0;
    vec1.data[1] = 2.0;
    vec1.data[2] = 3.0;
    vec1.data[3] = 4.0;
    
    var vec2 = try aps.EmbeddingVector.init(allocator, 4);
    defer vec2.deinit();
    vec2.data[0] = 2.0;
    vec2.data[1] = 3.0;
    vec2.data[2] = 4.0;
    vec2.data[3] = 5.0;
    
    // Test normalization
    vec1.normalize();
    var sum: f32 = 0.0;
    for (vec1.data) |val| {
        sum += val * val;
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    
    // Test dot product
    vec2.normalize();
    const dot = vec1.dotProduct(vec2);
    try testing.expect(dot > 0.9); // Should be close to 1 for similar vectors
}

test "AgentCoordinationSystem query processing" {
    const allocator = testing.allocator;
    var system = aps.AgentCoordinationSystem.init(allocator);
    defer {
        // Clean up the system
        system.agent_registry.deinit();
    }
    
    // Register some test agents
    const expertise_sets = [_][]const aps.ExpertiseDomain{
        &[_]aps.ExpertiseDomain{
            .{ .name = "programming", .proficiency = 0.9 },
            .{ .name = "algorithms", .proficiency = 0.8 },
        },
        &[_]aps.ExpertiseDomain{
            .{ .name = "emotional_intelligence", .proficiency = 0.9 },
            .{ .name = "empathy", .proficiency = 0.85 },
        },
    };
    
    for (expertise_sets) |expertise| {
        const def = aps.AgentDefinition{ .expertise_domains = expertise };
        _ = try system.agent_registry.registerAgent(def);
    }
    
    // Process a query
    const query = "Help me understand this code";
    const context = aps.QueryContext{};
    const user_id = aps.UserId{};
    
    const result = try system.processQuery(query, context, user_id);
    defer allocator.free(result.response);
    
    try testing.expect(result.response.len > 0);
}

test "Error handling for invalid operations" {
    var pool = aps.AgentCoordinationSystem.ActiveAgentPool.init();
    
    // Test invalid slot release
    pool.releaseSlot(100); // Should not crash
    
    // Test setAgent with invalid slot
    const allocator = testing.allocator;
    const def = aps.AgentDefinition{ .expertise_domains = &.{} };
    const agent = try aps.Agent.fromDefinition(allocator, def);
    defer agent.deinit();
    
    try testing.expectError(error.InvalidSlotIndex, pool.setAgent(100, agent));
}

test "AgentId generation is unique" {
    const id1 = aps.AgentId.generate();
    const id2 = aps.AgentId.generate();
    const id3 = aps.AgentId.generate();
    
    try testing.expect(id1.value != id2.value);
    try testing.expect(id2.value != id3.value);
    try testing.expect(id1.value != id3.value);
    
    // Test hash function
    try testing.expect(id1.hash() == id1.value);
}

test "KDTree insertion and search" {
    const allocator = testing.allocator;
    var tree = aps.AgentCoordinationSystem.KDTree.init(allocator);
    defer tree.deinit();
    
    // Insert some points
    const points = [_]struct { vec: aps.EmbeddingVector, id: aps.AgentId }{
        .{ .vec = try createVector(allocator, &[_]f32{ 1.0, 2.0, 3.0 }), .id = aps.AgentId{ .value = 1 } },
        .{ .vec = try createVector(allocator, &[_]f32{ 4.0, 5.0, 6.0 }), .id = aps.AgentId{ .value = 2 } },
        .{ .vec = try createVector(allocator, &[_]f32{ 7.0, 8.0, 9.0 }), .id = aps.AgentId{ .value = 3 } },
    };
    defer for (points) |p| {
        p.vec.deinit();
    };
    
    for (points) |p| {
        try tree.insert(p.vec, p.id);
    }
    
    // Search for nearest neighbors
    var query = try createVector(allocator, &[_]f32{ 3.0, 4.0, 5.0 });
    defer query.deinit();
    
    const neighbors = try tree.kNearestNeighbors(query, 2);
    defer allocator.free(neighbors);
    
    try testing.expect(neighbors.len <= 2);
    try testing.expect(neighbors.len > 0);
}

fn createVector(allocator: std.mem.Allocator, values: []const f32) !aps.EmbeddingVector {
    var vec = try aps.EmbeddingVector.init(allocator, values.len);
    @memcpy(vec.data, values);
    return vec;
}