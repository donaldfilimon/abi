const std = @import("std");
const testing = std.testing;

const ai = @import("../mod.zig");

test "federated coordinator init and deinit" {
    var coordinator = try ai.federated.FederatedCoordinator.init(testing.allocator, 0);
    defer coordinator.deinit();

    try testing.expect(coordinator.rounds == 0);
}

test "optimization SGD step" {
    var optimizer = try ai.optimization.create(testing.allocator, .sgd, 0.01);
    defer ai.optimization.destroy(testing.allocator, optimizer);

    var params = [_]f32{ 1.0, 2.0 };
    var grads = [_]f32{ 0.1, 0.2 };

    optimizer.step(&params, &grads);

    try testing.expect(params[0] < 1.0); // Should decrease
}

test "reinforcement learning agent select action" {
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, .q_learning, 2, 2);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.0 };
    const action = agent.selectAction(&state, std.Random.DefaultPrng.init(42).random());

    try testing.expect(action < 2);
}

test "transformer forward pass" {
    var transformer = try ai.transformer.Transformer.init(testing.allocator, .{ .num_layers = 2, .d_model = 64, .num_heads = 4, .d_ff = 128 }, 1000);
    defer transformer.deinit(testing.allocator);

    var token_ids = [_]usize{ 1, 2, 3 };
    const output = try transformer.forward(&token_ids, testing.allocator);
    defer testing.allocator.free(output);

    try testing.expect(output.len == token_ids.len * 64);
}
