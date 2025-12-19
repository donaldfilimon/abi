// Tests for Reinforcement Learning Module

const std = @import("std");
const ai = @import("abi").ai;

/// Test RL agent basic functionality
test "reinforcement learning agent basic operations" {
    const testing = std.testing;
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 100, 4);
    defer agent.deinit();

    // Test initial state
    try testing.expectEqual(@as(usize, 100), agent.state_size);
    try testing.expectEqual(@as(usize, 4), agent.action_size);
    try testing.expectEqual(ai.reinforcement_learning.Policy.epsilon_greedy, agent.policy);
    try testing.expectEqual(@as(f32, 0.1), agent.epsilon);

    // Test action selection
    const state = [_]f32{0.5, 0.3};
    const action = agent.act(&state);
    try testing.expect(action < 4); // Valid action range
}

/// Test Q-learning updates
test "reinforcement learning Q-learning" {
    const testing = std.testing;
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 10, 2);
    defer agent.deinit();

    // Create a simple experience
    const experience = ai.reinforcement_learning.Experience{
        .state = &[_]f32{0.0},
        .action = 0,
        .reward = 1.0,
        .next_state = &[_]f32{0.1},
        .done = false,
    };

    // Get Q-value before learning
    const q_before = agent.getQValue(0, 0);

    // Learn from experience
    try agent.learn(experience);

    // Q-value should change
    const q_after = agent.getQValue(0, 0);
    try testing.expect(q_before != q_after);
}

/// Test epsilon-greedy policy
test "reinforcement learning epsilon-greedy policy" {
    const testing = std.testing;

    // Test with epsilon = 1.0 (always explore)
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 10, 4);
    defer agent.deinit();
    agent.epsilon = 1.0;

    // With epsilon=1, should select random actions
    const state = [_]f32{0.0};
    var action_counts = [_]usize{0} ** 4;

    // Sample many actions to check randomness
    for (0..100) |_| {
        const action = agent.act(&state);
        action_counts[action] += 1;
    }

    // All actions should be selected at least once (with high probability)
    for (action_counts) |count| {
        try testing.expect(count > 0);
    }
}

/// Test greedy policy
test "reinforcement learning greedy policy" {
    const testing = std.testing;
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 10, 4);
    defer agent.deinit();

    // Set epsilon to 0 for greedy policy
    agent.epsilon = 0.0;

    // Manually set Q-values to favor action 2
    agent.setQValue(0, 2, 1.0);
    agent.setQValue(0, 0, 0.1);
    agent.setQValue(0, 1, 0.2);
    agent.setQValue(0, 3, 0.3);

    const state = [_]f32{0.0};
    const action = agent.act(&state);

    // Should always select action 2 (highest Q-value)
    try testing.expectEqual(@as(usize, 2), action);
}

/// Test softmax policy
test "reinforcement learning softmax policy" {
    const testing = std.testing;
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 10, 4);
    defer agent.deinit();

    agent.policy = .softmax;

    // Set Q-values with clear preference for action 0
    agent.setQValue(0, 0, 2.0);
    agent.setQValue(0, 1, 0.0);
    agent.setQValue(0, 2, 0.0);
    agent.setQValue(0, 3, 0.0);

    const state = [_]f32{0.0};

    // Sample many actions - should prefer action 0
    var action_counts = [_]usize{0} ** 4;
    for (0..1000) |_| {
        const action = agent.act(&state);
        action_counts[action] += 1;
    }

    // Action 0 should be selected most frequently
    try testing.expect(action_counts[0] > action_counts[1]);
    try testing.expect(action_counts[0] > action_counts[2]);
    try testing.expect(action_counts[0] > action_counts[3]);
}

/// Test GridWorld environment
test "reinforcement learning grid world" {
    const testing = std.testing;
    var env = try ai.reinforcement_learning.GridWorld.init(testing.allocator, 5, 5);
    defer env.deinit();

    // Test basic properties
    try testing.expectEqual(@as(usize, 25), env.getStateSize()); // 5x5 grid
    try testing.expectEqual(@as(usize, 4), env.getActionSize()); // 4 actions

    // Test reset
    const start_pos = env.reset();
    try testing.expect(start_pos[0] >= 0 and start_pos[0] < 5);
    try testing.expect(start_pos[1] >= 0 and start_pos[1] < 5);

    // Test step function
    const result = env.step(start_pos, 0); // Move up
    try testing.expect(result.done == (result.next_pos[0] == env.goal_pos[0] and
                                      result.next_pos[1] == env.goal_pos[1]));
    try testing.expect(result.reward <= 1.0 and result.reward >= -0.01);
}

/// Test agent-environment interaction
test "reinforcement learning agent-environment integration" {
    const testing = std.testing;
    var env = try ai.reinforcement_learning.GridWorld.init(testing.allocator, 3, 3);
    defer env.deinit();

    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 9, 4);
    defer agent.deinit();

    // Run a simple episode
    var state = env.reset();
    var total_reward: f32 = 0.0;
    var steps: usize = 0;

    while (steps < 10) { // Prevent infinite loops
        // Convert position to simple state representation
        const state_idx = state[0] * 3 + state[1];
        const simple_state = [_]f32{@as(f32, @floatFromInt(state_idx)) / 9.0};

        const action = agent.act(&simple_state);
        const result = env.step(state, action);

        // Create experience
        const next_state_idx = result.next_pos[0] * 3 + result.next_pos[1];
        const next_simple_state = [_]f32{@as(f32, @floatFromInt(next_state_idx)) / 9.0};

        const experience = ai.reinforcement_learning.Experience{
            .state = &simple_state,
            .action = action,
            .reward = result.reward,
            .next_state = &next_simple_state,
            .done = result.done,
        };

        try agent.learn(experience);

        total_reward += result.reward;
        state = result.next_pos;
        steps += 1;

        if (result.done) break;
    }

    // Should have accumulated some reward
    try testing.expect(total_reward >= -1.0); // At least not completely broken
    try testing.expect(steps > 0);
}

/// Test agent memory management
test "reinforcement learning agent memory management" {
    const testing = std.testing;

    // Test with large state space
    var agent = try ai.reinforcement_learning.Agent.init(testing.allocator, 1000, 10);
    defer agent.deinit();

    // Verify Q-table size
    try testing.expectEqual(@as(usize, 1000 * 10), agent.q_table.items.len);

    // Test multiple learning steps
    for (0..100) |_| {
        const experience = ai.reinforcement_learning.Experience{
            .state = &[_]f32{0.5},
            .action = 0,
            .reward = 0.1,
            .next_state = &[_]f32{0.6},
            .done = false,
        };
        try agent.learn(experience);
    }

    // Q-value should have changed
    const q_value = agent.getQValue(5, 0); // Approximate state index
    try testing.expect(q_value != 0.0);
}