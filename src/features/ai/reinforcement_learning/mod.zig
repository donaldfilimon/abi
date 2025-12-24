//! Reinforcement Learning Module
//!
//! Implements reinforcement learning algorithms for training agents through interaction with environments.
//!
//! ## Supported Algorithms
//!
//! - **Q-Learning**: Model-free reinforcement learning using Q-value estimation
//! - **SARSA**: On-policy temporal difference learning
//! - **Policy Gradient**: Direct policy optimization methods
//!
//! ## Key Concepts
//!
//! - **Agent**: Learns to make decisions by interacting with an environment
//! - **Environment**: Provides states, rewards, and transitions
//! - **Policy**: Mapping from states to actions
//! - **Value Function**: Estimates expected future rewards
//!
//! ## Usage Example
//!
//! ```zig
//! // Create a Q-learning agent
//! var agent = try QLearningAgent.init(allocator, state_size, action_count, .{});
//! defer agent.deinit();
//!
//! // Train by interacting with environment
//! for (0..num_episodes) |_| {
//!     var state = env.reset();
//!     while (!done) {
//!         const action = agent.chooseAction(state);
//!         const result = env.step(action);
//!         agent.learn(state, action, result.reward, result.next_state);
//!         state = result.next_state;
//!     }
//! }
//! ```

const std = @import("std");

/// Reinforcement learning algorithm types
pub const RLAlgorithm = enum {
    /// Q-Learning: Off-policy temporal difference learning
    q_learning,
    /// SARSA: On-policy temporal difference learning
    sarsa,
    /// Policy Gradient: Direct policy optimization
    policy_gradient,
};

/// Environment interface for RL agents
///
/// This interface defines how reinforcement learning agents interact with
/// environments. Environments must provide methods to reset state, take actions,
/// and report available actions.
pub const Environment = struct {
    /// Function pointer to reset environment to initial state
    /// Returns: Initial state vector
    resetFn: *const fn (*anyopaque) []f32,
    /// Function pointer to take an action and advance environment
    /// Parameters: context, action_index
    /// Returns: struct{state: next state vector, reward: scalar reward, done: episode finished flag}
    stepFn: *const fn (*anyopaque, usize) struct { state: []f32, reward: f32, done: bool },
    /// Function pointer to get number of possible actions
    /// Returns: Number of valid actions in this environment
    getActionCountFn: *const fn (*const anyopaque) usize,
    /// Opaque context pointer for environment implementation
    context: *anyopaque,

    pub fn reset(self: *Environment) []f32 {
        return self.resetFn(self.context);
    }

    pub fn step(self: *Environment, action: usize) struct { state: []f32, reward: f32, done: bool } {
        return self.stepFn(self.context, action);
    }

    pub fn getActionCount(self: *const Environment) usize {
        return self.getActionCountFn(self.context);
    }
};

/// Q-Learning agent
pub const QLearningAgent = struct {
    const Self = @This();

    q_table: std.ArrayList(f32),
    state_size: usize,
    action_count: usize,
    learning_rate: f32,
    discount_factor: f32,
    epsilon: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, state_size: usize, action_count: usize, config: struct {
        learning_rate: f32 = 0.1,
        discount_factor: f32 = 0.99,
        epsilon: f32 = 0.1,
    }) !QLearningAgent {
        const table_size = state_size * action_count;
        var q_table = try std.ArrayList(f32).initCapacity(allocator, table_size);
        q_table.expandToCapacity();
        @memset(q_table.items, 0.0);

        return QLearningAgent{
            .q_table = q_table,
            .state_size = state_size,
            .action_count = action_count,
            .learning_rate = config.learning_rate,
            .discount_factor = config.discount_factor,
            .epsilon = config.epsilon,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.q_table.deinit();
    }

    /// Get Q-value for state-action pair
    pub fn getQValue(self: *const Self, state: usize, action: usize) f32 {
        const index = state * self.action_count + action;
        return self.q_table.items[index];
    }

    /// Set Q-value for state-action pair
    pub fn setQValue(self: *Self, state: usize, action: usize, value: f32) void {
        const index = state * self.action_count + action;
        self.q_table.items[index] = value;
    }

    /// Choose action using epsilon-greedy policy
    pub fn chooseAction(self: *Self, state: usize) usize {
        // Epsilon-greedy exploration
        if (std.crypto.random.float(f32) < self.epsilon) {
            return std.crypto.random.uintLessThan(usize, self.action_count);
        }

        // Greedy action selection
        var best_action: usize = 0;
        var best_value = self.getQValue(state, 0);

        for (1..self.action_count) |action| {
            const value = self.getQValue(state, action);
            if (value > best_value) {
                best_value = value;
                best_action = action;
            }
        }

        return best_action;
    }

    /// Update Q-value using Q-learning update rule
    pub fn learn(self: *Self, state: usize, action: usize, reward: f32, next_state: usize) void {
        const current_q = self.getQValue(state, action);

        // Find maximum Q-value for next state
        var max_next_q = self.getQValue(next_state, 0);
        for (1..self.action_count) |a| {
            const q = self.getQValue(next_state, a);
            if (q > max_next_q) max_next_q = q;
        }

        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        const target = reward + self.discount_factor * max_next_q;
        const new_q = current_q + self.learning_rate * (target - current_q);

        self.setQValue(state, action, new_q);
    }
};

test "Q-Learning agent initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var agent = try QLearningAgent.init(allocator, 10, 4, .{});
    defer agent.deinit();

    try testing.expectEqual(@as(usize, 40), agent.q_table.items.len); // 10 states * 4 actions
}

test "Q-Learning action selection" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var agent = try QLearningAgent.init(allocator, 5, 3, .{ .epsilon = 0.0 }); // No exploration
    defer agent.deinit();

    // Set Q-values manually
    agent.setQValue(0, 0, 0.5);
    agent.setQValue(0, 1, 0.8);
    agent.setQValue(0, 2, 0.3);

    const action = agent.chooseAction(0);
    try testing.expectEqual(@as(usize, 1), action); // Should choose action with highest Q-value
}

test "Q-Learning learning update" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var agent = try QLearningAgent.init(allocator, 3, 2, .{
        .learning_rate = 0.1,
        .discount_factor = 0.9,
    });
    defer agent.deinit();

    // Set initial Q-values
    agent.setQValue(0, 0, 1.0);
    agent.setQValue(1, 0, 2.0);

    // Learn from experience: state=0, action=0, reward=1.0, next_state=1
    agent.learn(0, 0, 1.0, 1);

    // Q(0,0) should be updated: 1.0 + 0.1 * (1.0 + 0.9 * 2.0 - 1.0) = 1.18
    const expected_q = 1.0 + 0.1 * (1.0 + 0.9 * 2.0 - 1.0);
    try testing.expectApproxEqAbs(expected_q, agent.getQValue(0, 0), 0.001);
}

test "Q-Learning exploration vs exploitation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // High exploration rate
    var agent = try QLearningAgent.init(allocator, 2, 2, .{ .epsilon = 1.0 });
    defer agent.deinit();

    // With epsilon=1.0, should explore (random action)
    // This is probabilistic, so we run multiple times to check distribution
    var action_counts = [_]usize{ 0, 0 };
    for (0..100) |_| {
        const action = agent.chooseAction(0);
        action_counts[action] += 1;
    }

    // Both actions should be chosen approximately equally
    try testing.expect(action_counts[0] > 20);
    try testing.expect(action_counts[1] > 20);
}
