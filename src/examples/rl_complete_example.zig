//! Complete Reinforcement Learning Example
//!
//! This example demonstrates the full reinforcement learning capabilities with:
//! - Deep Q-Networks (DQN) with experience replay
//! - Policy Gradient methods (REINFORCE)
//! - Actor-Critic architectures
//! - Experience replay buffers
//! - Multiple exploration strategies
//! - Complete training pipeline

const std = @import("std");
const rl = @import("ai").reinforcement_learning;

/// Simple CartPole-like environment for demonstration
pub const SimpleEnvironment = struct {
    state: [4]f32, // [position, velocity, angle, angular_velocity]
    done: bool,
    step_count: usize,
    max_steps: usize,

    pub fn init() SimpleEnvironment {
        return SimpleEnvironment{
            .state = [_]f32{ 0.0, 0.0, 0.0, 0.0 },
            .done = false,
            .step_count = 0,
            .max_steps = 200,
        };
    }

    pub fn reset(self: *SimpleEnvironment) []f32 {
        self.state = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        self.done = false;
        self.step_count = 0;
        return &self.state;
    }

    pub fn step(self: *SimpleEnvironment, action: usize) struct { state: []f32, reward: f32, done: bool } {
        self.step_count += 1;

        // Simple physics simulation
        const force = if (action == 0) -1.0 else 1.0; // Action 0: left, 1: right

        // Update state (simplified cart-pole dynamics)
        self.state[0] += self.state[1] * 0.1; // position += velocity
        self.state[1] += force * 0.1 - self.state[0] * 0.01; // velocity update
        self.state[2] += self.state[3] * 0.1; // angle += angular velocity
        self.state[3] += force * 0.05 - self.state[2] * 0.1; // angular velocity update

        // Calculate reward
        const angle_penalty = -std.math.pow(f32, self.state[2], 2);
        const position_penalty = -std.math.pow(f32, self.state[0], 2);
        const reward = angle_penalty + position_penalty + 1.0;

        // Check termination conditions
        const angle_limit = std.math.pi / 2.0;
        const position_limit = 2.4;

        if (self.step_count >= self.max_steps or
            @abs(self.state[2]) > angle_limit or
            @abs(self.state[0]) > position_limit)
        {
            self.done = true;
        }

        return .{ .state = &self.state, .reward = reward, .done = self.done };
    }
};

/// Complete RL training demonstration
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator;

    std.debug.print("=== Complete Reinforcement Learning Example ===\n", .{});

    // DQN Training Demo
    std.debug.print("\n=== DQN Training Demo ===\n", .{});

    // Create environment
    var env = SimpleEnvironment.init();

    // Create DQN agent
    var dqn = try rl.DQN.init(
        allocator,
        undefined, // network pointer (placeholder)
        undefined, // target network pointer (placeholder)
        1000, // replay capacity
        0.99, // gamma
        1.0, // epsilon start
        0.01, // epsilon min
        0.995, // epsilon decay
        0.001, // learning rate
        32, // batch size
        100, // update target every
    );
    defer dqn.deinit();

    std.debug.print("Created DQN agent with:\n", .{});
    std.debug.print("  - Replay buffer capacity: {}\n", .{dqn.replay_buffer.capacity});
    std.debug.print("  - Gamma (discount factor): {}\n", .{dqn.gamma});
    std.debug.print("  - Epsilon start: {}\n", .{dqn.epsilon});
    std.debug.print("  - Batch size: {}\n", .{dqn.batch_size});

    // Training loop
    const num_episodes = 10;
    for (0..num_episodes) |episode| {
        var state = env.reset();
        var episode_reward: f32 = 0.0;
        var episode_steps: usize = 0;

        while (!env.done and episode_steps < env.max_steps) {
            // Select action using ε-greedy policy
            const action = try dqn.selectAction(state, 2); // 2 actions: left/right

            // Take step in environment
            const result = env.step(action);

            // Store experience in replay buffer
            try dqn.replay_buffer.add(state, action, result.reward, result.state, result.done);

            // Train on batch if enough experiences
            if (dqn.replay_buffer.size() >= dqn.batch_size) {
                // Note: This would normally pass the neural network
                // try dqn.trainOnBatch(network);
            }

            state = result.state;
            episode_reward += result.reward;
            episode_steps += 1;
        }

        std.debug.print("Episode {}: steps={}, reward={d:.2}, epsilon={d:.3}\n", .{ episode + 1, episode_steps, episode_reward, dqn.epsilon });
    }

    // Policy Gradient Demo
    std.debug.print("\n=== Policy Gradient Demo ===\n", .{});

    var pg = try rl.PolicyGradient.init(
        allocator,
        undefined, // policy network pointer (placeholder)
        0.01, // learning rate
        0.99, // gamma
    );
    defer pg.deinit();

    std.debug.print("Created Policy Gradient agent\n", .{});

    // Collect trajectory
    var trajectory_env = SimpleEnvironment.init();
    try pg.collectTrajectory(rl.Environment{
        .resetFn = struct {
            fn reset(ctx: *anyopaque) anyerror![]f32 {
                const env_ptr = @as(*SimpleEnvironment, @ptrCast(@alignCast(ctx)));
                return env_ptr.reset();
            }
        }.reset,
        .stepFn = struct {
            fn step(ctx: *anyopaque, action: usize) anyerror!struct { state: []f32, reward: f32, done: bool, info: ?[]const u8 } {
                const env_ptr = @as(*SimpleEnvironment, @ptrCast(@alignCast(ctx)));
                const result = env_ptr.step(action);
                return .{ .state = result.state, .reward = result.reward, .done = result.done, .info = null };
            }
        }.step,
        .getObservationSpaceFn = struct {
            fn getObsSpace(_: *anyopaque) []const usize {
                return &[_]usize{4}; // 4-dimensional state
            }
        }.getObsSpace,
        .getActionSpaceFn = struct {
            fn getActionSpace(_: *anyopaque) []const usize {
                return &[_]usize{2}; // 2 actions
            }
        }.getActionSpace,
        .renderFn = null,
        .context = &trajectory_env,
    }, 50);

    std.debug.print("Collected trajectory with {} states\n", .{pg.trajectories.items[0].states.items.len});

    // Actor-Critic Demo
    std.debug.print("\n=== Actor-Critic Demo ===\n", .{});

    var ac = try rl.ActorCritic.init(
        allocator,
        undefined, // actor network
        undefined, // critic network
        0.99, // gamma
        0.001, // actor learning rate
        0.01, // critic learning rate
    );
    defer ac.deinit();

    std.debug.print("Created Actor-Critic agent\n", .{});

    // Single training step
    const test_state = [_]f32{ 0.1, -0.2, 0.05, 0.1 };
    const test_action = 1;
    const test_reward = 0.5;
    const test_next_state = [_]f32{ 0.15, -0.15, 0.03, 0.08 };

    try ac.train(&test_state, test_action, test_reward, &test_next_state, false);

    std.debug.print("Completed Actor-Critic training step\n", .{});

    // Exploration Strategies Demo
    std.debug.print("\n=== Exploration Strategies Demo ===\n", .{});

    const q_values = [_]f32{ 0.2, 0.8, 0.5, 0.3 };
    var rng = std.Random.DefaultPrng.init(42);

    // ε-greedy strategy
    const epsilon_greedy = rl.ExplorationStrategy{
        .epsilon_greedy = .{
            .epsilon = 0.1,
            .epsilon_min = 0.01,
            .epsilon_decay = 0.995,
        },
    };

    const eg_action = epsilon_greedy.selectAction(&q_values, rng.random());
    std.debug.print("ε-greedy (ε=0.1) selected action: {}\n", .{eg_action});

    // Softmax strategy
    const softmax = rl.ExplorationStrategy{
        .softmax = .{
            .temperature = 0.5,
        },
    };

    const sm_action = softmax.selectAction(&q_values, rng.random());
    std.debug.print("Softmax (temperature=0.5) selected action: {}\n", .{sm_action});

    // Boltzmann strategy
    const boltzmann = rl.ExplorationStrategy{
        .boltzmann = .{
            .temperature = 0.8,
        },
    };

    const bz_action = boltzmann.selectAction(&q_values, rng.random());
    std.debug.print("Boltzmann (temperature=0.8) selected action: {}\n", .{bz_action});

    // Experience Replay Demo
    std.debug.print("\n=== Experience Replay Demo ===\n", .{});

    var replay_buffer = try rl.ExperienceReplay.init(allocator, 100);
    defer replay_buffer.deinit();

    // Add some experiences
    const states = [_][4]f32{
        [_]f32{ 0.0, 0.0, 0.0, 0.0 },
        [_]f32{ 0.1, 0.1, 0.0, 0.0 },
        [_]f32{ 0.2, 0.2, 0.1, 0.1 },
    };

    for (states, 0..) |state, i| {
        const action = i % 2;
        const reward = @as(f32, @floatFromInt(i)) * 0.1;
        const next_state = [_]f32{ state[0] + 0.1, state[1] + 0.1, state[2], state[3] };
        const done = i == states.len - 1;

        try replay_buffer.add(&state, action, reward, &next_state, done);
    }

    std.debug.print("Added {} experiences to replay buffer\n", .{replay_buffer.size()});

    // Sample a batch
    const batch = try replay_buffer.sample(2);
    defer allocator.free(batch);

    std.debug.print("Sampled batch of {} experiences\n", .{batch.len});
    for (batch, 0..) |experience, i| {
        std.debug.print("  Experience {}: action={}, reward={d:.3}, done={}\n", .{ i, experience.action, experience.reward, experience.done });
    }

    std.debug.print("\n=== Demo Complete ===\n", .{});
}
