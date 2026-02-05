//! Abbey Online Learning System
//!
//! Enables real-time adaptation and learning during conversations.
//! Implements various online learning algorithms and experience replay.

const std = @import("std");
const tensor = @import("tensor.zig");
const layer_mod = @import("layer.zig");
const types = @import("../../core/types.zig");

const F32Tensor = tensor.F32Tensor;

/// Error set for replay buffer operations.
pub const ReplayError = error{
    /// Not enough samples available for the requested batch size.
    InsufficientData,
} || std.mem.Allocator.Error;

/// Error set for online learning operations.
pub const LearningError = ReplayError || layer_mod.LayerError;

// ============================================================================
// Learning Experience
// ============================================================================

/// A single learning experience (state, action, reward, next_state)
pub const Experience = struct {
    state: F32Tensor,
    action: F32Tensor,
    reward: f32,
    next_state: ?F32Tensor,
    done: bool,
    timestamp: i64,
    metadata: ExperienceMetadata,

    pub const ExperienceMetadata = struct {
        emotion: u8 = 0,
        confidence: f32 = 0.5,
        topic_hash: u64 = 0,
        user_feedback: ?f32 = null,
    };

    pub fn deinit(self: *Experience, allocator: std.mem.Allocator) void {
        self.state.deinit();
        self.action.deinit();
        if (self.next_state) |*ns| ns.deinit();
        _ = allocator;
    }
};

// ============================================================================
// Experience Replay Buffer
// ============================================================================

/// Circular buffer for storing and sampling experiences
pub const ReplayBuffer = struct {
    allocator: std.mem.Allocator,
    buffer: []Experience,
    capacity: usize,
    size: usize = 0,
    position: usize = 0,
    priority_weights: ?[]f32 = null,
    use_priority: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: usize, use_priority: bool) !Self {
        const buffer = try allocator.alloc(Experience, capacity);
        var priority_weights: ?[]f32 = null;
        if (use_priority) {
            priority_weights = try allocator.alloc(f32, capacity);
            @memset(priority_weights.?, 1.0);
        }

        return Self{
            .allocator = allocator,
            .buffer = buffer,
            .capacity = capacity,
            .priority_weights = priority_weights,
            .use_priority = use_priority,
        };
    }

    pub fn deinit(self: *Self) void {
        for (0..self.size) |i| {
            self.buffer[i].deinit(self.allocator);
        }
        self.allocator.free(self.buffer);
        if (self.priority_weights) |pw| {
            self.allocator.free(pw);
        }
    }

    /// Add an experience to the buffer
    pub fn add(self: *Self, experience: Experience) void {
        if (self.size == self.capacity) {
            // Overwrite oldest
            self.buffer[self.position].deinit(self.allocator);
        }

        self.buffer[self.position] = experience;
        if (self.priority_weights) |pw| {
            // New experiences get max priority
            var max_priority: f32 = 1.0;
            for (0..self.size) |i| {
                if (pw[i] > max_priority) max_priority = pw[i];
            }
            pw[self.position] = max_priority;
        }

        self.position = (self.position + 1) % self.capacity;
        if (self.size < self.capacity) self.size += 1;
    }

    /// Sample a batch of experiences
    pub fn sample(self: *Self, batch_size: usize) ReplayError![]usize {
        if (self.size < batch_size) return error.InsufficientData;

        var indices = try self.allocator.alloc(usize, batch_size);
        errdefer self.allocator.free(indices);

        var prng = std.Random.DefaultPrng.init(@intCast(types.getTimestampNs() & 0xFFFFFFFFFFFFFFFF));
        const rand = prng.random();

        if (self.use_priority and self.priority_weights != null) {
            // Priority-based sampling
            const pw = self.priority_weights.?;
            var total: f32 = 0;
            for (0..self.size) |i| total += pw[i];

            for (0..batch_size) |b| {
                const target = rand.float(f32) * total;
                var cumsum: f32 = 0;
                for (0..self.size) |i| {
                    cumsum += pw[i];
                    if (cumsum >= target) {
                        indices[b] = i;
                        break;
                    }
                }
            }
        } else {
            // Uniform sampling
            for (0..batch_size) |b| {
                indices[b] = rand.uintLessThan(usize, self.size);
            }
        }

        return indices;
    }

    /// Update priorities based on TD errors
    pub fn updatePriorities(self: *Self, indices: []const usize, errors: []const f32, alpha: f32) void {
        if (self.priority_weights == null) return;

        const pw = self.priority_weights.?;
        for (indices, 0..) |idx, i| {
            pw[idx] = std.math.pow(f32, @abs(errors[i]) + 0.01, alpha);
        }
    }

    /// Get buffer statistics
    pub fn getStats(self: *const Self) struct { size: usize, capacity: usize, fill_ratio: f32 } {
        return .{
            .size = self.size,
            .capacity = self.capacity,
            .fill_ratio = @as(f32, @floatFromInt(self.size)) / @as(f32, @floatFromInt(self.capacity)),
        };
    }
};

// ============================================================================
// Optimizer
// ============================================================================

/// SGD with momentum
pub const SGDOptimizer = struct {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: std.AutoHashMapUnmanaged(usize, F32Tensor),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, learning_rate: f32, momentum: f32, weight_decay: f32) Self {
        return Self{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .momentum = momentum,
            .weight_decay = weight_decay,
            .velocities = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.velocities.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.velocities.deinit(self.allocator);
    }

    pub fn step(self: *Self, param_id: usize, param: *F32Tensor, grad: *const F32Tensor) !void {
        // Get or create velocity
        const vel_result = try self.velocities.getOrPut(self.allocator, param_id);
        if (!vel_result.found_existing) {
            vel_result.value_ptr.* = try F32Tensor.zeros(self.allocator, param.shape);
        }
        var velocity = vel_result.value_ptr;

        // v = momentum * v - lr * grad
        // p = p + v - weight_decay * p
        for (0..param.data.len) |i| {
            velocity.data[i] = self.momentum * velocity.data[i] - self.learning_rate * grad.data[i];
            param.data[i] += velocity.data[i] - self.weight_decay * param.data[i];
        }
    }
};

/// Adam optimizer
pub const AdamOptimizer = struct {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step_count: usize = 0,
    m: std.AutoHashMapUnmanaged(usize, F32Tensor), // First moment
    v: std.AutoHashMapUnmanaged(usize, F32Tensor), // Second moment
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) Self {
        return Self{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .m = .{},
            .v = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var mit = self.m.iterator();
        while (mit.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.m.deinit(self.allocator);

        var vit = self.v.iterator();
        while (vit.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.v.deinit(self.allocator);
    }

    pub fn step(self: *Self, param_id: usize, param: *F32Tensor, grad: *const F32Tensor) !void {
        self.step_count += 1;

        // Get or create moments
        const m_result = try self.m.getOrPut(self.allocator, param_id);
        if (!m_result.found_existing) {
            m_result.value_ptr.* = try F32Tensor.zeros(self.allocator, param.shape);
        }
        var m_vec = m_result.value_ptr;

        const v_result = try self.v.getOrPut(self.allocator, param_id);
        if (!v_result.found_existing) {
            v_result.value_ptr.* = try F32Tensor.zeros(self.allocator, param.shape);
        }
        var v_vec = v_result.value_ptr;

        // Bias correction
        const bc1 = 1.0 - std.math.pow(f32, self.beta1, @as(f32, @floatFromInt(self.step_count)));
        const bc2 = 1.0 - std.math.pow(f32, self.beta2, @as(f32, @floatFromInt(self.step_count)));

        for (0..param.data.len) |i| {
            const g = grad.data[i];

            // Update biased first moment estimate
            m_vec.data[i] = self.beta1 * m_vec.data[i] + (1.0 - self.beta1) * g;

            // Update biased second moment estimate
            v_vec.data[i] = self.beta2 * v_vec.data[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            const m_hat = m_vec.data[i] / bc1;
            const v_hat = v_vec.data[i] / bc2;

            // Update parameters
            param.data[i] -= self.learning_rate * (m_hat / (@sqrt(v_hat) + self.epsilon) + self.weight_decay * param.data[i]);
        }
    }
};

// ============================================================================
// Loss Functions
// ============================================================================

pub const LossFn = struct {
    /// Mean Squared Error
    pub fn mse(predictions: *const F32Tensor, targets: *const F32Tensor) f32 {
        var sum: f32 = 0;
        for (0..predictions.data.len) |i| {
            const diff = predictions.data[i] - targets.data[i];
            sum += diff * diff;
        }
        return sum / @as(f32, @floatFromInt(predictions.data.len));
    }

    /// MSE gradient
    pub fn mseGrad(allocator: std.mem.Allocator, predictions: *const F32Tensor, targets: *const F32Tensor) !F32Tensor {
        var grad = try F32Tensor.init(allocator, predictions.shape);
        const n = @as(f32, @floatFromInt(predictions.data.len));

        for (0..predictions.data.len) |i| {
            grad.data[i] = 2.0 * (predictions.data[i] - targets.data[i]) / n;
        }

        return grad;
    }

    /// Cross-entropy loss
    pub fn crossEntropy(predictions: *const F32Tensor, targets: *const F32Tensor) f32 {
        var sum: f32 = 0;
        for (0..predictions.data.len) |i| {
            const p = @max(predictions.data[i], 1e-7);
            sum -= targets.data[i] * @log(p);
        }
        return sum / @as(f32, @floatFromInt(predictions.data.len));
    }

    /// Huber loss (smooth L1)
    pub fn huber(predictions: *const F32Tensor, targets: *const F32Tensor, delta: f32) f32 {
        var sum: f32 = 0;
        for (0..predictions.data.len) |i| {
            const diff = @abs(predictions.data[i] - targets.data[i]);
            if (diff <= delta) {
                sum += 0.5 * diff * diff;
            } else {
                sum += delta * (diff - 0.5 * delta);
            }
        }
        return sum / @as(f32, @floatFromInt(predictions.data.len));
    }
};

// ============================================================================
// Online Learning Manager
// ============================================================================

/// Manages online learning for Abbey
pub const OnlineLearner = struct {
    allocator: std.mem.Allocator,
    replay_buffer: ReplayBuffer,
    optimizer: AdamOptimizer,
    batch_size: usize,
    learning_rate: f32,
    gamma: f32, // Discount factor
    update_interval: usize,
    updates_since_last: usize = 0,
    total_updates: usize = 0,
    running_loss: f32 = 0.0,
    loss_history: std.ArrayListUnmanaged(f32),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: LearnerConfig) !Self {
        var replay = try ReplayBuffer.init(allocator, config.buffer_size, config.use_priority);
        errdefer replay.deinit();

        const optimizer = AdamOptimizer.init(
            allocator,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
            config.weight_decay,
        );

        return Self{
            .allocator = allocator,
            .replay_buffer = replay,
            .optimizer = optimizer,
            .batch_size = config.batch_size,
            .learning_rate = config.learning_rate,
            .gamma = config.gamma,
            .update_interval = config.update_interval,
            .loss_history = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.replay_buffer.deinit();
        self.optimizer.deinit();
        self.loss_history.deinit(self.allocator);
    }

    pub const LearnerConfig = struct {
        buffer_size: usize = 10000,
        batch_size: usize = 32,
        learning_rate: f32 = 0.001,
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        epsilon: f32 = 1e-8,
        weight_decay: f32 = 0.0001,
        gamma: f32 = 0.99,
        update_interval: usize = 4,
        use_priority: bool = true,
    };

    /// Add a new experience
    pub fn addExperience(self: *Self, experience: Experience) void {
        self.replay_buffer.add(experience);
        self.updates_since_last += 1;
    }

    /// Check if we should update
    pub fn shouldUpdate(self: *const Self) bool {
        return self.updates_since_last >= self.update_interval and
            self.replay_buffer.size >= self.batch_size;
    }

    /// Perform a learning update.
    /// The model_forward callback uses LayerError for the forward pass surface.
    pub fn update(
        self: *Self,
        model_forward: *const fn (*const F32Tensor) layer_mod.LayerError!F32Tensor,
    ) LearningError!f32 {
        if (!self.shouldUpdate()) return 0.0;

        const indices = try self.replay_buffer.sample(self.batch_size);
        defer self.allocator.free(indices);

        var total_loss: f32 = 0;
        var td_errors = try self.allocator.alloc(f32, self.batch_size);
        defer self.allocator.free(td_errors);

        for (indices, 0..) |idx, i| {
            const exp = &self.replay_buffer.buffer[idx];

            // Compute predicted value
            var prediction = try model_forward(&exp.state);
            defer prediction.deinit();

            // Compute target
            var target_value = exp.reward;
            if (!exp.done and exp.next_state != null) {
                var next_prediction = try model_forward(&exp.next_state.?);
                defer next_prediction.deinit();
                target_value += self.gamma * next_prediction.max();
            }

            // Compute TD error
            const current_value = prediction.mean();
            const td_error = target_value - current_value;
            td_errors[i] = td_error;

            // Loss
            total_loss += td_error * td_error;
        }

        // Update priorities
        self.replay_buffer.updatePriorities(indices, td_errors, 0.6);

        const avg_loss = total_loss / @as(f32, @floatFromInt(self.batch_size));
        self.running_loss = 0.99 * self.running_loss + 0.01 * avg_loss;
        try self.loss_history.append(self.allocator, avg_loss);

        self.updates_since_last = 0;
        self.total_updates += 1;

        return avg_loss;
    }

    /// Get learning statistics
    pub fn getStats(self: *const Self) LearningStats {
        return .{
            .total_updates = self.total_updates,
            .buffer_size = self.replay_buffer.size,
            .running_loss = self.running_loss,
            .avg_loss = if (self.loss_history.items.len > 0) blk: {
                var sum: f32 = 0;
                const recent = @min(100, self.loss_history.items.len);
                for (self.loss_history.items[self.loss_history.items.len - recent ..]) |l| {
                    sum += l;
                }
                break :blk sum / @as(f32, @floatFromInt(recent));
            } else 0.0,
        };
    }

    pub const LearningStats = struct {
        total_updates: usize,
        buffer_size: usize,
        running_loss: f32,
        avg_loss: f32,
    };
};

// ============================================================================
// Gradient Accumulator
// ============================================================================

/// Accumulates gradients over multiple steps before applying
pub const GradientAccumulator = struct {
    allocator: std.mem.Allocator,
    accumulation_steps: usize,
    current_step: usize = 0,
    accumulated_grads: std.AutoHashMapUnmanaged(usize, F32Tensor),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, accumulation_steps: usize) Self {
        return Self{
            .allocator = allocator,
            .accumulation_steps = accumulation_steps,
            .accumulated_grads = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.accumulated_grads.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.accumulated_grads.deinit(self.allocator);
    }

    /// Accumulate a gradient
    pub fn accumulate(self: *Self, param_id: usize, grad: *const F32Tensor) !void {
        const result = try self.accumulated_grads.getOrPut(self.allocator, param_id);
        if (!result.found_existing) {
            result.value_ptr.* = try grad.clone();
        } else {
            try result.value_ptr.addInPlace(grad);
        }
    }

    /// Step and check if we should apply gradients
    pub fn step(self: *Self) bool {
        self.current_step += 1;
        return self.current_step >= self.accumulation_steps;
    }

    /// Get averaged gradients and reset
    pub fn getAndReset(self: *Self, param_id: usize) ?F32Tensor {
        if (self.accumulated_grads.get(param_id)) |grad| {
            var result = grad;
            result.scaleInPlace(1.0 / @as(f32, @floatFromInt(self.accumulation_steps)));
            _ = self.accumulated_grads.remove(param_id);
            self.current_step = 0;
            return result;
        }
        return null;
    }

    /// Reset all accumulated gradients
    pub fn reset(self: *Self) void {
        var it = self.accumulated_grads.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.accumulated_grads.clearRetainingCapacity();
        self.current_step = 0;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "replay buffer" {
    const allocator = std.testing.allocator;

    var buffer = try ReplayBuffer.init(allocator, 100, false);
    defer buffer.deinit();

    // Add experiences
    for (0..50) |i| {
        const state = try F32Tensor.fill(allocator, &.{4}, @as(f32, @floatFromInt(i)));
        const action = try F32Tensor.fill(allocator, &.{2}, 0.5);

        buffer.add(.{
            .state = state,
            .action = action,
            .reward = 1.0,
            .next_state = null,
            .done = false,
            .timestamp = types.getTimestampSec(),
            .metadata = .{},
        });
    }

    try std.testing.expectEqual(@as(usize, 50), buffer.size);

    // Sample
    const indices = try buffer.sample(10);
    defer allocator.free(indices);
    try std.testing.expectEqual(@as(usize, 10), indices.len);
}

test "adam optimizer" {
    const allocator = std.testing.allocator;

    var optimizer = AdamOptimizer.init(allocator, 0.01, 0.9, 0.999, 1e-8, 0.0);
    defer optimizer.deinit();

    var param = try F32Tensor.fill(allocator, &.{4}, 1.0);
    defer param.deinit();

    var grad = try F32Tensor.fill(allocator, &.{4}, 0.1);
    defer grad.deinit();

    try optimizer.step(0, &param, &grad);

    // Parameters should have changed
    try std.testing.expect(param.data[0] < 1.0);
}

test "loss functions" {
    const allocator = std.testing.allocator;

    var pred = try F32Tensor.fromSlice(allocator, &.{ 0.5, 0.5 }, &.{2});
    defer pred.deinit();

    var target = try F32Tensor.fromSlice(allocator, &.{ 1.0, 0.0 }, &.{2});
    defer target.deinit();

    const mse = LossFn.mse(&pred, &target);
    try std.testing.expect(mse > 0);
}
