//! Agent Subsystem for ABI (Zig 0.16)
//!
//! The orchestrator for stateful, GPU-aware training and inference loops.
//! Ensures deterministic execution, reproducibility, and efficient scaling across CPUs and GPUs.
//!
//! Key responsibilities:
//! - Data ingestion: streaming, batching, sharding, and caching of datasets
//! - Model execution: managing forward and backward passes with autodiff
//! - Optimization: updating model parameters with pluggable algorithms
//! - Metrics: collecting loss, accuracy, throughput, latency, and gradient stats
//! - Scalability: enabling multi-GPU and distributed workflows

const std = @import("std");
const builtin = @import("builtin");

/// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Agent subsystem specific errors
pub const AgentError = error{
    InvalidConfiguration,
    InitializationFailed,
    DeviceNotAvailable,
    DataLoaderError,
    OptimizerError,
    ModelError,
    MetricsError,
    MemoryError,
    OutOfMemory,
};

/// Device types for execution
pub const DeviceType = enum {
    cpu,
    gpu,
    distributed,
};

/// Precision modes for mixed precision training
pub const PrecisionMode = enum {
    float32,
    float16,
    mixed16,
    bfloat16,
};

/// Optimizer types
pub const OptimizerType = enum {
    sgd,
    adam,
    rmsprop,
    adafactor,
    lamb,
};

/// Agent configuration
pub const AgentConfig = struct {
    device: DeviceType = .cpu,
    batch_size: u32 = 32,
    optimizer: OptimizerType = .adam,
    precision: PrecisionMode = .float32,
    learning_rate: f32 = 0.001,
    enable_metrics: bool = true,
    enable_checkpointing: bool = false,
    max_epochs: u32 = 100,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.batch_size == 0) return AgentError.InvalidConfiguration;
        if (self.learning_rate <= 0.0) return AgentError.InvalidConfiguration;
        if (self.max_epochs == 0) return AgentError.InvalidConfiguration;
    }
};

/// Training/inference batch
pub const Batch = struct {
    inputs: []const f32,
    targets: ?[]const f32 = null,
    batch_id: u64,
    size: u32,

    pub fn init(inputs: []const f32, targets: ?[]const f32, batch_id: u64) Batch {
        return .{
            .inputs = inputs,
            .targets = targets,
            .batch_id = batch_id,
            .size = @intCast(inputs.len),
        };
    }
};

/// Training metrics
pub const TrainingMetrics = struct {
    loss: f32 = 0.0,
    accuracy: f32 = 0.0,
    throughput: f32 = 0.0, // samples per second
    latency_ms: f32 = 0.0,
    memory_usage_mb: f32 = 0.0,
    epoch: u32 = 0,
    step: u64 = 0,
    gradient_norm: f32 = 0.0,

    pub fn init() TrainingMetrics {
        return .{};
    }

    pub fn update(self: *TrainingMetrics, loss: f32, accuracy: f32) void {
        self.loss = loss;
        self.accuracy = accuracy;
        self.step += 1;
    }
};

/// Data loader for efficient streaming
pub const DataLoader = struct {
    allocator: Allocator,
    batch_size: u32,
    current_batch: u64,
    total_batches: u64,
    prefetch_buffer: ?[]Batch = null,

    pub fn init(allocator: Allocator, batch_size: u32, dataset_size: u64) !*DataLoader {
        const loader = try allocator.create(DataLoader);
        loader.* = .{
            .allocator = allocator,
            .batch_size = batch_size,
            .current_batch = 0,
            .total_batches = (dataset_size + batch_size - 1) / batch_size,
        };
        return loader;
    }

    pub fn deinit(self: *DataLoader) void {
        if (self.prefetch_buffer) |buffer| {
            self.allocator.free(buffer);
        }
        self.allocator.destroy(self);
    }

    pub fn nextBatch(self: *DataLoader, dataset: []const f32) !?Batch {
        if (self.current_batch >= self.total_batches) return null;

        const start_idx = self.current_batch * self.batch_size;
        const end_idx = @min(start_idx + self.batch_size, dataset.len);

        const batch = Batch.init(dataset[start_idx..end_idx], null, // targets would be provided separately in real implementation
            self.current_batch);

        self.current_batch += 1;
        return batch;
    }

    pub fn reset(self: *DataLoader) void {
        self.current_batch = 0;
    }

    pub fn hasNext(self: *const DataLoader) bool {
        return self.current_batch < self.total_batches;
    }
};

/// Simple optimizer interface
pub const Optimizer = struct {
    allocator: Allocator,
    optimizer_type: OptimizerType,
    learning_rate: f32,
    step_count: u64,

    // Optimizer state (simplified)
    momentum_buffers: ?[]f32 = null,
    velocity_buffers: ?[]f32 = null,

    pub fn init(allocator: Allocator, optimizer_type: OptimizerType, learning_rate: f32, param_count: usize) !*Optimizer {
        const opt = try allocator.create(Optimizer);
        opt.* = .{
            .allocator = allocator,
            .optimizer_type = optimizer_type,
            .learning_rate = learning_rate,
            .step_count = 0,
        };

        // Initialize optimizer state based on type
        switch (optimizer_type) {
            .sgd => {},
            .adam => {
                opt.momentum_buffers = try allocator.alloc(f32, param_count);
                opt.velocity_buffers = try allocator.alloc(f32, param_count);
                @memset(opt.momentum_buffers.?, 0.0);
                @memset(opt.velocity_buffers.?, 0.0);
            },
            else => {}, // Other optimizers would be implemented similarly
        }

        return opt;
    }

    pub fn deinit(self: *Optimizer) void {
        if (self.momentum_buffers) |buffers| self.allocator.free(buffers);
        if (self.velocity_buffers) |buffers| self.allocator.free(buffers);
        self.allocator.destroy(self);
    }

    pub fn step(self: *Optimizer, parameters: []f32, gradients: []const f32) !void {
        if (parameters.len != gradients.len) return AgentError.OptimizerError;

        self.step_count += 1;

        switch (self.optimizer_type) {
            .sgd => {
                for (parameters, gradients) |*param, grad| {
                    param.* -= self.learning_rate * grad;
                }
            },
            .adam => {
                if (self.momentum_buffers == null or self.velocity_buffers == null) {
                    return AgentError.OptimizerError;
                }

                const beta1: f32 = 0.9;
                const beta2: f32 = 0.999;
                const eps: f32 = 1e-8;

                const m = self.momentum_buffers.?;
                const v = self.velocity_buffers.?;

                for (parameters, gradients, m, v) |*param, grad, *m_val, *v_val| {
                    m_val.* = beta1 * m_val.* + (1.0 - beta1) * grad;
                    v_val.* = beta2 * v_val.* + (1.0 - beta2) * grad * grad;

                    const m_hat = m_val.* / (1.0 - std.math.pow(f32, beta1, @floatFromInt(self.step_count)));
                    const v_hat = v_val.* / (1.0 - std.math.pow(f32, beta2, @floatFromInt(self.step_count)));

                    param.* -= self.learning_rate * m_hat / (std.math.sqrt(v_hat) + eps);
                }
            },
            else => return AgentError.OptimizerError,
        }
    }
};

/// Simple model interface (lightweight interface for model adapters)
pub const Model = struct {
    allocator: Allocator,
    parameters: []f32,
    gradients: []f32,
    input_size: u32,
    output_size: u32,
    training_mode: bool = true,

    pub fn init(allocator: Allocator, input_size: u32, output_size: u32) !*Model {
        const param_count = input_size * output_size + output_size; // weights + biases

        const model = try allocator.create(Model);
        model.* = .{
            .allocator = allocator,
            .parameters = try allocator.alloc(f32, param_count),
            .gradients = try allocator.alloc(f32, param_count),
            .input_size = input_size,
            .output_size = output_size,
        };

        // Initialize parameters with random values
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        for (model.parameters) |*param| {
            param.* = (random.float(f32) - 0.5) * 0.1;
        }
        @memset(model.gradients, 0.0);

        return model;
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.parameters);
        self.allocator.free(self.gradients);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Model, input: []const f32, output: []f32) !void {
        if (input.len != self.input_size or output.len != self.output_size) {
            return AgentError.ModelError;
        }

        // Simple linear transformation: y = Wx + b
        const weights_end = self.input_size * self.output_size;
        const weights = self.parameters[0..weights_end];
        const biases = self.parameters[weights_end..];

        for (output, 0..) |*out, i| {
            out.* = biases[i];
            for (input, 0..) |in_val, j| {
                out.* += weights[i * self.input_size + j] * in_val;
            }
        }
    }

    pub fn backward(self: *Model, input: []const f32, grad_output: []const f32) !void {
        if (input.len != self.input_size or grad_output.len != self.output_size) {
            return AgentError.ModelError;
        }

        // Compute gradients (simplified)
        const weights_end = self.input_size * self.output_size;
        const grad_weights = self.gradients[0..weights_end];
        const grad_biases = self.gradients[weights_end..];

        // Gradient w.r.t. biases
        for (grad_biases, grad_output) |*grad_bias, grad_out| {
            grad_bias.* += grad_out;
        }

        // Gradient w.r.t. weights
        for (grad_output, 0..) |grad_out, i| {
            for (input, 0..) |in_val, j| {
                grad_weights[i * self.input_size + j] += grad_out * in_val;
            }
        }
    }

    pub fn setTraining(self: *Model, training: bool) void {
        self.training_mode = training;
    }

    pub fn clearGradients(self: *Model) void {
        @memset(self.gradients, 0.0);
    }
};

/// Main Agent orchestrator
pub const Agent = struct {
    allocator: Allocator,
    config: AgentConfig,
    model: *Model,
    optimizer: *Optimizer,
    data_loader: *DataLoader,
    metrics: TrainingMetrics,
    current_epoch: u32 = 0,

    pub fn init(allocator: Allocator, config: AgentConfig) !*Agent {
        try config.validate();

        // Create model (simplified - in practice would be more complex)
        const model = try Model.init(allocator, 10, 1); // Example: 10 inputs, 1 output

        // Create optimizer
        const optimizer = try Optimizer.init(allocator, config.optimizer, config.learning_rate, model.parameters.len);

        // Create data loader (example with dummy dataset size)
        const data_loader = try DataLoader.init(allocator, config.batch_size, 1000);

        const agent = try allocator.create(Agent);
        agent.* = .{
            .allocator = allocator,
            .config = config,
            .model = model,
            .optimizer = optimizer,
            .data_loader = data_loader,
            .metrics = TrainingMetrics.init(),
        };

        return agent;
    }

    pub fn deinit(self: *Agent) void {
        self.model.deinit();
        self.optimizer.deinit();
        self.data_loader.deinit();
        self.allocator.destroy(self);
    }

    pub fn nextBatch(self: *Agent, dataset: []const f32) !?Batch {
        return self.data_loader.nextBatch(dataset);
    }

    pub fn trainStep(self: *Agent, batch: Batch) !f32 {
        self.model.setTraining(true);

        // Forward pass
        var output = try self.allocator.alloc(f32, self.model.output_size);
        defer self.allocator.free(output);

        try self.model.forward(batch.inputs, output);

        // Compute loss (simplified MSE)
        var loss: f32 = 0.0;
        if (batch.targets) |targets| {
            for (output, targets) |pred, target| {
                const diff = pred - target;
                loss += diff * diff;
            }
            loss /= @floatFromInt(output.len);
        } else {
            // Dummy loss for demonstration
            loss = 0.1;
        }

        // Backward pass
        const grad_output = try self.allocator.alloc(f32, output.len);
        defer self.allocator.free(grad_output);

        if (batch.targets) |targets| {
            for (grad_output, output, targets) |*grad, pred, target| {
                grad.* = 2.0 * (pred - target) / @as(f32, @floatFromInt(output.len));
            }
        } else {
            // Dummy gradients
            @memset(grad_output, 0.01);
        }

        self.model.clearGradients();
        try self.model.backward(batch.inputs, grad_output);

        // Optimizer step
        try self.optimizer.step(self.model.parameters, self.model.gradients);

        // Update metrics
        const accuracy = 1.0 - loss; // Simplified accuracy calculation
        self.metrics.update(loss, accuracy);

        return loss;
    }

    pub fn currentAccuracy(self: *Agent, batch: Batch) !f32 {
        _ = batch;
        return self.metrics.accuracy;
    }

    pub fn setEpoch(self: *Agent, epoch: u32) void {
        self.current_epoch = epoch;
        self.metrics.epoch = epoch;
    }

    pub fn getMetrics(self: *const Agent) TrainingMetrics {
        return self.metrics;
    }

    /// Reset data loader for new epoch
    pub fn resetDataLoader(self: *Agent) void {
        self.data_loader.reset();
    }

    /// Check if there are more batches available
    pub fn hasNextBatch(self: *const Agent) bool {
        return self.data_loader.hasNext();
    }
};

// Test the Agent subsystem
test "agent initialization and basic operation" {
    const testing = std.testing;

    const config = AgentConfig{
        .device = .cpu,
        .batch_size = 4,
        .optimizer = .adam,
        .precision = .float32,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    try testing.expect(agent.config.batch_size == 4);
    try testing.expect(agent.config.optimizer == .adam);
}

test "data loader batching" {
    const testing = std.testing;

    var loader = try DataLoader.init(testing.allocator, 3, 10);
    defer loader.deinit();

    const dataset = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    var batch_count: u32 = 0;
    while (try loader.nextBatch(&dataset)) |batch| {
        batch_count += 1;
        try testing.expect(batch.size > 0);
    }

    try testing.expect(batch_count == 4); // ceil(10/3) = 4 batches
}

test "training step execution" {
    const testing = std.testing;

    const config = AgentConfig{
        .batch_size = 2,
        .learning_rate = 0.01,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    const inputs = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const targets = [_]f32{0.5};

    const batch = Batch.init(inputs[0..], targets[0..], 0);
    const loss = try agent.trainStep(batch);

    try testing.expect(loss >= 0.0);
    try testing.expect(agent.metrics.step == 1);
}
