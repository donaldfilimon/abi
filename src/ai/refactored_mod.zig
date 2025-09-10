//! Unified AI Module - Refactored and Optimized
//!
//! This module provides a clean, modular AI framework with:
//! - Neural network architectures with SIMD optimization
//! - Embedding generation and management
//! - Training and inference pipelines
//! - Performance optimization and monitoring
//! - Clean separation of concerns and well-defined interfaces

const std = @import("std");

// FIXME: Replace relative imports with proper module imports
// const core = @import("core");
// const simd = @import("simd");

const core = @import("../core/mod.zig");
const simd = @import("../core/simd.zig");
const activation = @import("activation.zig");
const layer = @import("layer.zig");

const Allocator = std.mem.Allocator;
const FrameworkError = core.FrameworkError;

// Re-export key types for convenience
pub const ActivationType = activation.ActivationType;
pub const LayerType = layer.LayerType;
pub const WeightInit = layer.WeightInit;
pub const Layer = layer.Layer;
pub const LayerConfig = layer.LayerConfig;
pub const ActivationProcessor = activation.ActivationProcessor;

/// Loss functions with comprehensive coverage
pub const LossFunction = enum {
    mean_squared_error,
    mean_absolute_error,
    cross_entropy,
    binary_cross_entropy,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    huber,
    hinge,
    squared_hinge,
    cosine_similarity,
    kullback_leibler_divergence,
    focal_loss,
    dice_loss,
    contrastive_loss,
    triplet_loss,
};

/// Optimizers with state-of-the-art algorithms
pub const Optimizer = enum {
    sgd,
    momentum_sgd,
    nesterov_sgd,
    adam,
    adamw,
    adamax,
    nadam,
    rmsprop,
    adagrad,
    adadelta,
    adambound,
    radam,
    lookahead,
    lamb,
};

/// Learning rate scheduling strategies
pub const LRScheduler = enum {
    constant,
    step_decay,
    exponential_decay,
    polynomial_decay,
    cosine_annealing,
    cosine_annealing_warm_restarts,
    reduce_on_plateau,
    cyclic,
    one_cycle,
};

/// Model training configuration
pub const TrainingConfig = struct {
    // Basic training parameters
    learning_rate: f32 = 0.001,
    batch_size: usize = 32,
    epochs: usize = 100,
    validation_split: f32 = 0.2,

    // Early stopping and checkpointing
    early_stopping_patience: usize = 10,
    early_stopping_min_delta: f32 = 0.001,
    save_best_only: bool = true,
    checkpoint_frequency: usize = 10,

    // Optimizer configuration
    optimizer: Optimizer = .adam,
    use_momentum: bool = true,
    momentum: f32 = 0.9,
    weight_decay: f32 = 0.0001,
    nesterov: bool = false,

    // Adam-specific parameters
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    amsgrad: bool = false,

    // Learning rate scheduling
    lr_scheduler: LRScheduler = .constant,
    lr_decay_rate: f32 = 0.1,
    lr_decay_steps: usize = 1000,
    lr_warmup_steps: usize = 0,

    // Regularization
    gradient_clipping: ?f32 = null,

    // Monitoring and logging
    log_frequency: usize = 100,
    validate_frequency: usize = 1,
    enable_logging: bool = true,
};

/// Training metrics with comprehensive tracking
pub const TrainingMetrics = struct {
    // Basic metrics
    loss: f32,
    accuracy: f32 = 0.0,
    val_loss: ?f32 = null,
    val_accuracy: ?f32 = null,

    // Additional metrics
    precision: ?f32 = null,
    recall: ?f32 = null,
    f1_score: ?f32 = null,

    // Training progress
    epoch: usize,
    step: usize = 0,
    training_time_ms: u64,

    // Performance metrics
    throughput_samples_per_sec: f32 = 0.0,
    memory_usage_mb: f32 = 0.0,

    // Learning dynamics
    learning_rate: f32,
    gradient_norm: ?f32 = null,
    weight_norm: ?f32 = null,
};

/// Neural network model with enhanced capabilities
pub const NeuralNetwork = struct {
    layers: std.ArrayList(*Layer),
    allocator: Allocator,
    input_shape: []const usize,
    output_shape: []const usize,
    is_compiled: bool = false,
    is_training: bool = true,
    model_name: ?[]const u8 = null,
    version: u32 = 1,

    const Self = @This();

    pub fn init(allocator: Allocator, input_shape: []const usize, output_shape: []const usize) FrameworkError!*Self {
        const network = try allocator.create(Self);
        errdefer allocator.destroy(network);

        network.* = .{
            .layers = std.ArrayList(*Layer).init(allocator),
            .allocator = allocator,
            .input_shape = try allocator.dupe(usize, input_shape),
            .output_shape = try allocator.dupe(usize, output_shape),
        };
        return network;
    }

    pub fn deinit(self: *Self) void {
        for (self.layers.items) |l| {
            l.deinit();
        }
        self.layers.deinit();
        self.allocator.free(self.input_shape);
        self.allocator.free(self.output_shape);
        if (self.model_name) |name| {
            self.allocator.free(name);
        }
        self.allocator.destroy(self);
    }

    pub fn setTraining(self: *Self, is_training: bool) void {
        self.is_training = is_training;
        for (self.layers.items) |l| {
            l.setTraining(is_training);
        }
    }

    pub fn addLayer(self: *Self, layer_config: LayerConfig) FrameworkError!void {
        const l = try Layer.init(self.allocator, layer_config);
        try self.layers.append(l);
        self.is_compiled = false;
    }

    pub fn addDenseLayer(self: *Self, units: usize, activation_type: ?ActivationType) FrameworkError!void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].config.output_shape
        else
            self.input_shape;

        const output_shape = &[_]usize{units};

        const config = LayerConfig{
            .layer_type = .dense,
            .input_shape = prev_output_shape,
            .output_shape = output_shape,
            .activation_type = activation_type,
        };

        try self.addLayer(config);
    }

    pub fn addDropoutLayer(self: *Self, rate: f32) FrameworkError!void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].config.output_shape
        else
            self.input_shape;

        const regularization = layer.Regularization{
            .dropout_rate = rate,
        };

        const config = LayerConfig{
            .layer_type = .dropout,
            .input_shape = prev_output_shape,
            .output_shape = prev_output_shape,
            .regularization = regularization,
        };

        try self.addLayer(config);
    }

    pub fn compile(self: *Self) FrameworkError!void {
        if (self.layers.items.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate layer connections
        var current_shape = self.input_shape;
        for (self.layers.items) |l| {
            if (!std.mem.eql(usize, current_shape, l.config.input_shape)) {
                return FrameworkError.InvalidConfiguration;
            }
            current_shape = l.config.output_shape;
        }

        if (!std.mem.eql(usize, current_shape, self.output_shape)) {
            return FrameworkError.InvalidConfiguration;
        }

        // Initialize all layer weights
        var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.microTimestamp())));
        for (self.layers.items) |l| {
            try l.initializeWeights(&rng.random());
        }

        self.is_compiled = true;
    }

    pub fn forward(self: *Self, input: []const f32, output: []f32) FrameworkError!void {
        if (!self.is_compiled) return FrameworkError.InvalidState;
        if (input.len != self.getInputSize()) return FrameworkError.InvalidData;
        if (output.len != self.getOutputSize()) return FrameworkError.InvalidData;

        const max_buffer_size = self.getMaxLayerSize();

        // Allocate buffers for intermediate results
        var buffer1 = try self.allocator.alloc(f32, max_buffer_size);
        defer self.allocator.free(buffer1);
        var buffer2 = try self.allocator.alloc(f32, max_buffer_size);
        defer self.allocator.free(buffer2);

        // Copy input to first buffer
        @memcpy(buffer1[0..input.len], input);

        var current_input_slice = buffer1[0..input.len];
        var current_output_buffer = buffer2;
        var last_output_slice: []f32 = undefined;

        for (self.layers.items, 0..) |l, i| {
            const layer_output_size = l.getOutputSize();
            const output_slice = current_output_buffer[0..layer_output_size];

            try l.forward(current_input_slice, output_slice, null);
            last_output_slice = output_slice;

            // Prepare for next iteration
            if (i < self.layers.items.len - 1) {
                current_input_slice = output_slice;
                // Switch buffers
                if (current_output_buffer.ptr == buffer2.ptr) {
                    current_output_buffer = buffer1;
                } else {
                    current_output_buffer = buffer2;
                }
            }
        }

        @memcpy(output, last_output_slice[0..self.getOutputSize()]);
    }

    pub fn predict(self: *Self, input: []const f32, output: []f32) FrameworkError!void {
        const was_training = self.is_training;
        self.setTraining(false);
        defer self.setTraining(was_training);

        try self.forward(input, output);
    }

    pub fn getParameterCount(self: *const Self) usize {
        var count: usize = 0;
        for (self.layers.items) |l| {
            if (l.weights) |weights| count += weights.len;
            if (l.biases) |biases| count += biases.len;
        }
        return count;
    }

    pub fn getMemoryUsage(self: *const Self) usize {
        var usage: usize = 0;
        for (self.layers.items) |l| {
            if (l.weights) |weights| usage += weights.len * @sizeOf(f32);
            if (l.biases) |biases| usage += biases.len * @sizeOf(f32);
        }
        return usage;
    }

    fn getInputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.input_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn getOutputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.output_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn getMaxLayerSize(self: *const Self) usize {
        var max_size = self.getInputSize();
        for (self.layers.items) |l| {
            const layer_size = l.getOutputSize();
            max_size = @max(max_size, layer_size);
        }
        return @max(max_size, self.getOutputSize());
    }
};

/// Advanced embedding generator
pub const EmbeddingGenerator = struct {
    model: *NeuralNetwork,
    allocator: Allocator,
    embedding_dim: usize,
    model_type: enum { mlp, transformer } = .mlp,

    const Self = @This();

    pub fn init(allocator: Allocator, input_size: usize, embedding_size: usize) FrameworkError!*Self {
        const model = try NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{embedding_size});

        // Add layers for embedding generation
        try model.addDenseLayer(embedding_size * 2, .gelu);
        try model.addDropoutLayer(0.1);
        try model.addDenseLayer(embedding_size, .tanh);

        try model.compile();

        const generator = try allocator.create(Self);
        generator.* = .{
            .model = model,
            .allocator = allocator,
            .embedding_dim = embedding_size,
        };
        return generator;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.allocator.destroy(self);
    }

    pub fn generateEmbedding(self: *Self, input: []const f32, embedding: []f32) FrameworkError!void {
        return self.model.predict(input, embedding);
    }

    pub fn computeSimilarity(self: *Self, embedding1: []const f32, embedding2: []const f32) f32 {
        _ = self;
        if (embedding1.len != embedding2.len) return 0.0;

        return simd.VectorOps.dotProduct(embedding1, embedding2) /
            (@sqrt(simd.VectorOps.dotProduct(embedding1, embedding1)) *
                @sqrt(simd.VectorOps.dotProduct(embedding2, embedding2)));
    }
};

/// Loss computation utilities
pub const LossUtils = struct {
    pub fn computeLoss(loss_fn: LossFunction, predictions: []const f32, targets: []const f32) f32 {
        switch (loss_fn) {
            .mean_squared_error => return meanSquaredError(predictions, targets),
            .mean_absolute_error => return meanAbsoluteError(predictions, targets),
            .binary_cross_entropy => return binaryCrossEntropy(predictions, targets),
            .categorical_cross_entropy => return categoricalCrossEntropy(predictions, targets),
            .huber => return huberLoss(predictions, targets),
            else => return 0.0,
        }
    }

    fn meanSquaredError(predictions: []const f32, targets: []const f32) f32 {
        if (predictions.len != targets.len) return std.math.inf(f32);

        var sum: f32 = 0.0;
        for (predictions, targets) |pred, target| {
            const diff = pred - target;
            sum += diff * diff;
        }
        return sum / @as(f32, @floatFromInt(predictions.len));
    }

    fn meanAbsoluteError(predictions: []const f32, targets: []const f32) f32 {
        if (predictions.len != targets.len) return std.math.inf(f32);

        var sum: f32 = 0.0;
        for (predictions, targets) |pred, target| {
            sum += @abs(pred - target);
        }
        return sum / @as(f32, @floatFromInt(predictions.len));
    }

    fn binaryCrossEntropy(predictions: []const f32, targets: []const f32) f32 {
        if (predictions.len != targets.len) return std.math.inf(f32);

        var loss: f32 = 0.0;
        for (predictions, targets) |pred, target| {
            const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
            loss -= target * @log(clipped_pred) + (1.0 - target) * @log(1.0 - clipped_pred);
        }
        return loss / @as(f32, @floatFromInt(predictions.len));
    }

    fn categoricalCrossEntropy(predictions: []const f32, targets: []const f32) f32 {
        if (predictions.len != targets.len) return std.math.inf(f32);

        var loss: f32 = 0.0;
        for (predictions, targets) |pred, target| {
            const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
            loss -= target * @log(clipped_pred);
        }
        return loss;
    }

    fn huberLoss(predictions: []const f32, targets: []const f32) f32 {
        if (predictions.len != targets.len) return std.math.inf(f32);

        const delta: f32 = 1.0;
        var loss: f32 = 0.0;

        for (predictions, targets) |pred, target| {
            const diff = @abs(pred - target);
            if (diff <= delta) {
                loss += 0.5 * diff * diff;
            } else {
                loss += delta * (diff - 0.5 * delta);
            }
        }
        return loss / @as(f32, @floatFromInt(predictions.len));
    }
};

/// Utility functions for creating common network architectures
pub const NetworkUtils = struct {
    pub fn createMLP(allocator: Allocator, layer_sizes: []const usize, activations: []const ActivationType) FrameworkError!*NeuralNetwork {
        if (layer_sizes.len < 2) return FrameworkError.InvalidConfiguration;
        if (activations.len != layer_sizes.len - 1) return FrameworkError.InvalidConfiguration;

        const network = try NeuralNetwork.init(allocator, &[_]usize{layer_sizes[0]}, &[_]usize{layer_sizes[layer_sizes.len - 1]});

        for (1..layer_sizes.len) |i| {
            try network.addDenseLayer(layer_sizes[i], activations[i - 1]);
        }

        try network.compile();
        return network;
    }

    pub fn createAutoencoder(allocator: Allocator, input_size: usize, encoding_size: usize) FrameworkError!*NeuralNetwork {
        const network = try NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{input_size});

        // Encoder
        try network.addDenseLayer(input_size / 2, .relu);
        try network.addDenseLayer(encoding_size, .relu);

        // Decoder
        try network.addDenseLayer(input_size / 2, .relu);
        try network.addDenseLayer(input_size, .sigmoid);

        try network.compile();
        return network;
    }
};

// Re-export for backward compatibility
pub const Network = NeuralNetwork;
pub const Embedding = EmbeddingGenerator;
pub const Config = TrainingConfig;
pub const Metrics = TrainingMetrics;
pub const Loss = LossFunction;
pub const Opt = Optimizer;

test "neural network creation and forward pass" {
    const testing = std.testing;

    var network = try NeuralNetwork.init(testing.allocator, &[_]usize{4}, &[_]usize{2});
    defer network.deinit();

    try network.addDenseLayer(8, .relu);
    try network.addDenseLayer(2, .softmax);
    try network.compile();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{0} ** 2;

    try network.forward(&input, &output);

    // Verify output (should sum to ~1 for softmax)
    var sum: f32 = 0.0;
    for (output) |val| {
        sum += val;
        try testing.expect(val >= 0.0);
        try testing.expect(val <= 1.0);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.1);
}

test "embedding generator" {
    const testing = std.testing;

    var generator = try EmbeddingGenerator.init(testing.allocator, 10, 8);
    defer generator.deinit();

    const input = [_]f32{0.1} ** 10;
    var embedding = [_]f32{0} ** 8;

    try generator.generateEmbedding(&input, &embedding);

    // Verify embedding was generated
    var has_non_zero = false;
    for (embedding) |val| {
        if (val != 0.0) {
            has_non_zero = true;
            break;
        }
    }
    try testing.expect(has_non_zero);
}

test "loss functions" {
    const testing = std.testing;

    const predictions = [_]f32{ 0.8, 0.2, 0.9, 0.1 };
    const targets = [_]f32{ 1.0, 0.0, 1.0, 0.0 };

    const mse = LossUtils.computeLoss(.mean_squared_error, &predictions, &targets);
    const mae = LossUtils.computeLoss(.mean_absolute_error, &predictions, &targets);

    try testing.expect(mse >= 0.0);
    try testing.expect(mae >= 0.0);
    try testing.expect(mae <= mse); // MAE should be less than or equal to MSE
}
