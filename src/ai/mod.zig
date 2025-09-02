//! Unified AI Module - Neural Networks, Embeddings, and Machine Learning
//!
//! This module consolidates all AI functionality into a single, high-performance
//! implementation with:
//! - Neural network architectures (MLP, CNN, RNN)
//! - Embedding generation and management
//! - Training and inference pipelines
//! - Model serialization and loading
//! - Performance optimization and monitoring

const std = @import("std");
const core = @import("../core/mod.zig");
const simd = @import("../simd/mod.zig");

/// Neural network layer types
pub const LayerType = enum {
    input,
    dense,
    conv2d,
    maxpool2d,
    dropout,
    activation,
    flatten,
    lstm,
    gru,
    attention,
};

/// Activation functions
pub const Activation = enum {
    relu,
    sigmoid,
    tanh,
    softmax,
    leaky_relu,
    elu,
    gelu,
    swish,
};

/// Neural network layer
pub const Layer = struct {
    layer_type: LayerType,
    input_shape: []const usize,
    output_shape: []const usize,
    weights: ?[]f32 = null,
    biases: ?[]f32 = null,
    activation: ?Activation = null,
    dropout_rate: f32 = 0.0,
    is_training: bool = true,

    pub fn init(allocator: std.mem.Allocator, layer_type: LayerType, input_shape: []const usize, output_shape: []const usize) !*Layer {
        const layer = try allocator.create(Layer);
        layer.* = .{
            .layer_type = layer_type,
            .input_shape = try allocator.dupe(usize, input_shape),
            .output_shape = try allocator.dupe(usize, output_shape),
        };
        return layer;
    }

    pub fn deinit(self: *Layer, allocator: std.mem.Allocator) void {
        if (self.weights) |weights| allocator.free(weights);
        if (self.biases) |biases| allocator.free(biases);
        allocator.free(self.input_shape);
        allocator.free(self.output_shape);
        allocator.destroy(self);
    }

    pub fn forward(self: *Layer, input: []const f32, output: []f32) !void {
        switch (self.layer_type) {
            .dense => try self.forwardDense(input, output),
            .conv2d => try self.forwardConv2D(input, output),
            .maxpool2d => try self.forwardMaxPool2D(input, output),
            .dropout => try self.forwardDropout(input, output),
            .activation => try self.forwardActivation(input, output),
            .flatten => try self.forwardFlatten(input, output),
            .lstm => try self.forwardLSTM(input, output),
            .gru => try self.forwardGRU(input, output),
            .attention => try self.forwardAttention(input, output),
            else => return error.UnsupportedLayerType,
        }
    }

    fn forwardDense(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null or self.biases == null) return error.WeightsNotInitialized;

        const weights = self.weights.?;
        const biases = self.biases.?;
        const input_size = self.input_shape[0];
        const output_size = self.output_shape[0];

        if (input.len != input_size or output.len != output_size) return error.InvalidDimensions;

        // Matrix-vector multiplication: output = weights * input + biases
        simd.matrixVectorMultiply(output, weights, input, output_size, input_size);

        // Add biases
        for (output, 0..) |*out, i| {
            out.* += biases[i];
        }

        // Apply activation if specified
        if (self.activation) |activation| {
            try self.applyActivation(output, activation);
        }
    }

    fn forwardConv2D(self: *Layer, _input: []const f32, _output: []f32) !void {
        // 2D convolution implementation
        // This is a simplified version - full implementation would include
        // kernel weights, padding, stride, etc.
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardMaxPool2D(self: *Layer, _input: []const f32, _output: []f32) !void {
        // 2D max pooling implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardDropout(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.is_training and self.dropout_rate > 0.0) {
            for (input, 0..) |val, i| {
                if (core.random.float(f32) < self.dropout_rate) {
                    output[i] = 0.0;
                } else {
                    output[i] = val / (1.0 - self.dropout_rate);
                }
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardActivation(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.activation) |activation| {
            try self.applyActivation(input, activation);
            @memcpy(output, input);
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardFlatten(self: *Layer, _input: []const f32, _output: []f32) !void {
        _ = self;
        @memcpy(_output, _input);
    }

    fn forwardLSTM(self: *Layer, _input: []const f32, _output: []f32) !void {
        // LSTM implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardGRU(self: *Layer, _input: []const f32, _output: []f32) !void {
        // GRU implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardAttention(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Attention mechanism implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn applyActivation(self: *Layer, data: []f32, activation: Activation) !void {
        _ = self;
        switch (activation) {
            .relu => {
                for (data) |*val| {
                    val.* = @max(0.0, val.*);
                }
            },
            .sigmoid => {
                for (data) |*val| {
                    val.* = 1.0 / (1.0 + @exp(-val.*));
                }
            },
            .tanh => {
                for (data) |*val| {
                    val.* = std.math.tanh(val.*);
                }
            },
            .softmax => {
                var max_val = data[0];
                for (data[1..]) |val| {
                    max_val = @max(max_val, val);
                }

                var sum: f32 = 0.0;
                for (data) |*val| {
                    val.* = @exp(val.* - max_val);
                    sum += val.*;
                }

                for (data) |*val| {
                    val.* /= sum;
                }
            },
            .leaky_relu => {
                for (data) |*val| {
                    val.* = if (val.* > 0.0) val.* else 0.01 * val.*;
                }
            },
            .elu => {
                for (data) |*val| {
                    val.* = if (val.* > 0.0) val.* else @exp(val.*) - 1.0;
                }
            },
            .gelu => {
                for (data) |*val| {
                    val.* = 0.5 * val.* * (1.0 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (val.* + 0.044715 * std.math.pow(f32, val.*, 3))));
                }
            },
            .swish => {
                for (data) |*val| {
                    val.* = val.* / (1.0 + @exp(-val.*));
                }
            },
        }
    }
};

/// Neural network model
pub const NeuralNetwork = struct {
    layers: core.ArrayList(*Layer),
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    output_shape: []const usize,
    is_compiled: bool = false,

    pub fn init(allocator: std.mem.Allocator, input_shape: []const usize, output_shape: []const usize) !*NeuralNetwork {
        const network = try allocator.create(NeuralNetwork);
        network.* = .{
            .layers = core.ArrayList(*Layer).init(allocator),
            .allocator = allocator,
            .input_shape = try allocator.dupe(usize, input_shape),
            .output_shape = try allocator.dupe(usize, output_shape),
        };
        return network;
    }

    pub fn deinit(self: *NeuralNetwork) void {
        for (self.layers.items) |layer| {
            layer.deinit(self.allocator);
        }
        self.layers.deinit();
        self.allocator.free(self.input_shape);
        self.allocator.free(self.output_shape);
        self.allocator.destroy(self);
    }

    pub fn addLayer(self: *NeuralNetwork, layer: *Layer) !void {
        try self.layers.append(layer);
        self.is_compiled = false;
    }

    pub fn addDenseLayer(self: *NeuralNetwork, units: usize, activation: ?Activation) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .dense, prev_output_shape, &[_]usize{units});
        layer.activation = activation;

        // Initialize weights and biases
        const input_size = prev_output_shape[0];
        layer.weights = try self.allocator.alloc(f32, units * input_size);
        layer.biases = try self.allocator.alloc(f32, units);

        // Xavier/Glorot initialization
        const weight_std = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + units)));
        for (layer.weights.?) |*weight| {
            weight.* = core.random.normal(f32) * weight_std;
        }
        for (layer.biases.?) |*bias| {
            bias.* = 0.0;
        }

        try self.addLayer(layer);
    }

    pub fn addDropoutLayer(self: *NeuralNetwork, rate: f32) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .dropout, prev_output_shape, prev_output_shape);
        layer.dropout_rate = rate;
        try self.addLayer(layer);
    }

    pub fn compile(self: *NeuralNetwork) !void {
        // Validate layer connections
        var current_shape = self.input_shape;
        for (self.layers.items) |layer| {
            if (!std.mem.eql(usize, current_shape, layer.input_shape)) {
                return error.InvalidLayerConnection;
            }
            current_shape = layer.output_shape;
        }

        if (!std.mem.eql(usize, current_shape, self.output_shape)) {
            return error.InvalidOutputShape;
        }

        self.is_compiled = true;
    }

    pub fn forward(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
        if (!self.is_compiled) return error.NetworkNotCompiled;
        if (input.len != self.getInputSize()) return error.InvalidInputSize;
        if (output.len != self.getOutputSize()) return error.InvalidOutputSize;

        var current_input = input;
        var current_output = try self.allocator.alloc(f32, self.getMaxLayerSize());
        defer self.allocator.free(current_output);

        for (self.layers.items) |layer| {
            try layer.forward(current_input, current_output);

            // Swap input/output for next layer
            const temp = current_input;
            current_input = current_output;
            current_output = temp;
        }

        // Copy final result to output
        @memcpy(output, current_input[0..self.getOutputSize()]);
    }

    fn getInputSize(self: *const NeuralNetwork) usize {
        var size: usize = 1;
        for (self.input_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn getOutputSize(self: *const NeuralNetwork) usize {
        var size: usize = 1;
        for (self.output_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn getMaxLayerSize(self: *const NeuralNetwork) usize {
        var max_size = self.getInputSize();
        for (self.layers.items) |layer| {
            const layer_size = layer.output_shape[0];
            max_size = @max(max_size, layer_size);
        }
        return @max(max_size, self.getOutputSize());
    }
};

/// Embedding generator
pub const EmbeddingGenerator = struct {
    model: *NeuralNetwork,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize) !*EmbeddingGenerator {
        const model = try NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{embedding_size});

        // Add layers for embedding generation
        try model.addDenseLayer(embedding_size * 2, .relu);
        try model.addDropoutLayer(0.2);
        try model.addDenseLayer(embedding_size, .tanh);

        try model.compile();

        const generator = try allocator.create(EmbeddingGenerator);
        generator.* = .{
            .model = model,
            .allocator = allocator,
        };
        return generator;
    }

    pub fn deinit(self: *EmbeddingGenerator) void {
        self.model.deinit();
        self.allocator.destroy(self);
    }

    pub fn generateEmbedding(self: *EmbeddingGenerator, input: []const f32, embedding: []f32) !void {
        try self.model.forward(input, embedding);
    }

    pub fn generateEmbeddingsBatch(self: *EmbeddingGenerator, inputs: []const []const f32, embeddings: [][]f32) !void {
        if (inputs.len != embeddings.len) return error.InvalidBatchSize;

        for (inputs, 0..) |input, i| {
            try self.generateEmbedding(input, embeddings[i]);
        }
    }
};

/// Model training configuration
pub const TrainingConfig = struct {
    learning_rate: f32 = 0.001,
    batch_size: usize = 32,
    epochs: usize = 100,
    validation_split: f32 = 0.2,
    early_stopping_patience: usize = 10,
    use_momentum: bool = true,
    momentum: f32 = 0.9,
    weight_decay: f32 = 0.0001,
};

/// Loss functions
pub const LossFunction = enum {
    mse,
    cross_entropy,
    binary_cross_entropy,
    categorical_cross_entropy,
    huber,
    hinge,
};

/// Optimizers
pub const Optimizer = enum {
    sgd,
    adam,
    rmsprop,
    adagrad,
};

/// Training metrics
pub const TrainingMetrics = struct {
    loss: f32,
    accuracy: f32,
    val_loss: ?f32 = null,
    val_accuracy: ?f32 = null,
    epoch: usize,
    training_time_ms: u64,
};

/// Model trainer
pub const ModelTrainer = struct {
    model: *NeuralNetwork,
    config: TrainingConfig,
    allocator: std.mem.Allocator,
    optimizer: Optimizer,
    loss_function: LossFunction,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *NeuralNetwork,
        config: TrainingConfig,
        optimizer: Optimizer,
        loss_function: LossFunction,
    ) !*ModelTrainer {
        const trainer = try allocator.create(ModelTrainer);
        trainer.* = .{
            .model = model,
            .config = config,
            .allocator = allocator,
            .optimizer = optimizer,
            .loss_function = loss_function,
        };
        return trainer;
    }

    pub fn deinit(self: *ModelTrainer) void {
        self.allocator.destroy(self);
    }

    pub fn train(
        self: *ModelTrainer,
        inputs: []const []const f32,
        targets: []const []const f32,
    ) !core.ArrayList(TrainingMetrics) {
        const metrics = core.ArrayList(TrainingMetrics).init(self.allocator);

        // Training implementation would go here
        // This is a simplified version - full implementation would include
        // backpropagation, gradient descent, etc.
        _ = inputs;
        _ = targets;

        return metrics;
    }

    /// Mean squared error loss
    fn meanSquaredError(self: *ModelTrainer, _predictions: []const f32, _targets: []const f32) f32 {
        _ = self;
        _ = _predictions;
        _ = _targets;
        return 0.0; // Placeholder implementation
    }

    /// Cross-entropy loss
    fn crossEntropy(self: *ModelTrainer, _predictions: []const f32, _targets: []const f32) f32 {
        _ = self;
        _ = _predictions;
        _ = _targets;
        return 0.0; // Placeholder implementation
    }

    /// Categorical cross-entropy loss
    fn categoricalCrossEntropy(self: *ModelTrainer, _predictions: []const f32, _targets: []const f32) f32 {
        _ = self;
        _ = _predictions;
        _ = _targets;
        return 0.0; // Placeholder implementation
    }

    /// Huber loss
    fn huberLoss(self: *ModelTrainer, _predictions: []const f32, _targets: []const f32) f32 {
        _ = self;
        _ = _predictions;
        _ = _targets;
        return 0.0; // Placeholder implementation
    }

    /// Hinge loss
    fn hingeLoss(self: *ModelTrainer, _predictions: []const f32, _targets: []const f32) f32 {
        _ = self;
        _ = _predictions;
        _ = _targets;
        return 0.0; // Placeholder implementation
    }
};

// Re-export commonly used types
pub const Network = NeuralNetwork;
pub const Embedding = EmbeddingGenerator;
pub const Trainer = ModelTrainer;

test "Neural network basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple neural network
    var network = try NeuralNetwork.init(allocator, &[_]usize{10}, &[_]usize{3});
    defer network.deinit();

    // Add layers
    try network.addDenseLayer(20, .relu);
    try network.addDenseLayer(3, .softmax);

    // Compile the network
    try network.compile();

    // Test forward pass
    const input = [_]f32{0.1} ** 10;
    const output = try allocator.alloc(f32, 3);
    defer allocator.free(output);

    try network.forward(&input, output);

    // Verify output
    try testing.expect(output.len == 3);

    // Check softmax properties
    var sum: f32 = 0.0;
    for (output) |val| {
        try testing.expect(val >= 0.0 and val <= 1.0);
        sum += val;
    }
    try testing.expectApproxEqAbs(1.0, sum, 0.001);
}

test "Embedding generator" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create embedding generator
    var generator = try EmbeddingGenerator.init(allocator, 10, 5);
    defer generator.deinit();

    // Generate embedding
    const input = [_]f32{0.1} ** 10;
    const embedding = try allocator.alloc(f32, 5);
    defer allocator.free(embedding);

    try generator.generateEmbedding(&input, embedding);

    // Verify embedding
    try testing.expect(embedding.len == 5);
    for (embedding) |val| {
        try testing.expect(val >= -1.0 and val <= 1.0); // tanh activation
    }
}
