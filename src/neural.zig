//! Neural Network Module for Self-Learning Vector Embeddings
//!
//! This module provides neural network capabilities for learning and generating
//! vector embeddings. It includes:
//! - Feed-forward neural network with configurable layers
//! - Backpropagation training with SGD optimizer
//! - Activation functions (ReLU, Sigmoid, Tanh)
//! - Embedding generation from raw input
//! - SIMD-accelerated matrix operations

const std = @import("std");
const math = std.math;
const core = @import("core/mod.zig");
const abi = @import("root.zig");
const simd_vector = @import("simd_vector.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Neural network layer types
pub const LayerType = enum {
    Dense,
    Embedding,
    Dropout,
};

/// Activation functions
pub const Activation = enum {
    ReLU,
    Sigmoid,
    Tanh,
    None,

    /// Apply activation function to a value
    pub fn apply(self: Activation, x: f32) f32 {
        return switch (self) {
            .ReLU => if (x > 0) x else 0,
            .Sigmoid => 1.0 / (1.0 + math.exp(-x)),
            .Tanh => math.tanh(x),
            .None => x,
        };
    }

    /// Derivative of activation function
    pub fn derivative(self: Activation, x: f32) f32 {
        return switch (self) {
            .ReLU => if (x > 0) 1 else 0,
            .Sigmoid => {
                const s = self.apply(x);
                return s * (1 - s);
            },
            .Tanh => {
                const t = math.tanh(x);
                return 1 - t * t;
            },
            .None => 1,
        };
    }
};

/// Neural network training configuration
pub const TrainingConfig = struct {
    learning_rate: f32 = 0.001,
    batch_size: usize = 32,
    epochs: usize = 100,
    momentum: f32 = 0.9,
    weight_decay: f32 = 0.0001,
    early_stopping_patience: usize = 10,
    validation_split: f32 = 0.2,
};

/// Layer configuration
pub const LayerConfig = struct {
    type: LayerType,
    input_size: usize,
    output_size: usize,
    activation: Activation = .None,
    dropout_rate: f32 = 0.0,
    weight_init_scale: f32 = 1.0,
};

/// Complete neural network configuration
pub const NetworkConfig = struct {
    input_size: usize,
    hidden_layers: []const LayerConfig,
    output_size: usize,
    training: TrainingConfig = .{},
};

/// Neural network layer
pub const Layer = struct {
    type: LayerType,
    weights: []f32,
    biases: []f32,
    activation: Activation,
    dropout_rate: f32,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,

    /// Initialize a new layer
    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !*Layer {
        const self = try allocator.create(Layer);
        self.* = .{
            .type = config.type,
            .weights = try allocator.alloc(f32, config.input_size * config.output_size),
            .biases = try allocator.alloc(f32, config.output_size),
            .activation = config.activation,
            .dropout_rate = config.dropout_rate,
            .input_size = config.input_size,
            .output_size = config.output_size,
            .allocator = allocator,
        };

        // Initialize weights with Xavier/Glorot initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.input_size + config.output_size)));
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();

        for (self.weights) |*w| {
            w.* = (random.float(f32) * 2 - 1) * scale;
        }
        for (self.biases) |*b| {
            b.* = 0;
        }

        return self;
    }

    /// Free layer resources
    pub fn deinit(self: *Layer) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
        self.allocator.destroy(self);
    }

    /// Forward pass through the layer
    pub fn forward(self: *Layer, input: []const f32, allocator: std.mem.Allocator) ![]f32 {
        std.debug.assert(input.len == self.input_size);
        var output = try allocator.alloc(f32, self.output_size);
        errdefer allocator.free(output);

        // Matrix multiplication with SIMD
        var i: usize = 0;
        while (i < self.output_size) : (i += 1) {
            const weights_start = i * self.input_size;
            const weights_end = weights_start + self.input_size;
            output[i] = simd_vector.dotProductSIMD(
                input,
                self.weights[weights_start..weights_end],
                .{},
            ) + self.biases[i];
            output[i] = self.activation.apply(output[i]);
        }

        // Apply dropout during training
        if (self.dropout_rate > 0) {
            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const random = rng.random();
            for (output) |*o| {
                if (random.float(f32) < self.dropout_rate) {
                    o.* = 0;
                } else {
                    o.* /= (1.0 - self.dropout_rate); // Scale to maintain expected value
                }
            }
        }

        return output;
    }

    /// Backward pass through the layer
    pub fn backward(
        self: *Layer,
        input: []const f32,
        output: []const f32,
        output_gradient: []const f32,
        learning_rate: f32,
        allocator: std.mem.Allocator,
    ) ![]f32 {
        std.debug.assert(output_gradient.len == self.output_size);
        var input_gradient = try allocator.alloc(f32, self.input_size);
        errdefer allocator.free(input_gradient);

        // Calculate gradients
        for (0..self.output_size) |i| {
            const gradient = output_gradient[i] *
                self.activation.derivative(output[i]);

            // Update biases
            self.biases[i] -= learning_rate * gradient;

            // Update weights and calculate input gradient
            const weights_start = i * self.input_size;
            for (0..self.input_size) |j| {
                const weight_idx = weights_start + j;
                input_gradient[j] += self.weights[weight_idx] * gradient;
                self.weights[weight_idx] -= learning_rate * gradient * input[j];
            }
        }

        return input_gradient;
    }
};

/// Neural network for learning embeddings
pub const NeuralNetwork = struct {
    layers: std.ArrayList(*Layer),
    allocator: std.mem.Allocator,

    /// Initialize a new neural network
    pub fn init(allocator: std.mem.Allocator) !*NeuralNetwork {
        const self = try allocator.create(NeuralNetwork);
        self.layers = try std.ArrayList(*Layer).initCapacity(allocator, 16);
        self.allocator = allocator;
        return self;
    }

    /// Free network resources
    pub fn deinit(self: *NeuralNetwork) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Add a layer to the network
    pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void {
        const layer = try Layer.init(self.allocator, config);
        try self.layers.append(self.allocator, layer);
    }

    /// Save network to file (basic implementation)
    pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
        _ = try file.write("NEURAL_NETWORK_V1\n");

        // Write layer count
        var layer_count_buf: [32]u8 = undefined;
        const layer_count_str = try std.fmt.bufPrint(&layer_count_buf, "{}\n", .{self.layers.items.len});
        _ = try file.write(layer_count_str);

        // Write each layer (simplified - just layer type and sizes)
        for (self.layers.items) |layer| {
            var layer_buf: [128]u8 = undefined;
            const layer_str = try std.fmt.bufPrint(&layer_buf, "{} {} {}\n", .{
                @intFromEnum(layer.type),
                layer.input_size,
                layer.output_size,
            });
            _ = try file.write(layer_str);
        }
    }

    /// Load network from file (basic implementation)
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork {
        // For now, return a simple error - full implementation needs more complex parsing
        // that would require more Zig standard library features
        _ = allocator;
        _ = path;
        return error.NotImplemented;
    }

    /// Forward pass through the network
    pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32 {
        var current = try self.allocator.dupe(f32, input);
        errdefer self.allocator.free(current);

        for (self.layers.items) |layer| {
            const next = try layer.forward(current, self.allocator);
            self.allocator.free(current);
            current = next;
        }

        return current;
    }

    /// Train the network on a single sample
    pub fn trainStep(
        self: *NeuralNetwork,
        input: []const f32,
        target: []const f32,
        learning_rate: f32,
    ) !f32 {
        // Forward pass with intermediate values
        var activations = try std.ArrayList([]f32).initCapacity(self.allocator, 8);
        defer {
            for (activations.items) |activation| {
                self.allocator.free(activation);
            }
            activations.deinit(self.allocator);
        }

        try activations.append(self.allocator, try self.allocator.dupe(f32, input));
        for (self.layers.items) |layer| {
            const output = try layer.forward(
                activations.items[activations.items.len - 1],
                self.allocator,
            );
            try activations.append(self.allocator, output);
        }

        // Calculate loss and output gradient
        const output = activations.items[activations.items.len - 1];
        var loss: f32 = 0;
        var output_gradient = try self.allocator.alloc(f32, output.len);
        defer self.allocator.free(output_gradient);

        for (output, target, 0..) |o, t, i| {
            const diff = o - t;
            loss += diff * diff;
            output_gradient[i] = 2 * diff; // MSE derivative
        }
        loss /= @as(f32, @floatFromInt(output.len));

        // Backward pass
        var gradient = output_gradient;
        var i: usize = self.layers.items.len;
        while (i > 0) : (i -= 1) {
            const layer = self.layers.items[i - 1];
            const layer_input = activations.items[i - 1];
            const layer_output = activations.items[i];
            const input_gradient = try layer.backward(
                layer_input,
                layer_output,
                gradient,
                learning_rate,
                self.allocator,
            );
            if (i > 1) {
                self.allocator.free(gradient);
                gradient = input_gradient;
            }
        }

        return loss;
    }
};

test "neural network basics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple network
    var nn = try NeuralNetwork.init(allocator);
    defer nn.deinit();

    // Add layers
    try nn.addLayer(.{
        .type = .Dense,
        .input_size = 2,
        .output_size = 3,
        .activation = .ReLU,
    });
    try nn.addLayer(.{
        .type = .Dense,
        .input_size = 3,
        .output_size = 1,
        .activation = .Sigmoid,
    });

    // Test forward pass
    const input = [_]f32{ 0.5, -0.2 };
    const output = try nn.forward(&input);
    defer allocator.free(output);

    try testing.expect(output.len == 1);
    try testing.expect(output[0] >= 0 and output[0] <= 1);

    // Test training
    const target = [_]f32{0.7};
    const loss = try nn.trainStep(&input, &target, 0.1);
    try testing.expect(loss >= 0);
}
