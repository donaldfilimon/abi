//! ML Components - Modernized for Zig 0.16
//!
//! Machine learning components with proper memory management and initialization patterns

const std = @import("std");
const collections = @import("../core/collections.zig");

/// Neural network layer types
pub const LayerType = enum {
    dense,
    conv2d,
    max_pool2d,
    dropout,
    batch_norm,

    pub fn requiresWeights(self: LayerType) bool {
        return switch (self) {
            .dense, .conv2d, .batch_norm => true,
            .max_pool2d, .dropout => false,
        };
    }
};

/// Activation functions
pub const Activation = enum {
    none,
    relu,
    sigmoid,
    tanh,
    softmax,

    pub fn apply(self: Activation, x: f32) f32 {
        return switch (self) {
            .none => x,
            .relu => @max(0.0, x),
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .tanh => std.math.tanh(x),
            .softmax => x, // Note: Softmax requires vector operation, not scalar
        };
    }
};

/// Layer configuration
pub const LayerConfig = struct {
    layer_type: LayerType,
    input_size: u32,
    output_size: u32,
    activation: Activation = .none,
    dropout_rate: f32 = 0.0,
    use_bias: bool = true,
};

/// Neural network layer
pub const Layer = struct {
    const Self = @This();

    config: LayerConfig,
    weights: ?[]f32 = null,
    biases: ?[]f32 = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !Self {
        var layer = Self{
            .config = config,
            .allocator = allocator,
        };

        if (config.layer_type.requiresWeights()) {
            // Initialize weights with Xavier initialization
            const weight_count = config.input_size * config.output_size;
            layer.weights = try allocator.alloc(f32, weight_count);

            // Xavier initialization
            const variance = 2.0 / @as(f32, @floatFromInt(config.input_size + config.output_size));
            const std_dev = @sqrt(variance);

            var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
            const random = prng.random();

            for (layer.weights.?) |*weight| {
                weight.* = random.floatNorm(f32) * std_dev;
            }

            if (config.use_bias) {
                layer.biases = try allocator.alloc(f32, config.output_size);
                @memset(layer.biases.?, 0.0);
            }
        }

        return layer;
    }

    pub fn deinit(self: *Self) void {
        if (self.weights) |weights| {
            self.allocator.free(weights);
        }
        if (self.biases) |biases| {
            self.allocator.free(biases);
        }
    }

    pub fn forward(self: *const Self, input: []const f32, output: []f32) !void {
        if (input.len != self.config.input_size or output.len != self.config.output_size) {
            return error.InvalidDimensions;
        }

        switch (self.config.layer_type) {
            .dense => try self.forwardDense(input, output),
            else => return error.NotImplemented,
        }

        // Apply activation function
        for (output) |*val| {
            val.* = self.config.activation.apply(val.*);
        }
    }

    fn forwardDense(self: *const Self, input: []const f32, output: []f32) !void {
        const weights = self.weights orelse return error.NoWeights;

        // Matrix multiplication: output = input * weights + bias
        for (0..self.config.output_size) |i| {
            var sum: f32 = 0.0;
            for (0..self.config.input_size) |j| {
                const weight_idx = j * self.config.output_size + i;
                sum += input[j] * weights[weight_idx];
            }

            if (self.biases) |biases| {
                sum += biases[i];
            }

            output[i] = sum;
        }
    }
};

/// Simple neural network
pub const NeuralNetwork = struct {
    const Self = @This();

    layers: collections.ArrayList(Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .layers = collections.ArrayList(Layer){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);
    }

    pub fn addLayer(self: *Self, config: LayerConfig) !void {
        const layer = try Layer.init(self.allocator, config);
        try self.layers.append(self.allocator, layer);
    }

    pub fn forward(self: *const Self, input: []const f32, output: []f32) !void {
        if (self.layers.items.len == 0) return error.NoLayers;

        // Setup arena for intermediate buffers
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        var current_input = input;
        var buffers = collections.ArrayList([]f32){};
        defer buffers.deinit(arena_allocator);

        const layers = self.layers.items;
        for (layers) |*layer| {
            const buffer = try arena_allocator.alloc(f32, layer.config.output_size);
            try buffers.append(arena_allocator, buffer);

            try layer.forward(current_input, buffer);
            current_input = buffer;
        }

        // Copy final output
        const final_output = buffers.items[buffers.items.len - 1];
        @memcpy(output[0..final_output.len], final_output);
    }

    pub fn getInputSize(self: *const Self) ?u32 {
        const layers = self.layers.items;
        if (layers.len == 0) return null;
        return layers[0].config.input_size;
    }

    pub fn getOutputSize(self: *const Self) ?u32 {
        const layers = self.layers.items;
        if (layers.len == 0) return null;
        return layers[layers.len - 1].config.output_size;
    }
};

/// Vector operations for ML computations
pub const VectorOps = struct {
    /// Dot product of two vectors
    pub fn dot(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var result: f32 = 0.0;
        for (a, b) |a_val, b_val| {
            result += a_val * b_val;
        }
        return result;
    }

    /// L2 norm of a vector
    pub fn norm(vec: []const f32) f32 {
        return @sqrt(dot(vec, vec));
    }

    /// Normalize vector in-place
    pub fn normalize(vec: []f32) void {
        const n = norm(vec);
        if (n > 0.0) {
            for (vec) |*val| {
                val.* /= n;
            }
        }
    }

    /// Cosine similarity between two vectors
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        const dot_product = dot(a, b);
        const norm_a = norm(a);
        const norm_b = norm(b);

        if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
        return dot_product / (norm_a * norm_b);
    }

    /// Element-wise addition
    pub fn add(a: []const f32, b: []const f32, result: []f32) void {
        std.debug.assert(a.len == b.len and a.len == result.len);
        for (a, b, result) |a_val, b_val, *res_val| {
            res_val.* = a_val + b_val;
        }
    }

    /// Element-wise multiplication
    pub fn multiply(a: []const f32, b: []const f32, result: []f32) void {
        std.debug.assert(a.len == b.len and a.len == result.len);
        for (a, b, result) |a_val, b_val, *res_val| {
            res_val.* = a_val * b_val;
        }
    }

    /// Scalar multiplication
    pub fn scale(vec: []f32, scalar: f32) void {
        for (vec) |*val| {
            val.* *= scalar;
        }
    }
};

/// Simple data structures for ML
pub const DataStructures = struct {
    /// Training example
    pub const TrainingExample = struct {
        input: []f32,
        target: []f32,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, input: []const f32, target: []const f32) !TrainingExample {
            return .{
                .input = try allocator.dupe(f32, input),
                .target = try allocator.dupe(f32, target),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *TrainingExample) void {
            self.allocator.free(self.input);
            self.allocator.free(self.target);
        }
    };

    /// Dataset for training
    pub const Dataset = struct {
        examples: collections.ArrayList(TrainingExample),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Dataset {
            return .{
                .examples = collections.ArrayList(TrainingExample){},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Dataset) void {
            for (self.examples.items) |*example| {
                example.deinit();
            }
            self.examples.deinit(self.allocator);
        }

        pub fn addExample(self: *Dataset, input: []const f32, target: []const f32) !void {
            const example = try TrainingExample.init(self.allocator, input, target);
            try self.examples.append(self.allocator, example);
        }

        pub fn size(self: *const Dataset) usize {
            return self.examples.items.len;
        }

        pub fn get(self: *const Dataset, index: usize) *TrainingExample {
            const items = self.examples.items;
            if (index >= items.len) @panic("Index out of bounds");
            return &items[index];
        }
    };
};

test "ml components - layer creation" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_size = 4,
        .output_size = 3,
        .activation = .relu,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    try testing.expectEqual(@as(u32, 4), layer.config.input_size);
    try testing.expectEqual(@as(u32, 3), layer.config.output_size);
    try testing.expect(layer.weights != null);
    try testing.expect(layer.biases != null);
}

test "ml components - neural network" {
    const testing = std.testing;

    var network = NeuralNetwork.init(testing.allocator);
    defer network.deinit();

    try network.addLayer(.{
        .layer_type = .dense,
        .input_size = 2,
        .output_size = 3,
        .activation = .relu,
    });

    try network.addLayer(.{
        .layer_type = .dense,
        .input_size = 3,
        .output_size = 1,
        .activation = .sigmoid,
    });

    const input = [_]f32{ 0.5, -0.3 };
    var output = [_]f32{0.0};

    try network.forward(&input, &output);
    // Output should be between 0 and 1 due to sigmoid activation
    try testing.expect(output[0] >= 0.0 and output[0] <= 1.0);
}

test "ml components - vector operations" {
    const testing = std.testing;

    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0 };

    const dot_result = VectorOps.dot(&a, &b);
    try testing.expectEqual(@as(f32, 20.0), dot_result); // 1*2 + 2*3 + 3*4 = 20

    const norm_a = VectorOps.norm(&a);
    try testing.expect(@abs(norm_a - @sqrt(14.0)) < 0.001);
}
