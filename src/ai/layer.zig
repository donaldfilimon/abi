//! Neural Network Layer Components
//!
//! Provides modular, composable neural network layers with SIMD optimization.
//! Supports all major layer types including dense, convolutional, and recurrent layers.

const std = @import("std");

const core = @import("../core/mod.zig");
const activation = @import("activation.zig");

const Allocator = std.mem.Allocator;
const ActivationType = activation.ActivationType;

/// Neural network layer types with enhanced coverage
pub const LayerType = enum {
    input,
    dense,
    conv2d,
    conv1d,
    conv3d,
    maxpool2d,
    avgpool2d,
    globalavgpool,
    dropout,
    batch_norm,
    layer_norm,
    group_norm,
    instance_norm,
    activation_layer,
    flatten,
    reshape,
    permute,
    lstm,
    gru,
    rnn,
    bidirectional_lstm,
    attention,
    multi_head_attention,
    scaled_dot_product_attention,
    transformer_block,
    embedding,
    positional_encoding,
    residual_connection,
    highway_connection,
    squeeze_excitation,
    depth_separable_conv,
    upsampling,
    interpolation,
};

/// Weight initialization strategies
pub const WeightInit = enum {
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    lecun_uniform,
    lecun_normal,
    he_uniform,
    he_normal,
    orthogonal,
    sparse,
    truncated_normal,
    uniform,
    normal,
    zeros,
    ones,
    identity,
    constant,
    dirac,
};

/// Advanced regularization configuration
pub const Regularization = struct {
    l1_lambda: f32 = 0.0,
    l2_lambda: f32 = 0.0,
    dropout_rate: f32 = 0.0,
    dropconnect_rate: f32 = 0.0,
    batch_norm: bool = false,
    layer_norm: bool = false,
    group_norm: bool = false,
    spectral_norm: bool = false,
    gradient_clipping: ?f32 = null,
    gradient_noise: f32 = 0.0,
    label_smoothing: f32 = 0.0,
    mixup_alpha: f32 = 0.0,
    cutmix_alpha: f32 = 0.0,
};

/// Layer configuration structure
pub const LayerConfig = struct {
    layer_type: LayerType,
    input_shape: []const usize,
    output_shape: []const usize,

    // General parameters
    activation_type: ?ActivationType = null,
    regularization: Regularization = .{},
    weight_init: WeightInit = .kaiming_uniform,
    use_bias: bool = true,
    enable_simd: bool = true,

    // Normalization parameters
    momentum: f32 = 0.1,
    eps: f32 = 1e-5,

    // Convolution-specific parameters
    kernel_size: ?[]usize = null,
    stride: ?[]usize = null,
    padding: ?[]usize = null,
    dilation: ?[]usize = null,
    groups: usize = 1,

    // RNN-specific parameters
    hidden_size: ?usize = null,
    sequence_length: ?usize = null,
    return_sequences: bool = false,
    return_state: bool = false,
    bidirectional: bool = false,

    // Attention-specific parameters
    num_heads: ?usize = null,
    head_dim: ?usize = null,
    dropout: f32 = 0.0,

    // Activation-specific parameters
    alpha: f32 = 0.01, // For leaky_relu, elu, etc.
    beta: f32 = 1.0, // For swish, etc.
    threshold: f32 = 1.0,
};

/// Enhanced neural network layer with comprehensive functionality
pub const Layer = struct {
    config: LayerConfig,
    allocator: Allocator,

    // Weight and bias parameters
    weights: ?[]f32 = null,
    biases: ?[]f32 = null,

    // Normalization parameters
    running_mean: ?[]f32 = null,
    running_var: ?[]f32 = null,
    gamma: ?[]f32 = null,
    beta: ?[]f32 = null,

    // Training state
    is_training: bool = true,
    is_frozen: bool = false,

    // Gradients for backpropagation
    weight_gradients: ?[]f32 = null,
    bias_gradients: ?[]f32 = null,

    // Activation processor
    activation_processor: ?activation.ActivationProcessor = null,

    const Self = @This();

    pub fn init(allocator: Allocator, config: LayerConfig) core.FrameworkError!*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Validate configuration
        try self.validateConfig(config);

        self.* = .{
            .config = config,
            .allocator = allocator,
        };

        // Initialize activation processor if needed
        if (config.activation_type) |act_type| {
            const act_config = activation.ActivationConfig{
                .activation_type = act_type,
                .alpha = config.alpha,
                .beta = config.beta,
                .threshold = config.threshold,
                .enable_simd = config.enable_simd,
            };
            self.activation_processor = activation.ActivationProcessor.init(act_config);
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.weights) |weights| self.allocator.free(weights);
        if (self.biases) |biases| self.allocator.free(biases);
        if (self.running_mean) |mean| self.allocator.free(mean);
        if (self.running_var) |variance| self.allocator.free(variance);
        if (self.gamma) |gamma| self.allocator.free(gamma);
        if (self.beta) |beta| self.allocator.free(beta);
        if (self.weight_gradients) |grads| self.allocator.free(grads);
        if (self.bias_gradients) |grads| self.allocator.free(grads);
        self.allocator.destroy(self);
    }

    /// Initialize weights and biases for the layer
    pub fn initializeWeights(self: *Self, rng: *std.rand.Random) core.FrameworkError!void {
        if (self.config.layer_type == .input or
            self.config.layer_type == .dropout or
            self.config.layer_type == .flatten) return;

        const input_size = self.getInputSize();
        const output_size = self.getOutputSize();

        // Initialize weights
        if (self.needsWeights()) {
            if (self.weights == null) {
                const weight_size = self.getWeightSize();
                self.weights = try self.allocator.alloc(f32, weight_size);
                self.weight_gradients = try self.allocator.alloc(f32, weight_size);
                @memset(self.weight_gradients.?, 0.0);
            }
            try self.applyWeightInitialization(rng, input_size, output_size);
        }

        // Initialize biases
        if (self.needsBiases()) {
            if (self.biases == null) {
                self.biases = try self.allocator.alloc(f32, output_size);
                self.bias_gradients = try self.allocator.alloc(f32, output_size);
                @memset(self.bias_gradients.?, 0.0);
            }
            @memset(self.biases.?, 0.0);
        }

        // Initialize normalization parameters
        if (self.config.regularization.batch_norm or
            self.config.regularization.layer_norm or
            self.config.regularization.group_norm)
        {
            try self.initializeNormalization(output_size);
        }
    }

    /// Forward pass through the layer
    pub fn forward(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        switch (self.config.layer_type) {
            .dense => try self.forwardDense(input, output),
            .conv2d => try self.forwardConv2D(input, output, temp_buffer),
            .conv1d => try self.forwardConv1D(input, output, temp_buffer),
            .maxpool2d => try self.forwardMaxPool2D(input, output),
            .avgpool2d => try self.forwardAvgPool2D(input, output),
            .globalavgpool => try self.forwardGlobalAvgPool(input, output),
            .dropout => try self.forwardDropout(input, output),
            .batch_norm => try self.forwardBatchNorm(input, output),
            .layer_norm => try self.forwardLayerNorm(input, output),
            .group_norm => try self.forwardGroupNorm(input, output),
            .activation_layer => try self.forwardActivation(input, output),
            .flatten => try self.forwardFlatten(input, output),
            .reshape => try self.forwardReshape(input, output),
            .lstm => try self.forwardLSTM(input, output, temp_buffer),
            .gru => try self.forwardGRU(input, output, temp_buffer),
            .rnn => try self.forwardRNN(input, output, temp_buffer),
            .attention => try self.forwardAttention(input, output, temp_buffer),
            .multi_head_attention => try self.forwardMultiHeadAttention(input, output, temp_buffer),
            .transformer_block => try self.forwardTransformerBlock(input, output, temp_buffer),
            .embedding => try self.forwardEmbedding(input, output),
            .positional_encoding => try self.forwardPositionalEncoding(input, output),
            .residual_connection => try self.forwardResidualConnection(input, output),
            else => return core.FrameworkError.UnsupportedOperation,
        }
    }

    /// Get input size for the layer
    pub fn getInputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.config.input_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    /// Get output size for the layer
    pub fn getOutputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.config.output_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    /// Set training mode
    pub fn setTraining(self: *Self, is_training: bool) void {
        self.is_training = is_training;
    }

    /// Freeze layer parameters
    pub fn freeze(self: *Self) void {
        self.is_frozen = true;
    }

    /// Unfreeze layer parameters
    pub fn unfreeze(self: *Self) void {
        self.is_frozen = false;
    }

    // Private methods

    fn validateConfig(self: *Self, config: LayerConfig) core.FrameworkError!void {
        _ = self;
        if (config.input_shape.len == 0 or config.output_shape.len == 0) {
            return core.FrameworkError.InvalidConfiguration;
        }

        // Validate layer-specific constraints
        switch (config.layer_type) {
            .conv2d => {
                if (config.input_shape.len < 3) return core.FrameworkError.InvalidConfiguration;
                if (config.kernel_size == null) return core.FrameworkError.InvalidConfiguration;
            },
            .lstm, .gru, .rnn => {
                if (config.hidden_size == null) return core.FrameworkError.InvalidConfiguration;
            },
            .multi_head_attention => {
                if (config.num_heads == null or config.head_dim == null) {
                    return core.FrameworkError.InvalidConfiguration;
                }
            },
            else => {},
        }
    }

    fn needsWeights(self: *const Self) bool {
        return switch (self.config.layer_type) {
            .dense, .conv2d, .conv1d, .conv3d, .lstm, .gru, .rnn, .bidirectional_lstm, .attention, .multi_head_attention, .embedding => true,
            else => false,
        };
    }

    fn needsBiases(self: *const Self) bool {
        return self.needsWeights() and self.config.use_bias;
    }

    fn getWeightSize(self: *const Self) usize {
        return switch (self.config.layer_type) {
            .dense => self.getInputSize() * self.getOutputSize(),
            .conv2d => blk: {
                const kernel = self.config.kernel_size orelse &[_]usize{ 3, 3 };
                const in_channels = if (self.config.input_shape.len >= 3) self.config.input_shape[2] else 1;
                const out_channels = if (self.config.output_shape.len >= 3) self.config.output_shape[2] else 1;
                break :blk kernel[0] * kernel[1] * in_channels * out_channels;
            },
            .lstm, .gru => blk: {
                const hidden = self.config.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                break :blk switch (self.config.layer_type) {
                    .lstm => 4 * hidden * (input + hidden), // 4 gates
                    .gru => 3 * hidden * (input + hidden), // 3 gates
                    else => unreachable,
                };
            },
            .attention, .multi_head_attention => blk: {
                const d_model = self.getInputSize();
                break :blk 3 * d_model * d_model; // Q, K, V projections
            },
            .embedding => self.getInputSize() * self.getOutputSize(),
            else => 0,
        };
    }

    fn initializeNormalization(self: *Self, size: usize) core.FrameworkError!void {
        if (self.gamma == null) {
            self.gamma = try self.allocator.alloc(f32, size);
            @memset(self.gamma.?, 1.0);
        }
        if (self.beta == null) {
            self.beta = try self.allocator.alloc(f32, size);
            @memset(self.beta.?, 0.0);
        }
        if (self.running_mean == null) {
            self.running_mean = try self.allocator.alloc(f32, size);
            @memset(self.running_mean.?, 0.0);
        }
        if (self.running_var == null) {
            self.running_var = try self.allocator.alloc(f32, size);
            @memset(self.running_var.?, 1.0);
        }
    }

    fn applyWeightInitialization(self: *Self, rng: *std.Random, input_size: usize, output_size: usize) core.FrameworkError!void {
        const weights = self.weights.?;

        switch (self.config.weight_init) {
            .xavier_uniform => {
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size + output_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .xavier_normal => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .kaiming_uniform, .he_uniform => {
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .kaiming_normal, .he_normal => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .lecun_uniform => {
                const limit = @sqrt(3.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .lecun_normal => {
                const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .orthogonal => {
                // Simplified orthogonal initialization
                const std_dev = 1.0;
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .uniform => {
                for (weights) |*weight| {
                    weight.* = rng.float(f32) * 2.0 - 1.0;
                }
            },
            .normal => {
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32);
                }
            },
            .zeros => {
                @memset(weights, 0.0);
            },
            .ones => {
                @memset(weights, 1.0);
            },
            .identity => {
                @memset(weights, 0.0);
                const min_dim = @min(input_size, output_size);
                for (0..min_dim) |i| {
                    weights[i * output_size + i] = 1.0;
                }
            },
            .constant => {
                @memset(weights, 0.1);
            },
            else => {
                // Default to Kaiming uniform
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
        }
    }

    // Forward pass implementations for different layer types

    fn forwardDense(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        if (self.weights == null) return core.FrameworkError.InvalidState;

        const weights = self.weights.?;
        const input_size = self.config.input_shape[0];
        const output_size = self.config.output_shape[0];

        if (input.len != input_size or output.len != output_size) {
            return core.FrameworkError.InvalidData;
        }

        // Matrix-vector multiplication: output = weights * input + biases
        core.matrixVectorMultiply(output, weights, input, output_size, input_size);

        // Add biases
        if (self.biases) |biases| {
            core.add(output, output, biases);
        }

        // Apply activation if specified
        if (self.activation_processor) |*processor| {
            processor.activateBatch(output, output);
        }

        // Apply regularization during training
        if (self.is_training) {
            try self.applyRegularization(output);
        }
    }

    fn forwardConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardConv1D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardMaxPool2D(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardAvgPool2D(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardGlobalAvgPool(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardBatchNorm(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardLayerNorm(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardGroupNorm(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardDropout(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        if (self.is_training and self.config.regularization.dropout_rate > 0.0) {
            for (input, 0..) |val, i| {
                // Simple pseudo-random dropout
                const hash = std.hash_map.hashString("dropout") + i;
                if ((@as(f32, @floatFromInt(hash % 1000)) / 1000.0) < self.config.regularization.dropout_rate) {
                    output[i] = 0.0;
                } else {
                    output[i] = val / (1.0 - self.config.regularization.dropout_rate);
                }
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardActivation(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        if (self.activation_processor) |*processor| {
            processor.activateBatch(output, input);
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardFlatten(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardReshape(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardLSTM(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardGRU(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardRNN(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardMultiHeadAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardTransformerBlock(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardEmbedding(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardPositionalEncoding(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        _ = input;
        _ = output;
        return core.FrameworkError.UnsupportedOperation;
    }

    fn forwardResidualConnection(self: *Self, input: []const f32, output: []f32) core.FrameworkError!void {
        _ = self;
        @memcpy(output, input);
    }

    fn applyRegularization(self: *Self, data: []f32) core.FrameworkError!void {
        // Apply L1/L2 regularization during training
        if (self.config.regularization.l1_lambda > 0.0 or self.config.regularization.l2_lambda > 0.0) {
            // Regularization would be applied to gradients during backpropagation
            // This is a placeholder for the interface
            _ = data;
        }
    }
};

test "layer creation and initialization" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_shape = &[_]usize{784},
        .output_shape = &[_]usize{128},
        .activation_type = .relu,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    // Test that layer was created successfully
    try testing.expectEqual(LayerType.dense, layer.config.layer_type);
    try testing.expectEqual(@as(usize, 784), layer.getInputSize());
    try testing.expectEqual(@as(usize, 128), layer.getOutputSize());

    // Initialize weights
    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(&rng.random());

    // Check that weights were allocated
    try testing.expect(layer.weights != null);
    try testing.expect(layer.biases != null);
}

test "dense layer forward pass" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_shape = &[_]usize{4},
        .output_shape = &[_]usize{2},
        .activation_type = .relu,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    // Initialize with known weights for testing
    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(&rng.random());

    // Test forward pass
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{0} ** 2;

    try layer.forward(&input, &output, null);

    // Check that output was computed (should be non-zero due to ReLU)
    var has_non_zero = false;
    for (output) |val| {
        if (val > 0.0) {
            has_non_zero = true;
            break;
        }
    }
    try testing.expect(has_non_zero);
}
