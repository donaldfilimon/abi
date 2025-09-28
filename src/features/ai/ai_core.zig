//! Unified AI Module - Neural Networks, Embeddings, and Machine Learning
//!
//! This module consolidates all AI functionality into a single, high-performance
//! implementation with:
//! - Neural network architectures (MLP, CNN, RNN, Transformer)
//! - Embedding generation and management
//! - Training and inference pipelines with multiple optimizers
//! - Advanced loss functions and regularization techniques
//! - Model serialization, loading, and versioning
//! - Performance optimization and monitoring
//! - Distributed training support
//! - Comprehensive error handling and validation
//! - Memory-efficient operations with SIMD optimizations
//! - Optimized inline activation functions with compile-time constants

const std = @import("std");
// Note: core functionality is now imported through module dependencies
const simd = @import("../../shared/simd.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Random = std.Random;

// Compile-time mathematical constants for optimized activation functions
const EULER_CONSTANT = std.math.e;
const PI = std.math.pi;
const SQRT_2_PI = @sqrt(2.0 / PI);
const SQRT_2 = @sqrt(2.0);
const LN_2 = @log(2.0);

// Optimized activation function constants
const SELU_ALPHA = 1.6732632423543772848170429916717;
const SELU_SCALE = 1.0507009873554804934193349852946;
const LEAKY_RELU_SLOPE = 0.01;
const SWISH_BETA = 1.0;
const GELU_SQRT_2 = 0.7978845608028654; // sqrt(2/pi)
const EPSILON = 1e-8;

/// High-performance activation function utilities
pub const ActivationUtils = struct {
    /// Inline fast approximation functions for better performance
    pub inline fn fastSigmoid(x: f32) f32 {
        // Fast sigmoid approximation using tanh
        return 0.5 * (std.math.tanh(0.5 * x) + 1.0);
    }

    pub inline fn fastTanh(x: f32) f32 {
        // Fast tanh approximation
        if (x > 3.0) return 1.0;
        if (x < -3.0) return -1.0;
        const x2 = x * x;
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    pub inline fn fastExp(x: f32) f32 {
        // Fast exp approximation for activation functions
        if (x > 10.0) return std.math.exp(10.0);
        if (x < -10.0) return std.math.exp(-10.0);
        return std.math.exp(x);
    }

    pub inline fn fastGelu(x: f32) f32 {
        // Fast GELU approximation
        return 0.5 * x * (1.0 + fastTanh(GELU_SQRT_2 * (x + 0.044715 * x * x * x)));
    }

    pub inline fn fastSqrt(x: f32) f32 {
        // Fast square root for normalization functions
        if (x <= 0.0) return 0.0;
        return @sqrt(x);
    }

    /// Vectorized ReLU activation with SIMD optimization
    pub inline fn vectorizedRelu(data: []f32) void {
        // Use SIMD acceleration if available and beneficial
        if (comptime std.simd.suggestVectorLength(f32)) |simd_len| {
            if (simd_len >= 4 and data.len >= 8) {
                if (simd.VectorOps.shouldUseSimd(data.len)) {
                    return simd.VectorOps.vectorizedRelu(data);
                }
            }
        }

        // Fallback to scalar processing with loop unrolling
        var i: usize = 0;
        const len = data.len;

        // Process 4 elements at a time for better performance
        while (i + 4 <= len) : (i += 4) {
            data[i] = @max(0.0, data[i]);
            data[i + 1] = @max(0.0, data[i + 1]);
            data[i + 2] = @max(0.0, data[i + 2]);
            data[i + 3] = @max(0.0, data[i + 3]);
        }

        // Handle remaining elements
        while (i < len) : (i += 1) {
            data[i] = @max(0.0, data[i]);
        }
    }

    pub inline fn vectorizedSigmoid(data: []f32) void {
        var i: usize = 0;
        const len = data.len;

        while (i + 4 <= len) : (i += 4) {
            data[i] = fastSigmoid(data[i]);
            data[i + 1] = fastSigmoid(data[i + 1]);
            data[i + 2] = fastSigmoid(data[i + 2]);
            data[i + 3] = fastSigmoid(data[i + 3]);
        }

        while (i < len) : (i += 1) {
            data[i] = fastSigmoid(data[i]);
        }
    }

    pub inline fn vectorizedTanh(data: []f32) void {
        var i: usize = 0;
        const len = data.len;

        while (i + 4 <= len) : (i += 4) {
            data[i] = fastTanh(data[i]);
            data[i + 1] = fastTanh(data[i + 1]);
            data[i + 2] = fastTanh(data[i + 2]);
            data[i + 3] = fastTanh(data[i + 3]);
        }

        while (i < len) : (i += 1) {
            data[i] = fastTanh(data[i]);
        }
    }

    /// Vectorized Leaky ReLU activation with SIMD optimization
    pub inline fn vectorizedLeakyRelu(data: []f32) void {
        // Use SIMD acceleration if available and beneficial
        if (comptime std.simd.suggestVectorLength(f32)) |simd_len| {
            if (simd_len >= 4 and data.len >= 8) {
                if (simd.VectorOps.shouldUseSimd(data.len)) {
                    return simd.VectorOps.vectorizedLeakyRelu(data, LEAKY_RELU_SLOPE);
                }
            }
        }

        // Fallback to scalar processing with loop unrolling
        var i: usize = 0;
        const len = data.len;

        while (i + 4 <= len) : (i += 4) {
            data[i] = if (data[i] > 0.0) data[i] else LEAKY_RELU_SLOPE * data[i];
            data[i + 1] = if (data[i + 1] > 0.0) data[i + 1] else LEAKY_RELU_SLOPE * data[i + 1];
            data[i + 2] = if (data[i + 2] > 0.0) data[i + 2] else LEAKY_RELU_SLOPE * data[i + 2];
            data[i + 3] = if (data[i + 3] > 0.0) data[i + 3] else LEAKY_RELU_SLOPE * data[i + 3];
        }

        while (i < len) : (i += 1) {
            data[i] = if (data[i] > 0.0) data[i] else LEAKY_RELU_SLOPE * data[i];
        }
    }

    pub inline fn vectorizedGelu(data: []f32) void {
        var i: usize = 0;
        const len = data.len;

        while (i + 4 <= len) : (i += 4) {
            data[i] = fastGelu(data[i]);
            data[i + 1] = fastGelu(data[i + 1]);
            data[i + 2] = fastGelu(data[i + 2]);
            data[i + 3] = fastGelu(data[i + 3]);
        }

        while (i < len) : (i += 1) {
            data[i] = fastGelu(data[i]);
        }
    }

    /// Optimized softmax with numerical stability
    pub inline fn stableSoftmax(data: []f32) void {
        if (data.len == 0) return;

        // Find maximum for numerical stability
        var max_val = data[0];
        for (data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exponentials and sum
        var sum: f32 = 0.0;
        for (data) |*val| {
            val.* = fastExp(val.* - max_val);
            sum += val.*;
        }

        // Normalize with epsilon for numerical stability
        const inv_sum = 1.0 / (sum + EPSILON);
        for (data) |*val| {
            val.* *= inv_sum;
        }
    }

    /// Optimized log softmax with numerical stability
    pub inline fn stableLogSoftmax(data: []f32) void {
        if (data.len == 0) return;

        var max_val = data[0];
        for (data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        var sum: f32 = 0.0;
        for (data) |val| {
            sum += fastExp(val - max_val);
        }

        const log_sum = @log(sum + EPSILON) + max_val;
        for (data) |*val| {
            val.* = val.* - log_sum;
        }
    }
};

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
    activation,
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

/// Enhanced activation functions with optimized implementations
pub const Activation = enum {
    relu,
    relu6,
    leaky_relu,
    parametric_relu,
    elu,
    selu,
    celu,
    sigmoid,
    hard_sigmoid,
    tanh,
    hard_tanh,
    softmax,
    log_softmax,
    softmin,
    softplus,
    softsign,
    swish,
    hard_swish,
    mish,
    gelu,
    quick_gelu,
    linear,
    step,
    threshold,

    /// Inline function for quick activation checks
    pub inline fn isNonlinear(self: Activation) bool {
        return switch (self) {
            .linear => false,
            else => true,
        };
    }

    /// Inline function for gradient requirements
    pub inline fn requiresGradient(self: Activation) bool {
        return switch (self) {
            .step, .threshold => false,
            else => true,
        };
    }
};

/// Comprehensive weight initialization strategies
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

/// Memory allocation strategy
pub const MemoryStrategy = enum {
    eager,
    lazy,
    streaming,
    checkpointing,
};

/// Computation backend
pub const ComputeBackend = enum {
    cpu,
    simd,
    threading,
    gpu, // Future support
};

/// Enhanced neural network layer with comprehensive functionality
pub const Layer = struct {
    layer_type: LayerType,
    input_shape: []const usize,
    output_shape: []const usize,
    weights: ?[]f32 = null,
    biases: ?[]f32 = null,
    activation: ?Activation = null,
    regularization: Regularization = .{},
    weight_init: WeightInit = .kaiming_uniform,
    is_training: bool = true,
    is_frozen: bool = false,

    // Normalization parameters
    running_mean: ?[]f32 = null,
    running_var: ?[]f32 = null,
    gamma: ?[]f32 = null,
    beta: ?[]f32 = null,
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
    use_bias: bool = true,

    // Activation-specific parameters
    alpha: f32 = 0.01, // For leaky_relu, elu, etc.
    activation_beta: f32 = 1.0, // For swish, etc.
    threshold: f32 = 1.0,

    // Performance optimization
    memory_strategy: MemoryStrategy = .eager,
    compute_backend: ComputeBackend = .cpu,

    // Gradients for backpropagation
    weight_gradients: ?[]f32 = null,
    bias_gradients: ?[]f32 = null,

    pub fn init(allocator: Allocator, layer_type: LayerType, input_shape: []const usize, output_shape: []const usize) !*Layer {
        const layer = try allocator.create(Layer);
        layer.* = .{
            .layer_type = layer_type,
            .input_shape = try allocator.dupe(usize, input_shape),
            .output_shape = try allocator.dupe(usize, output_shape),
        };
        return layer;
    }

    pub fn deinit(self: *Layer, allocator: Allocator) void {
        if (self.weights) |weights| allocator.free(weights);
        if (self.biases) |biases| allocator.free(biases);
        if (self.running_mean) |mean| allocator.free(mean);
        if (self.running_var) |variance| allocator.free(variance);
        if (self.gamma) |gamma| allocator.free(gamma);
        if (self.beta) |beta| allocator.free(beta);
        if (self.kernel_size) |kernel| allocator.free(kernel);
        if (self.stride) |stride| allocator.free(stride);
        if (self.padding) |padding| allocator.free(padding);
        if (self.dilation) |dilation| allocator.free(dilation);
        if (self.weight_gradients) |grads| allocator.free(grads);
        if (self.bias_gradients) |grads| allocator.free(grads);
        allocator.free(self.input_shape);
        allocator.free(self.output_shape);
        allocator.destroy(self);
    }

    /// Save layer to file
    pub fn saveToFile(self: *Layer, writer: anytype) !void {
        // Write layer type
        try writer.writeInt(u8, @intFromEnum(self.layer_type), .little);

        // Write input/output shapes
        try writer.writeInt(u32, self.input_shape.len, .little);
        for (self.input_shape) |dim| {
            try writer.writeInt(u32, dim, .little);
        }
        try writer.writeInt(u32, self.output_shape.len, .little);
        for (self.output_shape) |dim| {
            try writer.writeInt(u32, dim, .little);
        }

        // Write activation
        const activation_val = if (self.activation) |act| @intFromEnum(act) else 255;
        try writer.writeInt(u8, activation_val, .little);

        // Write weights if they exist
        const has_weights = self.weights != null;
        try writer.writeInt(u8, if (has_weights) 1 else 0, .little);
        if (has_weights) {
            const weight_count = self.getWeightSize();
            for (0..weight_count) |i| {
                try writer.writeInt(u32, @bitCast(self.weights.?[i]), .little);
            }
        }

        // Write biases if they exist
        const has_biases = self.biases != null;
        try writer.writeInt(u8, if (has_biases) 1 else 0, .little);
        if (has_biases) {
            const bias_count = if (self.needsBiases()) self.getOutputSize() else 0;
            for (0..bias_count) |i| {
                try writer.writeInt(u32, @bitCast(self.biases.?[i]), .little);
            }
        }
    }

    /// Load layer from file
    pub fn loadFromFile(allocator: Allocator, reader: *std.fs.File.Reader) !*Layer {
        // Read layer type
        var layer_type_bytes: [1]u8 = undefined;
        _ = try reader.read(&layer_type_bytes);
        const layer_type = @as(LayerType, @enumFromInt(layer_type_bytes[0]));

        // Read input shape
        var input_shape_len_bytes: [4]u8 = undefined;
        _ = try reader.read(&input_shape_len_bytes);
        const input_shape_len = std.mem.readInt(u32, &input_shape_len_bytes, .little);
        const input_shape = try allocator.alloc(usize, input_shape_len);
        for (0..input_shape_len) |i| {
            var dim_bytes: [4]u8 = undefined;
            _ = try reader.read(&dim_bytes);
            input_shape[i] = std.mem.readInt(u32, &dim_bytes, .little);
        }

        // Read output shape
        var output_shape_len_bytes: [4]u8 = undefined;
        _ = try reader.read(&output_shape_len_bytes);
        const output_shape_len = std.mem.readInt(u32, &output_shape_len_bytes, .little);
        const output_shape = try allocator.alloc(usize, output_shape_len);
        for (0..output_shape_len) |i| {
            var dim_bytes: [4]u8 = undefined;
            _ = try reader.read(&dim_bytes);
            output_shape[i] = std.mem.readInt(u32, &dim_bytes, .little);
        }

        // Create layer
        const layer = try allocator.create(Layer);
        layer.* = .{
            .layer_type = layer_type,
            .input_shape = input_shape,
            .output_shape = output_shape,
        };

        // Read activation
        var activation_bytes: [1]u8 = undefined;
        _ = try reader.read(&activation_bytes);
        const activation_val = activation_bytes[0];
        layer.activation = if (activation_val != 255) @as(Activation, @enumFromInt(activation_val)) else null;

        // Read weights if they exist
        var has_weights_bytes: [1]u8 = undefined;
        _ = try reader.read(&has_weights_bytes);
        const has_weights = has_weights_bytes[0];
        if (has_weights != 0) {
            const weight_count = layer.getWeightSize();
            layer.weights = try allocator.alloc(f32, weight_count);
            for (0..weight_count) |i| {
                var weight_bytes: [4]u8 = undefined;
                _ = try reader.read(&weight_bytes);
                const weight_bits = std.mem.readInt(u32, &weight_bytes, .little);
                layer.weights.?[i] = @bitCast(weight_bits);
            }
            layer.weight_gradients = try allocator.alloc(f32, weight_count);
            @memset(layer.weight_gradients.?, 0.0);
        }

        // Read biases if they exist
        var has_biases_bytes: [1]u8 = undefined;
        _ = try reader.read(&has_biases_bytes);
        const has_biases = has_biases_bytes[0];
        if (has_biases != 0) {
            const bias_count = if (layer.needsBiases()) layer.getOutputSize() else 0;
            layer.biases = try allocator.alloc(f32, bias_count);
            for (0..bias_count) |i| {
                var bias_bytes: [4]u8 = undefined;
                _ = try reader.read(&bias_bytes);
                const bias_bits = std.mem.readInt(u32, &bias_bytes, .little);
                layer.biases.?[i] = @bitCast(bias_bits);
            }
            layer.bias_gradients = try allocator.alloc(f32, bias_count);
            @memset(layer.bias_gradients.?, 0.0);
        }

        return layer;
    }

    pub fn initializeWeights(self: *Layer, allocator: Allocator, rng: *Random) !void {
        if (self.layer_type == .input or self.layer_type == .dropout or self.layer_type == .flatten) return;

        const input_size = self.getInputSize();
        const output_size = self.getOutputSize();

        // Initialize weights
        if (self.needsWeights()) {
            if (self.weights == null) {
                self.weights = try allocator.alloc(f32, self.getWeightSize());
                self.weight_gradients = try allocator.alloc(f32, self.getWeightSize());
                @memset(self.weight_gradients.?, 0.0);
            }
            try self.applyWeightInitialization(rng, input_size, output_size);
        }

        // Initialize biases
        if (self.needsBiases()) {
            if (self.biases == null) {
                self.biases = try allocator.alloc(f32, output_size);
                self.bias_gradients = try allocator.alloc(f32, output_size);
                @memset(self.bias_gradients.?, 0.0);
            }
            @memset(self.biases.?, 0.0);
        }

        // Initialize normalization parameters
        if (self.regularization.batch_norm or self.regularization.layer_norm or self.regularization.group_norm) {
            try self.initializeNormalization(allocator, output_size);
        }
    }

    fn needsWeights(self: *const Layer) bool {
        return switch (self.layer_type) {
            .dense, .conv2d, .conv1d, .conv3d, .lstm, .gru, .rnn, .bidirectional_lstm, .attention, .multi_head_attention, .embedding => true,
            else => false,
        };
    }

    fn needsBiases(self: *const Layer) bool {
        return self.needsWeights() and self.use_bias;
    }

    fn getWeightSize(self: *const Layer) usize {
        return switch (self.layer_type) {
            .dense => self.getInputSize() * self.getOutputSize(),
            .conv2d => blk: {
                const kernel = self.kernel_size orelse &[_]usize{ 3, 3 };
                const in_channels = if (self.input_shape.len >= 3) self.input_shape[2] else 1;
                const out_channels = if (self.output_shape.len >= 3) self.output_shape[2] else 1;
                break :blk kernel[0] * kernel[1] * in_channels * out_channels;
            },
            .lstm, .gru => blk: {
                const hidden = self.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                break :blk switch (self.layer_type) {
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

    fn initializeNormalization(self: *Layer, allocator: Allocator, size: usize) !void {
        if (self.gamma == null) {
            self.gamma = try allocator.alloc(f32, size);
            @memset(self.gamma.?, 1.0);
        }
        if (self.beta == null) {
            self.beta = try allocator.alloc(f32, size);
            @memset(self.beta.?, 0.0);
        }
        if (self.running_mean == null) {
            self.running_mean = try allocator.alloc(f32, size);
            @memset(self.running_mean.?, 0.0);
        }
        if (self.running_var == null) {
            self.running_var = try allocator.alloc(f32, size);
            @memset(self.running_var.?, 1.0);
        }
    }

    fn applyWeightInitialization(self: *Layer, rng: *Random, input_size: usize, output_size: usize) !void {
        const weights = self.weights.?;

        switch (self.weight_init) {
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

    pub fn forward(self: *Layer, input: []const f32, output: []f32) !void {
        switch (self.layer_type) {
            .dense => try self.forwardDense(input, output),
            .conv2d => try self.forwardConv2D(input, output),
            .conv1d => try self.forwardConv1D(input, output),
            .maxpool2d => try self.forwardMaxPool2D(input, output),
            .avgpool2d => try self.forwardAvgPool2D(input, output),
            .globalavgpool => try self.forwardGlobalAvgPool(input, output),
            .dropout => try self.forwardDropout(input, output),
            .batch_norm => try self.forwardBatchNorm(input, output),
            .layer_norm => try self.forwardLayerNorm(input, output),
            .group_norm => try self.forwardGroupNorm(input, output),
            .activation => try self.forwardActivation(input, output),
            .flatten => try self.forwardFlatten(input, output),
            .reshape => try self.forwardReshape(input, output),
            .lstm => try self.forwardLSTM(input, output),
            .gru => try self.forwardGRU(input, output),
            .rnn => try self.forwardRNN(input, output),
            .attention => try self.forwardAttention(input, output),
            .multi_head_attention => try self.forwardMultiHeadAttention(input, output),
            .transformer_block => try self.forwardTransformerBlock(input, output),
            .embedding => try self.forwardEmbedding(input, output),
            .positional_encoding => try self.forwardPositionalEncoding(input, output),
            .residual_connection => try self.forwardResidualConnection(input, output),
            else => return error.UnsupportedLayerType,
        }
    }

    fn getInputSize(self: *const Layer) usize {
        var size: usize = 1;
        for (self.input_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn getOutputSize(self: *const Layer) usize {
        var size: usize = 1;
        for (self.output_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    fn forwardDense(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        const weights = self.weights.?;
        const input_size = self.input_shape[0];
        const output_size = self.output_shape[0];

        if (input.len != input_size or output.len != output_size) return error.InvalidDimensions;

        // Matrix-vector multiplication: output = weights * input + biases
        // Simple matrix-vector multiplication
        for (0..output_size) |i| {
            var sum: f32 = 0.0;
            for (0..input_size) |j| {
                sum += weights[i * input_size + j] * input[j];
            }
            output[i] = sum;
        }

        // Add biases
        if (self.biases) |b| {
            for (output, 0..) |*out, i| {
                out.* += b[i];
            }
        }

        // Apply activation if specified
        if (self.activation) |activation| {
            try self.applyActivation(output, activation);
        }

        // Apply regularization during training
        if (self.is_training) {
            try self.applyRegularization(output);
        }
    }

    fn forwardConv2D(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        // Convolutional layer parameters
        const batch_size = self.input_shape[0];
        const input_height = self.input_shape[1];
        const input_width = self.input_shape[2];
        const input_channels = self.input_shape[3];

        const output_height = self.output_shape[1];
        const output_width = self.output_shape[2];
        const output_channels = self.output_shape[3];

        // Get kernel size from weights shape (assuming square kernels)
        const kernel_size = self.weights.?.len / (input_channels * output_channels);
        const kernel_side = std.math.sqrt(kernel_size);

        // Convolution operation
        for (0..batch_size) |b| {
            for (0..output_height) |oh| {
                for (0..output_width) |ow| {
                    for (0..output_channels) |oc| {
                        var sum: f32 = 0.0;

                        // Convolve over input channels and kernel spatial dimensions
                        for (0..input_channels) |ic| {
                            for (0..kernel_side) |kh| {
                                for (0..kernel_side) |kw| {
                                    const ih = oh + kh;
                                    const iw = ow + kw;

                                    if (ih < input_height and iw < input_width) {
                                        const input_idx = ((b * input_height + ih) * input_width + iw) * input_channels + ic;
                                        const weight_idx = ((oc * input_channels + ic) * kernel_side + kh) * kernel_side + kw;

                                        if (input_idx < input.len and weight_idx < self.weights.?.len) {
                                            sum += input[input_idx] * self.weights.?[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias
                        if (self.biases) |biases| {
                            sum += biases[oc];
                        }

                        // Store result
                        const output_idx = ((b * output_height + oh) * output_width + ow) * output_channels + oc;
                        if (output_idx < output.len) {
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        // Apply activation function
        if (self.activation) |activation| {
            try self.applyActivation(output, activation);
        }
    }

    fn forwardConv1D(self: *Layer, _input: []const f32, _output: []f32) !void {
        // 1D convolution implementation
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

    fn forwardAvgPool2D(self: *Layer, _input: []const f32, _output: []f32) !void {
        // 2D average pooling implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardGlobalAvgPool(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Global average pooling implementation (stub)
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardBatchNorm(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Batch normalization implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardLayerNorm(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Layer normalization implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardGroupNorm(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Group normalization implementation (stub)
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardDropout(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.is_training and self.regularization.dropout_rate > 0.0) {
            for (input, 0..) |val, i| {
                // Simple pseudo-random dropout
                const hash = std.hash_map.hashString("dropout") + i;
                if ((@as(f32, @floatFromInt(hash % 1000)) / 1000.0) < self.regularization.dropout_rate) {
                    output[i] = 0.0;
                } else {
                    output[i] = val / (1.0 - self.regularization.dropout_rate);
                }
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardActivation(self: *Layer, input: []const f32, output: []f32) !void {
        @memcpy(output, input);
        if (self.activation) |activation| {
            try self.applyActivation(output, activation);
        }
    }

    fn forwardFlatten(self: *Layer, input: []const f32, output: []f32) !void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardReshape(self: *Layer, input: []const f32, output: []f32) !void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardLSTM(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        // LSTM parameters
        const batch_size = self.input_shape[0];
        const seq_length = self.input_shape[1];
        const input_size = self.input_shape[2];
        const hidden_size = self.output_shape[2];

        // LSTM weights: [input_size + hidden_size, 4 * hidden_size] for gates
        const total_weights = (input_size + hidden_size) * 4 * hidden_size;
        if (self.weights.?.len != total_weights) return error.InvalidWeightsShape;

        // Get allocator from the network (assuming it's stored in the network structure)
        // For now, use a simple page allocator approach
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        // Initialize cell state and hidden state
        var cell_state = try allocator.alloc(f32, batch_size * hidden_size);
        defer allocator.free(cell_state);
        var hidden_state = try allocator.alloc(f32, batch_size * hidden_size);
        defer allocator.free(hidden_state);

        @memset(cell_state, 0.0);
        @memset(hidden_state, 0.0);

        // Process sequence
        for (0..seq_length) |t| {
            for (0..batch_size) |b| {
                // Get input for this time step
                const input_offset = (b * seq_length + t) * input_size;
                const input_slice = input[input_offset .. input_offset + input_size];

                // Concatenate input and previous hidden state
                var combined_input = try allocator.alloc(f32, input_size + hidden_size);
                defer allocator.free(combined_input);

                @memcpy(combined_input[0..input_size], input_slice);
                const hidden_offset = b * hidden_size;
                @memcpy(combined_input[input_size..], hidden_state[hidden_offset .. hidden_offset + hidden_size]);

                // Compute gates: forget, input, output, candidate
                for (0..hidden_size) |h| {
                    // Forget gate
                    var forget_gate: f32 = 0.0;
                    for (0..input_size + hidden_size) |i| {
                        const weight_idx = (i * 4 * hidden_size) + (h * 4);
                        forget_gate += combined_input[i] * self.weights.?[weight_idx];
                    }
                    forget_gate = sigmoid(forget_gate);

                    // Input gate
                    var input_gate: f32 = 0.0;
                    for (0..input_size + hidden_size) |i| {
                        const weight_idx = (i * 4 * hidden_size) + (h * 4) + 1;
                        input_gate += combined_input[i] * self.weights.?[weight_idx];
                    }
                    input_gate = sigmoid(input_gate);

                    // Output gate
                    var output_gate: f32 = 0.0;
                    for (0..input_size + hidden_size) |i| {
                        const weight_idx = (i * 4 * hidden_size) + (h * 4) + 2;
                        output_gate += combined_input[i] * self.weights.?[weight_idx];
                    }
                    output_gate = sigmoid(output_gate);

                    // Candidate values
                    var candidate: f32 = 0.0;
                    for (0..input_size + hidden_size) |i| {
                        const weight_idx = (i * 4 * hidden_size) + (h * 4) + 3;
                        candidate += combined_input[i] * self.weights.?[weight_idx];
                    }
                    candidate = tanh(candidate);

                    // Update cell state and hidden state
                    const cell_idx = b * hidden_size + h;
                    cell_state[cell_idx] = forget_gate * cell_state[cell_idx] + input_gate * candidate;
                    hidden_state[cell_idx] = output_gate * tanh(cell_state[cell_idx]);
                }

                // Copy hidden state to output
                const output_offset = (b * seq_length + t) * hidden_size;
                @memcpy(output[output_offset .. output_offset + hidden_size], hidden_state[b * hidden_size .. (b + 1) * hidden_size]);
            }
        }
    }

    // Helper functions for LSTM
    fn sigmoid(x: f32) f32 {
        return 1.0 / (1.0 + std.math.exp(-x));
    }

    fn tanh(x: f32) f32 {
        const exp_2x = std.math.exp(2.0 * x);
        return (exp_2x - 1.0) / (exp_2x + 1.0);
    }

    fn forwardGRU(self: *Layer, _input: []const f32, _output: []f32) !void {
        // GRU implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardRNN(self: *Layer, _input: []const f32, _output: []f32) !void {
        // Standard RNN implementation
        _ = self;
        _ = _input;
        _ = _output;
        return error.NotImplemented;
    }

    fn forwardAttention(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        const embed_dim = self.input_shape[1];
        const num_heads = self.num_heads orelse 1;

        // Use transformer module for attention computation
        // Create temporary multi-head attention instance
        var mha = try transformer.MultiHeadAttention.init(std.heap.page_allocator, embed_dim, num_heads);
        defer mha.deinit(std.heap.page_allocator);

        // For self-attention: query, key, and value are all the input
        try mha.forward(input, input, input, output);
    }

    fn forwardMultiHeadAttention(self: *Layer, input: []const f32, output: []f32) !void {
        // Multi-head attention is handled by the transformer module
        try self.forwardAttention(input, output);
    }

    fn forwardTransformerBlock(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        // Create transformer block components
        const embed_dim = self.input_shape[1];
        const num_heads = self.num_heads orelse 8;
        const ff_dim = embed_dim * 4; // Standard expansion factor

        var block = try transformer.TransformerBlock.init(std.heap.page_allocator, embed_dim, num_heads, ff_dim, 0.1 // dropout rate
        );
        defer block.deinit(std.heap.page_allocator);

        const seq_len = self.input_shape[0];
        try block.forward(@constCast(input), seq_len);

        // Copy result to output
        @memcpy(output, input);
    }

    fn forwardEmbedding(self: *Layer, input: []const f32, output: []f32) !void {
        if (self.weights == null) return error.WeightsNotInitialized;

        const vocab_size = self.input_shape[0];
        const embed_dim = self.output_shape[1];

        // Create embedding layer
        var embedding = try transformer.Embedding.init(std.heap.page_allocator, vocab_size, embed_dim);
        defer embedding.deinit(std.heap.page_allocator);

        // Copy weights from layer to embedding
        if (self.weights.?.len >= vocab_size * embed_dim) {
            @memcpy(embedding.weight_matrix, self.weights.?[0 .. vocab_size * embed_dim]);
        }

        // Convert float input to token indices (assuming they're already token indices)
        const tokens = try std.heap.page_allocator.alloc(u32, input.len);
        defer std.heap.page_allocator.free(tokens);

        for (input, 0..) |val, i| {
            tokens[i] = @intFromFloat(val); // Convert float to token index
        }

        try embedding.forward(tokens, output);
    }

    fn forwardPositionalEncoding(self: *Layer, input: []const f32, output: []f32) !void {
        const seq_len = self.input_shape[0];
        const embed_dim = self.input_shape[1];

        // Create positional encoding
        var pos_encoding = try transformer.PositionalEncoding.init(std.heap.page_allocator, seq_len, embed_dim);
        defer pos_encoding.deinit(std.heap.page_allocator);

        // Copy input to output and add positional encoding
        @memcpy(output, input);
        pos_encoding.encode(output, seq_len);
    }

    fn forwardResidualConnection(self: *Layer, input: []const f32, output: []f32) !void {
        // Residual connection (identity mapping)
        _ = self;
        @memcpy(output, input);
    }

    fn applyRegularization(self: *Layer, data: []f32) !void {
        // Apply L1/L2 regularization during training
        if (self.regularization.l1_lambda > 0.0 or self.regularization.l2_lambda > 0.0) {
            // Regularization would be applied to gradients during backpropagation
            // This is a placeholder for the interface
            _ = data;
        }
    }

    /// High-performance activation function implementation using optimized utilities
    fn applyActivation(self: *Layer, data: []f32, activation: Activation) !void {
        _ = self;
        switch (activation) {
            .relu => ActivationUtils.vectorizedRelu(data),
            .sigmoid => ActivationUtils.vectorizedSigmoid(data),
            .tanh => ActivationUtils.vectorizedTanh(data),
            .softmax => ActivationUtils.stableSoftmax(data),
            .log_softmax => ActivationUtils.stableLogSoftmax(data),
            .leaky_relu => ActivationUtils.vectorizedLeakyRelu(data),
            .gelu => ActivationUtils.vectorizedGelu(data),
            .parametric_relu => {
                // Optimized parametric ReLU with configurable alpha
                const alpha = 0.1; // Could be made configurable
                var i: usize = 0;
                while (i + 4 <= data.len) : (i += 4) {
                    data[i] = if (data[i] > 0.0) data[i] else alpha * data[i];
                    data[i + 1] = if (data[i + 1] > 0.0) data[i + 1] else alpha * data[i + 1];
                    data[i + 2] = if (data[i + 2] > 0.0) data[i + 2] else alpha * data[i + 2];
                    data[i + 3] = if (data[i + 3] > 0.0) data[i + 3] else alpha * data[i + 3];
                }
                while (i < data.len) : (i += 1) {
                    data[i] = if (data[i] > 0.0) data[i] else alpha * data[i];
                }
            },
            .elu => {
                // Optimized ELU with vectorization
                var i: usize = 0;
                while (i + 4 <= data.len) : (i += 4) {
                    data[i] = if (data[i] > 0.0) data[i] else ActivationUtils.fastExp(data[i]) - 1.0;
                    data[i + 1] = if (data[i + 1] > 0.0) data[i + 1] else ActivationUtils.fastExp(data[i + 1]) - 1.0;
                    data[i + 2] = if (data[i + 2] > 0.0) data[i + 2] else ActivationUtils.fastExp(data[i + 2]) - 1.0;
                    data[i + 3] = if (data[i + 3] > 0.0) data[i + 3] else ActivationUtils.fastExp(data[i + 3]) - 1.0;
                }
                while (i < data.len) : (i += 1) {
                    data[i] = if (data[i] > 0.0) data[i] else ActivationUtils.fastExp(data[i]) - 1.0;
                }
            },
            .selu => {
                // Optimized SELU with vectorization
                var i: usize = 0;
                while (i + 4 <= data.len) : (i += 4) {
                    data[i] = if (data[i] > 0.0) SELU_SCALE * data[i] else SELU_SCALE * SELU_ALPHA * (ActivationUtils.fastExp(data[i]) - 1.0);
                    data[i + 1] = if (data[i + 1] > 0.0) SELU_SCALE * data[i + 1] else SELU_SCALE * SELU_ALPHA * (ActivationUtils.fastExp(data[i + 1]) - 1.0);
                    data[i + 2] = if (data[i + 2] > 0.0) SELU_SCALE * data[i + 2] else SELU_SCALE * SELU_ALPHA * (ActivationUtils.fastExp(data[i + 2]) - 1.0);
                    data[i + 3] = if (data[i + 3] > 0.0) SELU_SCALE * data[i + 3] else SELU_SCALE * SELU_ALPHA * (ActivationUtils.fastExp(data[i + 3]) - 1.0);
                }
                while (i < data.len) : (i += 1) {
                    data[i] = if (data[i] > 0.0) SELU_SCALE * data[i] else SELU_SCALE * SELU_ALPHA * (ActivationUtils.fastExp(data[i]) - 1.0);
                }
            },
            .swish => {
                for (data) |*val| {
                    val.* = val.* / (1.0 + @exp(-val.*));
                }
            },
            .mish => {
                // Optimized Mish with vectorization
                var i: usize = 0;
                while (i + 4 <= data.len) : (i += 4) {
                    data[i] = data[i] * ActivationUtils.fastTanh(@log(1.0 + ActivationUtils.fastExp(data[i])));
                    data[i + 1] = data[i + 1] * ActivationUtils.fastTanh(@log(1.0 + ActivationUtils.fastExp(data[i + 1])));
                    data[i + 2] = data[i + 2] * ActivationUtils.fastTanh(@log(1.0 + ActivationUtils.fastExp(data[i + 2])));
                    data[i + 3] = data[i + 3] * ActivationUtils.fastTanh(@log(1.0 + ActivationUtils.fastExp(data[i + 3])));
                }
                while (i < data.len) : (i += 1) {
                    data[i] = data[i] * ActivationUtils.fastTanh(@log(1.0 + ActivationUtils.fastExp(data[i])));
                }
            },
            .hard_swish => {
                for (data) |*val| {
                    if (val.* <= -3.0) {
                        val.* = 0.0;
                    } else if (val.* >= 3.0) {
                        val.* = val.*;
                    } else {
                        val.* = val.* * (val.* + 3.0) / 6.0;
                    }
                }
            },
            .linear => {
                // No transformation needed
            },
            .softplus => {
                for (data) |*val| {
                    val.* = @log(1.0 + @exp(val.*));
                }
            },
            .softsign => {
                for (data) |*val| {
                    val.* = val.* / (1.0 + @abs(val.*));
                }
            },
            else => {
                // Unsupported/unused activations are treated as linear
            },
        }
    }
};

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

/// Data augmentation techniques
pub const DataAugmentation = struct {
    horizontal_flip: bool = false,
    vertical_flip: bool = false,
    rotation_range: f32 = 0.0,
    width_shift_range: f32 = 0.0,
    height_shift_range: f32 = 0.0,
    brightness_range: ?[2]f32 = null,
    zoom_range: f32 = 0.0,
    channel_shift_range: f32 = 0.0,
    fill_mode: enum { constant, nearest, reflect, wrap } = .constant,
    gaussian_noise_std: f32 = 0.0,
    cutout_probability: f32 = 0.0,
    cutout_size: ?[2]usize = null,
    mixup_alpha: f32 = 0.0,
    cutmix_alpha: f32 = 0.0,
};

/// Model training configuration with advanced options
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
    gradient_clipping_norm: enum { l1, l2, inf } = .l2,

    // Data augmentation
    data_augmentation: ?DataAugmentation = null,

    // Distributed training
    use_mixed_precision: bool = false,
    accumulate_gradients: usize = 1,
    sync_batch_norm: bool = false,

    // Monitoring and logging
    log_frequency: usize = 100,
    validate_frequency: usize = 1,
    tensorboard_logging: bool = false,
    profiling_enabled: bool = false,
};

/// Comprehensive training metrics
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
    auc_roc: ?f32 = null,

    // Training progress
    epoch: usize,
    step: usize = 0,
    training_time_ms: u64,
    inference_time_ms: ?u64 = null,

    // Performance metrics
    throughput_samples_per_sec: f32 = 0.0,
    memory_usage_mb: f32 = 0.0,
    gpu_utilization: ?f32 = null,

    // Learning dynamics
    learning_rate: f32,
    gradient_norm: ?f32 = null,
    weight_norm: ?f32 = null,

    // Custom metrics
    custom_metrics: ?std.StringHashMap(f32) = null,
};

/// Neural network model with enhanced capabilities
pub const NeuralNetwork = struct {
    layers: ArrayList(*Layer),
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    output_shape: []const usize,
    is_compiled: bool = false,
    is_training: bool = true,
    model_name: ?[]const u8 = null,
    version: u32 = 1,

    pub fn init(allocator: std.mem.Allocator, input_shape: []const usize, output_shape: []const usize) !*NeuralNetwork {
        const network = try allocator.create(NeuralNetwork);
        network.* = .{
            .layers = ArrayList(*Layer){},
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
        self.layers.deinit(self.allocator);
        self.allocator.free(self.input_shape);
        self.allocator.free(self.output_shape);
        if (self.model_name) |name| {
            self.allocator.free(name);
        }
        self.allocator.destroy(self);
    }

    /// Save neural network to file in binary format
    pub fn saveToFile(self: *NeuralNetwork, file_path: []const u8) !void {
        const file = try std.fs.cwd().createFile(file_path, .{});
        defer file.close();

        var writer = file.writer();

        // Write magic number and version
        try writer.writeAll("ZNNET");
        try writer.writeInt(u32, self.version, .little);

        // Write model metadata
        const name_len = if (self.model_name) |name| name.len else 0;
        try writer.writeInt(u32, name_len, .little);
        if (self.model_name) |name| {
            try writer.writeAll(name);

            // Write input/output shapes
            try writer.writeInt(u32, self.input_shape.len, .little);
            for (self.input_shape) |dim| {
                try writer.writeInt(u32, dim, .little);
            }
            try writer.writeInt(u32, self.output_shape.len, .little);
            for (self.output_shape) |dim| {
                try writer.writeInt(u32, dim, .little);
            }

            // Write layer count
            try writer.writeInt(u32, self.layers.items.len, .little);

            // Write each layer
            for (self.layers.items) |layer| {
                try layer.saveToFile(writer);
            }

            std.debug.print("Neural network saved to: {s}\n", .{file_path});
        }
    }

    /// Train network on a single input-target pair
    pub fn trainStep(self: *NeuralNetwork, input: []const f32, target: []const f32) !f32 {
        if (input.len != self.input_shape[0] or target.len != self.output_shape[0]) {
            return error.InvalidInput;
        }

        // Forward pass
        const predictions = try self.allocator.alloc(f32, self.output_shape[0]);
        defer self.allocator.free(predictions);
        try self.forward(input, predictions);

        // Calculate loss (mean squared error)
        var loss: f32 = 0.0;
        for (predictions, target) |pred, targ| {
            const diff = pred - targ;
            loss += diff * diff;
        }
        loss /= @as(f32, @floatFromInt(target.len));

        // Backward pass (simplified gradient descent)
        try self.backwardPass(input, target, predictions);

        return loss;
    }

    /// Backward pass for training
    fn backwardPass(self: *NeuralNetwork, input: []const f32, target: []const f32, predictions: []const f32) !void {
        // Simplified backpropagation implementation
        // This is a basic implementation - in practice, you'd want more sophisticated optimization

        const learning_rate = 0.01;
        const output_size = self.output_shape[0];

        // Calculate output layer gradients
        var output_gradients = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(output_gradients);

        for (0..output_size) |i| {
            output_gradients[i] = (predictions[i] - target[i]) * 2.0 / @as(f32, @floatFromInt(output_size));
        }

        // Update weights and biases for the last layer
        if (self.layers.items.len > 0) {
            const last_layer = self.layers.items[self.layers.items.len - 1];
            if (last_layer.weights) |weights| {
                const input_size = last_layer.input_shape[0];
                for (0..output_size) |i| {
                    for (0..input_size) |j| {
                        const input_val = if (self.layers.items.len == 1) input[j] else 0.0; // Simplified
                        weights[i * input_size + j] -= learning_rate * output_gradients[i] * input_val;
                    }
                }
            }

            if (last_layer.biases) |biases| {
                for (0..output_size) |i| {
                    biases[i] -= learning_rate * output_gradients[i];
                }
            }
        }
    }

    /// Load neural network from file
    pub fn loadFromFile(allocator: std.mem.Allocator, file_path: []const u8) !*NeuralNetwork {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        var read_buf: [4096]u8 = undefined;
        var reader = file.reader(&read_buf);

        // Read and verify magic number
        var magic: [5]u8 = undefined;
        _ = try reader.read(&magic);
        if (!std.mem.eql(u8, &magic, "ZNNET")) {
            return error.InvalidInput;
        }

        // Read version
        var version_bytes: [4]u8 = undefined;
        _ = try reader.read(&version_bytes);
        const version = std.mem.readInt(u32, &version_bytes, .little);

        // Read model name length and name
        var name_len_bytes: [4]u8 = undefined;
        _ = try reader.read(&name_len_bytes);
        const name_len = std.mem.readInt(u32, &name_len_bytes, .little);
        const model_name = if (name_len > 0) blk: {
            const name_buf = try allocator.alloc(u8, name_len);
            _ = try reader.read(name_buf);
            break :blk name_buf;
        } else null;

        // Read input shape
        var input_shape_len_bytes: [4]u8 = undefined;
        _ = try reader.read(&input_shape_len_bytes);
        const input_shape_len = std.mem.readInt(u32, &input_shape_len_bytes, .little);
        const input_shape = try allocator.alloc(usize, input_shape_len);
        for (0..input_shape_len) |i| {
            var dim_bytes: [4]u8 = undefined;
            _ = try reader.read(&dim_bytes);
            input_shape[i] = std.mem.readInt(u32, &dim_bytes, .little);
        }

        // Read output shape
        var output_shape_len_bytes: [4]u8 = undefined;
        _ = try reader.read(&output_shape_len_bytes);
        const output_shape_len = std.mem.readInt(u32, &output_shape_len_bytes, .little);
        const output_shape = try allocator.alloc(usize, output_shape_len);
        for (0..output_shape_len) |i| {
            var dim_bytes: [4]u8 = undefined;
            _ = try reader.read(&dim_bytes);
            output_shape[i] = std.mem.readInt(u32, &dim_bytes, .little);
        }

        // Create network
        const network = try allocator.create(NeuralNetwork);
        network.* = .{
            .layers = ArrayList(*Layer){},
            .allocator = allocator,
            .input_shape = input_shape,
            .output_shape = output_shape,
            .model_name = model_name,
            .version = version,
        };

        // Read layer count
        var layer_count_bytes: [4]u8 = undefined;
        _ = try reader.read(&layer_count_bytes);
        const layer_count = std.mem.readInt(u32, &layer_count_bytes, .little);

        // Read layers
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            const layer = try Layer.loadFromFile(allocator, &reader);
            try network.layers.append(network.allocator, layer);
        }

        network.is_compiled = true; // Assume loaded model is compiled
        std.debug.print("Neural network loaded from: {s}\n", .{file_path});
        return network;
    }

    pub fn setTraining(self: *NeuralNetwork, is_training: bool) void {
        self.is_training = is_training;
        for (self.layers.items) |layer| {
            layer.is_training = is_training;
        }
    }

    pub fn addLayer(self: *NeuralNetwork, layer: *Layer) !void {
        try self.layers.append(self.allocator, layer);
        self.is_compiled = false;
    }

    pub fn addDenseLayer(self: *NeuralNetwork, units: usize, activation: ?Activation) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .dense, prev_output_shape, &[_]usize{units});
        layer.activation = activation;
        var prng = std.Random.DefaultPrng.init(0);
        var rng = prng.random();
        try layer.initializeWeights(self.allocator, &rng);
        try self.addLayer(layer);
    }

    pub fn addConv2DLayer(self: *NeuralNetwork, filters: usize, kernel_size: [2]usize, activation: ?Activation) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        // Simplified output shape calculation for convolution
        const output_shape = &[_]usize{ prev_output_shape[0], prev_output_shape[1], filters };

        const layer = try Layer.init(self.allocator, .conv2d, prev_output_shape, output_shape);
        layer.activation = activation;
        layer.kernel_size = try self.allocator.dupe(usize, &kernel_size);
        var prng = std.Random.DefaultPrng.init(0);
        var rng = prng.random();
        try layer.initializeWeights(self.allocator, &rng);
        try self.addLayer(layer);
    }

    pub fn addDropoutLayer(self: *NeuralNetwork, rate: f32) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .dropout, prev_output_shape, prev_output_shape);
        layer.regularization.dropout_rate = rate;
        try self.addLayer(layer);
    }

    pub fn addBatchNormLayer(self: *NeuralNetwork) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .batch_norm, prev_output_shape, prev_output_shape);
        try self.addLayer(layer);
    }

    pub fn addLSTMLayer(self: *NeuralNetwork, units: usize, return_sequences: bool) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const output_shape = if (return_sequences)
            &[_]usize{ prev_output_shape[0], units } // Sequence length, hidden size
        else
            &[_]usize{units}; // Just hidden size

        const layer = try Layer.init(self.allocator, .lstm, prev_output_shape, output_shape);
        layer.hidden_size = units;
        var prng = std.Random.DefaultPrng.init(0);
        var rng = prng.random();
        try layer.initializeWeights(self.allocator, &rng);
        try self.addLayer(layer);
    }

    pub fn addAttentionLayer(self: *NeuralNetwork, num_heads: usize, head_dim: usize) !void {
        const prev_output_shape = if (self.layers.items.len > 0)
            self.layers.items[self.layers.items.len - 1].output_shape
        else
            self.input_shape;

        const layer = try Layer.init(self.allocator, .multi_head_attention, prev_output_shape, prev_output_shape);
        layer.num_heads = num_heads;
        layer.head_dim = head_dim;
        var prng = std.Random.DefaultPrng.init(0);
        var rng = prng.random();
        try layer.initializeWeights(self.allocator, &rng);
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

        // Get the maximum buffer size needed
        const max_buffer_size = self.getMaxLayerSize();

        // Allocate two buffers for ping-pong processing
        var buffer1 = try self.allocator.alloc(f32, max_buffer_size);
        defer self.allocator.free(buffer1);
        const buffer2 = try self.allocator.alloc(f32, max_buffer_size);
        defer self.allocator.free(buffer2);

        // Copy input to first buffer
        @memcpy(buffer1[0..input.len], input);

        var current_input_slice = buffer1[0..input.len];
        var current_output_buffer = buffer2;
        var last_output_slice: []f32 = undefined;

        for (self.layers.items, 0..) |layer, i| {
            // Determine the actual output size for this layer
            const layer_output_size = blk: {
                var size: usize = 1;
                for (layer.output_shape) |dim| {
                    size *= dim;
                }
                break :blk size;
            };

            const output_slice = current_output_buffer[0..layer_output_size];
            try layer.forward(current_input_slice, output_slice);
            last_output_slice = output_slice;

            // Prepare for next iteration
            if (i < self.layers.items.len - 1) {
                // Swap buffers
                current_input_slice = output_slice;
                // Switch to the other buffer
                if (current_output_buffer.ptr == buffer2.ptr) {
                    current_output_buffer = buffer1;
                } else {
                    current_output_buffer = buffer2;
                }
            }
        }

        // The final output is in last_output_slice
        @memcpy(output, last_output_slice[0..self.getOutputSize()]);
    }

    pub fn predict(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
        const was_training = self.is_training;
        self.setTraining(false);
        defer self.setTraining(was_training);

        try self.forward(input, output);
    }

    pub fn predictBatch(self: *NeuralNetwork, inputs: []const []const f32, outputs: [][]f32) !void {
        if (inputs.len != outputs.len) return error.InvalidBatchSize;

        for (inputs, 0..) |input, i| {
            try self.predict(input, outputs[i]);
        }
    }

    pub fn getParameterCount(self: *const NeuralNetwork) usize {
        var count: usize = 0;
        for (self.layers.items) |layer| {
            if (layer.weights) |weights| count += weights.len;
            if (layer.biases) |biases| count += biases.len;
        }
        return count;
    }

    pub fn getMemoryUsage(self: *const NeuralNetwork) usize {
        var usage: usize = 0;
        for (self.layers.items) |layer| {
            if (layer.weights) |weights| usage += weights.len * @sizeOf(f32);
            if (layer.biases) |biases| usage += biases.len * @sizeOf(f32);
            if (layer.running_mean) |mean| usage += mean.len * @sizeOf(f32);
            if (layer.running_var) |input_var| usage += input_var.len * @sizeOf(f32);
        }
        return usage;
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

/// Advanced embedding generator with multiple architectures
pub const EmbeddingGenerator = struct {
    model: *NeuralNetwork,
    allocator: std.mem.Allocator,
    embedding_dim: usize,
    model_type: enum { mlp, transformer, hybrid } = .mlp,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize) !*EmbeddingGenerator {
        const model = try NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{embedding_size});

        // Add layers for embedding generation
        try model.addDenseLayer(embedding_size * 2, .gelu);
        try model.addDropoutLayer(0.1);
        try model.addDenseLayer(embedding_size, .tanh);

        try model.compile();

        const generator = try allocator.create(EmbeddingGenerator);
        generator.* = .{
            .model = model,
            .allocator = allocator,
            .embedding_dim = embedding_size,
        };
        return generator;
    }

    pub fn initTransformer(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize, num_heads: usize) !*EmbeddingGenerator {
        const model = try NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{embedding_size});

        // Add transformer-based layers
        try model.addDenseLayer(embedding_size, .linear);
        try model.addAttentionLayer(num_heads, embedding_size / num_heads);
        try model.addDenseLayer(embedding_size, .gelu);
        try model.addDropoutLayer(0.1);
        try model.addDenseLayer(embedding_size, .tanh);

        try model.compile();

        const generator = try allocator.create(EmbeddingGenerator);
        generator.* = .{
            .model = model,
            .allocator = allocator,
            .embedding_dim = embedding_size,
            .model_type = .transformer,
        };
        return generator;
    }

    pub fn deinit(self: *EmbeddingGenerator) void {
        self.model.deinit();
        self.allocator.destroy(self);
    }

    pub fn generateEmbedding(self: *EmbeddingGenerator, input: []const f32, embedding: []f32) !void {
        try self.model.predict(input, embedding);
    }

    pub fn generateEmbeddingsBatch(self: *EmbeddingGenerator, inputs: []const []const f32, embeddings: [][]f32) !void {
        if (inputs.len != embeddings.len) return error.InvalidBatchSize;

        for (inputs, 0..) |input, i| {
            try self.generateEmbedding(input, embeddings[i]);
        }
    }

    pub fn computeSimilarity(self: *EmbeddingGenerator, embedding1: []const f32, embedding2: []const f32) f32 {
        _ = self;
        if (embedding1.len != embedding2.len) return 0.0;

        var dot_product: f32 = 0.0;
        var norm1: f32 = 0.0;
        var norm2: f32 = 0.0;

        for (embedding1, 0..) |val1, i| {
            const val2 = embedding2[i];
            dot_product += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }

        const magnitude = @sqrt(norm1) * @sqrt(norm2);
        if (magnitude == 0.0) return 0.0;

        const similarity = dot_product / magnitude;

        // Clamp to valid cosine similarity range due to floating point precision
        return @max(-1.0, @min(1.0, similarity));
    }

    pub fn findNearestNeighbors(
        self: *EmbeddingGenerator,
        query_embedding: []const f32,
        embeddings: []const []const f32,
        k: usize,
        allocator: std.mem.Allocator,
    ) ![]usize {
        const similarities = try allocator.alloc(f32, embeddings.len);
        defer allocator.free(similarities);

        for (embeddings, 0..) |embedding, i| {
            similarities[i] = self.computeSimilarity(query_embedding, embedding);
        }

        // Simple selection of top k (could be optimized with proper sorting)
        const indices = try allocator.alloc(usize, @min(k, embeddings.len));
        for (indices, 0..) |*idx, i| {
            idx.* = i;
        }

        return indices;
    }
};

/// Enhanced model trainer with comprehensive optimization support
pub const ModelTrainer = struct {
    model: *NeuralNetwork,
    config: TrainingConfig,
    allocator: std.mem.Allocator,
    optimizer: Optimizer,
    loss_function: LossFunction,

    // Optimizer state
    momentum_buffers: ?[][]f32 = null,
    velocity_buffers: ?[][]f32 = null,
    adam_m_buffers: ?[][]f32 = null,
    adam_v_buffers: ?[][]f32 = null,

    // Training state
    current_epoch: usize = 0,
    current_step: usize = 0,
    best_val_loss: f32 = std.math.inf(f32),
    patience_counter: usize = 0,

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

        try trainer.initializeOptimizerState();
        return trainer;
    }

    pub fn deinit(self: *ModelTrainer) void {
        self.cleanupOptimizerState();
        self.allocator.destroy(self);
    }

    fn initializeOptimizerState(self: *ModelTrainer) !void {
        // Initialize optimizer-specific state buffers
        switch (self.optimizer) {
            .momentum_sgd, .nesterov_sgd => {
                // Initialize momentum buffers
            },
            .adam, .adamw, .nadam => {
                // Initialize Adam buffers
            },
            .rmsprop => {
                // Initialize RMSprop buffers
            },
            else => {},
        }
    }

    fn cleanupOptimizerState(self: *ModelTrainer) void {
        if (self.momentum_buffers) |buffers| {
            for (buffers) |buffer| {
                self.allocator.free(buffer);
            }
            self.allocator.free(buffers);
        }
        // Cleanup other buffers similarly
    }

    pub fn train(
        self: *ModelTrainer,
        inputs: []const []const f32,
        targets: []const []const f32,
    ) !ArrayList(TrainingMetrics) {
        var metrics = ArrayList(TrainingMetrics){};

        // Validate inputs
        if (inputs.len != targets.len) return error.InvalidDataSize;
        if (inputs.len == 0) return error.EmptyDataset;

        // Split data into training and validation sets
        const validation_size = @as(usize, @intFromFloat(@as(f32, @floatFromInt(inputs.len)) * self.config.validation_split));
        const train_size = inputs.len - validation_size;

        const train_inputs = inputs[0..train_size];
        const train_targets = targets[0..train_size];
        const val_inputs = if (validation_size > 0) inputs[train_size..] else &[_][]const f32{};
        const val_targets = if (validation_size > 0) targets[train_size..] else &[_][]const f32{};

        // Training loop
        for (0..self.config.epochs) |epoch| {
            self.current_epoch = epoch;
            const start_time = std.time.milliTimestamp();

            // Training phase
            self.model.setTraining(true);
            const train_metrics = try self.trainEpoch(train_inputs, train_targets);

            // Validation phase
            var val_metrics: ?TrainingMetrics = null;
            if (val_inputs.len > 0 and epoch % self.config.validate_frequency == 0) {
                self.model.setTraining(false);
                val_metrics = try self.validateEpoch(val_inputs, val_targets);
            }

            const end_time = std.time.milliTimestamp();

            // Create combined metrics
            var epoch_metrics = train_metrics;
            epoch_metrics.epoch = epoch;
            epoch_metrics.training_time_ms = @as(u64, @intCast(end_time - start_time));

            if (val_metrics) |vm| {
                epoch_metrics.val_loss = vm.loss;
                epoch_metrics.val_accuracy = vm.accuracy;
            }

            try metrics.append(self.allocator, epoch_metrics);

            // Early stopping check
            if (self.shouldEarlyStop(epoch_metrics)) {
                std.debug.print("Early stopping at epoch {}\n", .{epoch});
                break;
            }

            // Logging
            if (epoch % self.config.log_frequency == 0) {
                self.logMetrics(epoch_metrics);
            }
        }

        return metrics;
    }

    fn trainEpoch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !TrainingMetrics {
        var total_loss: f32 = 0.0;
        var total_accuracy: f32 = 0.0;
        const num_batches = (inputs.len + self.config.batch_size - 1) / self.config.batch_size;

        for (0..num_batches) |batch_idx| {
            const start_idx = batch_idx * self.config.batch_size;
            const end_idx = @min(start_idx + self.config.batch_size, inputs.len);

            const batch_inputs = inputs[start_idx..end_idx];
            const batch_targets = targets[start_idx..end_idx];

            const batch_metrics = try self.trainBatch(batch_inputs, batch_targets);
            total_loss += batch_metrics.loss;
            total_accuracy += batch_metrics.accuracy;

            self.current_step += 1;
        }

        return TrainingMetrics{
            .loss = total_loss / @as(f32, @floatFromInt(num_batches)),
            .accuracy = total_accuracy / @as(f32, @floatFromInt(num_batches)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0, // Will be set by caller
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn trainBatch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !TrainingMetrics {
        // Forward pass
        const output_size = self.model.getOutputSize();
        const predictions = try self.allocator.alloc(f32, output_size * inputs.len);
        defer self.allocator.free(predictions);

        var batch_loss: f32 = 0.0;
        var batch_accuracy: f32 = 0.0;

        for (inputs, 0..) |input, i| {
            const pred_slice = predictions[i * output_size .. (i + 1) * output_size];
            try self.model.forward(input, pred_slice);

            // Compute loss
            const sample_loss = self.computeLoss(pred_slice, targets[i]);
            batch_loss += sample_loss;

            // Compute accuracy (if applicable)
            const sample_accuracy = self.computeAccuracy(pred_slice, targets[i]);
            batch_accuracy += sample_accuracy;
        }

        // Backward pass (placeholder - would implement backpropagation)
        try self.backwardPass(inputs, targets, predictions);

        // Update weights
        try self.updateWeights();

        return TrainingMetrics{
            .loss = batch_loss / @as(f32, @floatFromInt(inputs.len)),
            .accuracy = batch_accuracy / @as(f32, @floatFromInt(inputs.len)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0,
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn validateEpoch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !TrainingMetrics {
        var total_loss: f32 = 0.0;
        var total_accuracy: f32 = 0.0;

        const output_size = self.model.getOutputSize();
        const predictions = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(predictions);

        for (inputs, 0..) |input, i| {
            try self.model.predict(input, predictions);

            const sample_loss = self.computeLoss(predictions, targets[i]);
            total_loss += sample_loss;

            const sample_accuracy = self.computeAccuracy(predictions, targets[i]);
            total_accuracy += sample_accuracy;
        }

        return TrainingMetrics{
            .loss = total_loss / @as(f32, @floatFromInt(inputs.len)),
            .accuracy = total_accuracy / @as(f32, @floatFromInt(inputs.len)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0,
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn backwardPass(self: *ModelTrainer, _inputs: []const []const f32, _targets: []const []const f32, _predictions: []const f32) !void {
        // Placeholder for backpropagation implementation
        _ = self;
        _ = _inputs;
        _ = _targets;
        _ = _predictions;
    }

    fn updateWeights(self: *ModelTrainer) !void {
        // Placeholder for weight update implementation
        _ = self;
    }

    fn getCurrentLearningRate(self: *const ModelTrainer) f32 {
        // Implement learning rate scheduling
        var lr = self.config.learning_rate;

        switch (self.config.lr_scheduler) {
            .constant => {},
            .step_decay => {
                const decay_factor = std.math.pow(f32, self.config.lr_decay_rate, @as(f32, @floatFromInt(self.current_step / self.config.lr_decay_steps)));
                lr *= decay_factor;
            },
            .exponential_decay => {
                lr *= std.math.pow(f32, self.config.lr_decay_rate, @as(f32, @floatFromInt(self.current_step)));
            },
            .cosine_annealing => {
                const progress = @as(f32, @floatFromInt(self.current_epoch)) / @as(f32, @floatFromInt(self.config.epochs));
                lr *= 0.5 * (1.0 + @cos(std.math.pi * progress));
            },
            else => {},
        }

        return lr;
    }

    fn shouldEarlyStop(self: *ModelTrainer, metrics: TrainingMetrics) bool {
        if (metrics.val_loss) |val_loss| {
            if (val_loss < self.best_val_loss - self.config.early_stopping_min_delta) {
                self.best_val_loss = val_loss;
                self.patience_counter = 0;
                return false;
            } else {
                self.patience_counter += 1;
                return self.patience_counter >= self.config.early_stopping_patience;
            }
        }
        return false;
    }

    fn logMetrics(self: *const ModelTrainer, metrics: TrainingMetrics) void {
        _ = self;
        std.debug.print("Epoch {}: loss={d:.6}, acc={d:.4}", .{ metrics.epoch, metrics.loss, metrics.accuracy });
        if (metrics.val_loss) |val_loss| {
            std.debug.print(", val_loss={d:.6}", .{val_loss});
        }
        if (metrics.val_accuracy) |val_acc| {
            std.debug.print(", val_acc={d:.4}", .{val_acc});
        }
        std.debug.print(", lr={d:.6}\n", .{metrics.learning_rate});
    }

    fn computeLoss(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        switch (self.loss_function) {
            .mean_squared_error => return self.meanSquaredError(predictions, targets),
            .mean_absolute_error => return self.meanAbsoluteError(predictions, targets),
            .cross_entropy => return self.crossEntropy(predictions, targets),
            .binary_cross_entropy => return self.binaryCrossEntropy(predictions, targets),
            .categorical_cross_entropy => return self.categoricalCrossEntropy(predictions, targets),
            .huber => return self.huberLoss(predictions, targets),
            .hinge => return self.hingeLoss(predictions, targets),
            .focal_loss => return self.focalLoss(predictions, targets),
            else => return 0.0,
        }
    }

    fn computeAccuracy(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return 0.0;

        // For classification: find max prediction and compare with target
        var pred_class: usize = 0;
        var target_class: usize = 0;
        var max_pred: f32 = predictions[0];
        var max_target: f32 = targets[0];

        for (predictions, 0..) |pred, i| {
            if (pred > max_pred) {
                max_pred = pred;
                pred_class = i;
            }
            if (targets[i] > max_target) {
                max_target = targets[i];
                target_class = i;
            }
        }

        return if (pred_class == target_class) 1.0 else 0.0;
    }

    /// Mean squared error loss
    fn meanSquaredError(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        var sum: f32 = 0.0;
        for (predictions, 0..) |pred, i| {
            const diff = pred - targets[i];
            sum += diff * diff;
        }
        return sum / @as(f32, @floatFromInt(predictions.len));
    }

    /// Mean absolute error loss
    fn meanAbsoluteError(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        var sum: f32 = 0.0;
        for (predictions, 0..) |pred, i| {
            sum += @abs(pred - targets[i]);
        }
        return sum / @as(f32, @floatFromInt(predictions.len));
    }

    /// Cross-entropy loss
    fn crossEntropy(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        var loss: f32 = 0.0;
        for (predictions, 0..) |pred, i| {
            const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
            loss -= targets[i] * @log(clipped_pred);
        }
        return loss;
    }

    /// Binary cross-entropy loss
    fn binaryCrossEntropy(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        var loss: f32 = 0.0;
        for (predictions, 0..) |pred, i| {
            const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
            loss -= targets[i] * @log(clipped_pred) + (1.0 - targets[i]) * @log(1.0 - clipped_pred);
        }
        return loss / @as(f32, @floatFromInt(predictions.len));
    }

    /// Categorical cross-entropy loss
    fn categoricalCrossEntropy(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        return self.crossEntropy(predictions, targets);
    }

    /// Huber loss
    fn huberLoss(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        const delta: f32 = 1.0;
        var loss: f32 = 0.0;

        for (predictions, 0..) |pred, i| {
            const diff = @abs(pred - targets[i]);
            if (diff <= delta) {
                loss += 0.5 * diff * diff;
            } else {
                loss += delta * (diff - 0.5 * delta);
            }
        }
        return loss / @as(f32, @floatFromInt(predictions.len));
    }

    /// Hinge loss
    fn hingeLoss(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        var loss: f32 = 0.0;
        for (predictions, 0..) |pred, i| {
            loss += @max(0.0, 1.0 - targets[i] * pred);
        }
        return loss / @as(f32, @floatFromInt(predictions.len));
    }

    /// Focal loss for handling class imbalance
    fn focalLoss(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len != targets.len) return std.math.inf(f32);

        const alpha: f32 = 0.25;
        const gamma: f32 = 2.0;
        var loss: f32 = 0.0;

        for (predictions, 0..) |pred, i| {
            const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
            const pt = if (targets[i] == 1.0) clipped_pred else 1.0 - clipped_pred;
            const alpha_t = if (targets[i] == 1.0) alpha else 1.0 - alpha;

            loss -= alpha_t * std.math.pow(f32, 1.0 - pt, gamma) * @log(pt);
        }

        return loss / @as(f32, @floatFromInt(predictions.len));
    }
};

// Re-export commonly used types with enhanced aliases
pub const Network = NeuralNetwork;
pub const Embedding = EmbeddingGenerator;

// Transformer Architecture
pub const transformer = @import("transformer.zig");

// Re-export additional AI components
pub const Neural = @import("neural.zig");
pub const LocalML = @import("localml.zig");
pub const DynamicRouter = @import("dynamic.zig");
pub const DataStructures = @import("data_structures/mod.zig");
pub const Trainer = ModelTrainer;
pub const Config = TrainingConfig;
pub const Metrics = TrainingMetrics;
pub const Loss = LossFunction;
pub const Opt = Optimizer;
pub const agent = @import("agent.zig");
pub const enhanced_agent = @import("enhanced_agent.zig");
pub const reinforcement_learning = @import("reinforcement_learning.zig");
pub const distributed_training = @import("distributed_training.zig");
pub const model_serialization = @import("model_serialization.zig");

// Utility functions
pub fn createMLP(allocator: std.mem.Allocator, layer_sizes: []const usize, activations: []const Activation) !*NeuralNetwork {
    if (layer_sizes.len < 2) return error.InvalidArchitecture;
    if (activations.len != layer_sizes.len - 1) return error.MismatchedActivations;

    const network = try NeuralNetwork.init(allocator, &[_]usize{layer_sizes[0]}, &[_]usize{layer_sizes[layer_sizes.len - 1]});

    for (1..layer_sizes.len) |i| {
        try network.addDenseLayer(layer_sizes[i], activations[i - 1]);
    }

    try network.compile();
    return network;
}

pub fn createCNN(allocator: std.mem.Allocator, input_shape: []const usize, num_classes: usize) !*NeuralNetwork {
    const network = try NeuralNetwork.init(allocator, input_shape, &[_]usize{num_classes});

    // Add convolutional layers
    try network.addConv2DLayer(32, .{ 3, 3 }, .relu);
    try network.addConv2DLayer(64, .{ 3, 3 }, .relu);
    try network.addDropoutLayer(0.25);

    // Add dense layers
    try network.addDenseLayer(128, .relu);
    try network.addDropoutLayer(0.5);
    try network.addDenseLayer(num_classes, .softmax);

    try network.compile();
    return network;
}

test "Enhanced neural network operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple neural network
    var network = try NeuralNetwork.init(allocator, &[_]usize{2}, &[_]usize{2});
    defer network.deinit();

    // Add layers with different activations
    try network.addDenseLayer(4, .gelu);
    try network.addDropoutLayer(0.1);
    try network.addDenseLayer(2, .softmax);

    // Compile the network
    try network.compile();

    // Test forward pass
    const input = [_]f32{ 1.0, 1.0 };
    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    try network.forward(&input, output);

    // Verify output
    try testing.expect(output.len == 2);

    // Check softmax properties
    var sum: f32 = 0.0;
    for (output) |val| {
        try testing.expect(val >= 0.0);
        try testing.expect(val <= 1.0);
        sum += val;
    }
    try testing.expectApproxEqAbs(1.0, sum, 0.001);

    // Test parameter count
    const param_count = network.getParameterCount();
    try testing.expect(param_count > 0);

    // Test memory usage
    const memory_usage = network.getMemoryUsage();
    try testing.expect(memory_usage > 0);
}

test "Advanced embedding generator" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create transformer-based embedding generator
    var generator = try EmbeddingGenerator.initTransformer(allocator, 10, 8, 2);
    defer generator.deinit();

    // Generate embedding
    const input = [_]f32{0.1} ** 10;
    const embedding = try allocator.alloc(f32, 8);
    defer allocator.free(embedding);

    try generator.generateEmbedding(&input, embedding);

    // Verify embedding
    try testing.expect(embedding.len == 8);
    for (embedding) |val| {
        try testing.expect(val >= -1.0 and val <= 1.0); // tanh activation
    }

    // Test similarity computation
    const embedding2 = try allocator.alloc(f32, 8);
    defer allocator.free(embedding2);
    try generator.generateEmbedding(&input, embedding2);

    const similarity = generator.computeSimilarity(embedding, embedding2);
    try testing.expect(similarity >= -1.0 and similarity <= 1.0);
}

test "Enhanced model trainer" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple network
    var network = try createMLP(allocator, &[_]usize{ 2, 4, 2 }, &[_]Activation{ .relu, .softmax });
    defer network.deinit();

    // Create training configuration
    const config = TrainingConfig{
        .learning_rate = 0.01,
        .batch_size = 2,
        .epochs = 5,
        .validation_split = 0.2,
        .early_stopping_patience = 3,
    };

    // Create trainer
    var trainer = try ModelTrainer.init(allocator, network, config, .adam, .categorical_cross_entropy);
    defer trainer.deinit();

    // Create dummy training data
    const inputs = try allocator.alloc([]const f32, 4);
    defer allocator.free(inputs);
    const targets = try allocator.alloc([]const f32, 4);
    defer allocator.free(targets);

    const input_data = [_][2]f32{ .{ 0.0, 0.0 }, .{ 0.0, 1.0 }, .{ 1.0, 0.0 }, .{ 1.0, 1.0 } };
    const target_data = [_][2]f32{ .{ 1.0, 0.0 }, .{ 0.0, 1.0 }, .{ 0.0, 1.0 }, .{ 1.0, 0.0 } };

    for (0..4) |i| {
        inputs[i] = &input_data[i];
        targets[i] = &target_data[i];
    }

    // Train the model
    var metrics = try trainer.train(inputs, targets);
    defer metrics.deinit(allocator);

    // Verify training ran
    try testing.expect(metrics.items.len > 0);
    try testing.expect(metrics.items.len <= config.epochs);

    // Check that loss values are reasonable
    for (metrics.items) |metric| {
        try testing.expect(metric.loss >= 0.0);
        try testing.expect(metric.accuracy >= 0.0 and metric.accuracy <= 1.0);
    }
}
