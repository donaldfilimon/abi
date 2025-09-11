//! Neural Network Activation Functions
//!
//! Comprehensive collection of activation functions with SIMD optimization.
//! Provides both element-wise and batch activation functions.

const std = @import("std");

const core = @import("../core/mod.zig");

/// Available activation function types
pub const ActivationType = enum {
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
};

/// Activation function configuration
pub const ActivationConfig = struct {
    activation_type: ActivationType,
    alpha: f32 = 0.01, // For leaky_relu, elu, etc.
    beta: f32 = 1.0, // For swish, etc.
    threshold: f32 = 1.0,
    enable_simd: bool = true,
};

/// High-performance activation function processor
pub const ActivationProcessor = struct {
    config: ActivationConfig,

    pub fn init(config: ActivationConfig) ActivationProcessor {
        return .{ .config = config };
    }

    /// Apply activation function to a single value
    pub fn activate(self: *const ActivationProcessor, x: f32) f32 {
        return switch (self.config.activation_type) {
            .relu => @max(0.0, x),
            .relu6 => @max(0.0, @min(6.0, x)),
            .leaky_relu => if (x > 0.0) x else self.config.alpha * x,
            .parametric_relu => if (x > 0.0) x else self.config.alpha * x,
            .elu => if (x > 0.0) x else self.config.alpha * (@exp(x) - 1.0),
            .selu => blk: {
                const alpha: f32 = 1.6732632423543772848170429916717;
                const scale: f32 = 1.0507009873554804934193349852946;
                if (x > 0.0) {
                    break :blk scale * x;
                } else {
                    break :blk scale * alpha * (@exp(x) - 1.0);
                }
            },
            .celu => if (x > 0.0) x else self.config.alpha * (@exp(x / self.config.alpha) - 1.0),
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .hard_sigmoid => @max(0.0, @min(1.0, 0.2 * x + 0.5)),
            .tanh => std.math.tanh(x),
            .hard_tanh => @max(-1.0, @min(1.0, x)),
            .softplus => @log(1.0 + @exp(x)),
            .softsign => x / (1.0 + @abs(x)),
            .swish => x / (1.0 + @exp(-self.config.beta * x)),
            .hard_swish => blk: {
                if (x <= -3.0) break :blk 0.0;
                if (x >= 3.0) break :blk x;
                break :blk x * (x + 3.0) / 6.0;
            },
            .mish => x * std.math.tanh(@log(1.0 + @exp(x))),
            .gelu => 0.5 * x * (1.0 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (x + 0.044715 * std.math.pow(f32, x, 3)))),
            .quick_gelu => x * self.sigmoid(1.702 * x),
            .linear => x,
            .step => if (x > 0.0) 1.0 else 0.0,
            .threshold => if (x > self.config.threshold) x else 0.0,
            .softmax, .log_softmax, .softmin => @panic("Use activateBatch for softmax-like functions"),
        };
    }

    /// Apply activation function to an array (with SIMD optimization)
    pub fn activateBatch(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        std.debug.assert(output.len == input.len);

        // Special handling for softmax-like functions that require global normalization
        switch (self.config.activation_type) {
            .softmax => self.softmax(output, input),
            .log_softmax => self.logSoftmax(output, input),
            .softmin => self.softmin(output, input),
            else => {
                // Element-wise functions with SIMD optimization
                if (self.config.enable_simd and self.canUseSIMD()) {
                    self.activateBatchSIMD(output, input);
                } else {
                    self.activateBatchScalar(output, input);
                }
            },
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(self: *const ActivationProcessor, x: f32, y: f32) f32 {
        return switch (self.config.activation_type) {
            .relu => if (x > 0.0) 1.0 else 0.0,
            .relu6 => if (x > 0.0 and x < 6.0) 1.0 else 0.0,
            .leaky_relu => if (x > 0.0) 1.0 else self.config.alpha,
            .sigmoid => y * (1.0 - y),
            .tanh => 1.0 - y * y,
            .linear => 1.0,
            .swish => {
                const sigmoid_val = 1.0 / (1.0 + @exp(-self.config.beta * x));
                return sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val) * self.config.beta;
            },
            .gelu => blk: {
                const sqrt_2_pi = @sqrt(2.0 / std.math.pi);
                const pdf_factor = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
                const tanh_input = sqrt_2_pi * (x + 0.044715 * x * x * x);
                const sech2 = 1.0 - std.math.tanh(tanh_input) * std.math.tanh(tanh_input);
                break :blk 0.5 * (1.0 + std.math.tanh(tanh_input)) + 0.5 * x * sech2 * pdf_factor;
            },
            else => @panic("Derivative not implemented for this activation function"),
        };
    }

    /// Batch derivative computation
    pub fn derivativeBatch(self: *const ActivationProcessor, output: []f32, input: []const f32, forward_output: []const f32) void {
        std.debug.assert(output.len == input.len and input.len == forward_output.len);

        for (output, input, forward_output) |*out, in, fwd| {
            out.* = self.derivative(in, fwd);
        }
    }

    // Private implementation methods

    fn canUseSIMD(self: *const ActivationProcessor) bool {
        _ = self;
        return core.Features.has_simd;
    }

    fn activateBatchScalar(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        for (output, input) |*out, in| {
            out.* = self.activate(in);
        }
    }

    fn activateBatchSIMD(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        // SIMD optimization for common activation functions
        switch (self.config.activation_type) {
            .relu => self.reluSIMD(output, input),
            .sigmoid => self.sigmoidSIMD(output, input),
            .tanh => self.tanhSIMD(output, input),
            .linear => @memcpy(output, input),
            else => self.activateBatchScalar(output, input), // Fallback
        }
    }

    fn reluSIMD(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        if (!@hasDecl(std.simd, "f32x4")) {
            return self.activateBatchScalar(output, input);
        }

        const zero_vec = std.simd.f32x4{ 0, 0, 0, 0 };
        var i: usize = 0;
        const vec_len = input.len & ~@as(usize, 3);

        while (i < vec_len) : (i += 4) {
            const in_vec: std.simd.f32x4 = input[i .. i + 4][0..4].*;
            const out_vec = @max(zero_vec, in_vec);
            output[i .. i + 4][0..4].* = out_vec;
        }

        // Handle remaining elements
        while (i < input.len) : (i += 1) {
            output[i] = @max(0.0, input[i]);
        }
    }

    fn sigmoidSIMD(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        // For now, fallback to scalar implementation
        // TODO: Implement fast SIMD sigmoid approximation
        self.activateBatchScalar(output, input);
    }

    fn tanhSIMD(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        // For now, fallback to scalar implementation
        // TODO: Implement fast SIMD tanh approximation
        self.activateBatchScalar(output, input);
    }

    /// Scalar sigmoid helper used by certain activations (e.g., quick_gelu)
    fn sigmoid(self: *const ActivationProcessor, x: f32) f32 {
        _ = self;
        return 1.0 / (1.0 + @exp(-x));
    }

    fn softmax(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        _ = self;
        if (input.len == 0) return;

        // Find maximum for numerical stability
        var max_val = input[0];
        for (input[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exponentials and sum
        var sum: f32 = 0.0;
        for (output, input) |*out, in| {
            out.* = @exp(in - max_val);
            sum += out.*;
        }

        // Normalize
        if (sum > 0.0) {
            for (output) |*out| {
                out.* /= sum;
            }
        }
    }

    fn logSoftmax(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        _ = self;
        if (input.len == 0) return;

        // Find maximum for numerical stability
        var max_val = input[0];
        for (input[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute log-sum-exp
        var sum: f32 = 0.0;
        for (input) |in| {
            sum += @exp(in - max_val);
        }
        const log_sum = @log(sum) + max_val;

        // Compute log-softmax
        for (output, input) |*out, in| {
            out.* = in - log_sum;
        }
    }

    fn softmin(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        // Softmin is equivalent to softmax over negated inputs, implemented without extra allocation
        _ = self;
        if (input.len == 0) return;

        // For numerical stability, compute the max of the negated inputs (i.e., -min(input))
        var max_neg = -input[0];
        for (input[1..]) |val| {
            const neg = -val;
            if (neg > max_neg) max_neg = neg;
        }

        // Compute exponentials of negated, shifted inputs and accumulate sum
        var sum: f32 = 0.0;
        for (output, input) |*out, in| {
            const e = @exp(-in - max_neg);
            out.* = e;
            sum += e;
        }

        // Normalize
        if (sum > 0.0) {
            const inv_sum = 1.0 / sum;
            for (output) |*out| {
                out.* *= inv_sum;
            }
        }
    }
};

/// Activation function registry for dynamic dispatch
pub const ActivationRegistry = struct {
    const ActivationFn = *const fn ([]f32, []const f32) void;

    functions: std.StringHashMap(ActivationFn),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ActivationRegistry {
        return .{
            .functions = std.StringHashMap(ActivationFn).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ActivationRegistry) void {
        self.functions.deinit();
    }

    pub fn register(self: *ActivationRegistry, name: []const u8, func: ActivationFn) !void {
        try self.functions.put(name, func);
    }

    pub fn get(self: *ActivationRegistry, name: []const u8) ?ActivationFn {
        return self.functions.get(name);
    }
};

test "activation functions" {
    const testing = std.testing;

    const processor = ActivationProcessor.init(.{ .activation_type = .relu });

    // Test single value activation
    try testing.expectEqual(@as(f32, 0.0), processor.activate(-1.0));
    try testing.expectEqual(@as(f32, 1.0), processor.activate(1.0));

    // Test batch activation
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var output = [_]f32{0} ** 4;
    processor.activateBatch(&output, &input);

    try testing.expectEqual(@as(f32, 0.0), output[0]);
    try testing.expectEqual(@as(f32, 0.0), output[1]);
    try testing.expectEqual(@as(f32, 1.0), output[2]);
    try testing.expectEqual(@as(f32, 2.0), output[3]);
}

test "softmax activation" {
    const testing = std.testing;

    const processor = ActivationProcessor.init(.{ .activation_type = .softmax });

    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{0} ** 3;
    processor.activateBatch(&output, &input);

    // Check that outputs sum to 1
    var sum: f32 = 0.0;
    for (output) |val| {
        sum += val;
        try testing.expect(val >= 0.0);
        try testing.expect(val <= 1.0);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}
