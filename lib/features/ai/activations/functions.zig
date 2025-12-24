//! Neural Network Activation Functions
//!
//! Comprehensive collection of activation functions with SIMD optimization.
//! Provides both element-wise and batch activation functions with detailed
//! mathematical definitions, derivatives, and properties for each function.

const std = @import("std");

// core functionality is now imported through module dependencies

/// Available activation function types with detailed mathematical definitions
pub const ActivationType = enum {
    /// **Rectified Linear Unit (ReLU)**: f(x) = max(0, x)
    /// - **Derivative**: f'(x) = 1 if x > 0, else 0
    /// - **Properties**: Introduces non-linearity; computationally efficient; may cause "dying ReLU" problem where neurons become inactive.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Rectifier)
    relu,

    /// **ReLU6**: f(x) = min(max(0, x), 6)
    /// - **Derivative**: f'(x) = 1 if 0 < x < 6, else 0
    /// - **Properties**: Similar to ReLU but caps the output at 6; used in mobile networks for quantization benefits.
    /// - **Reference**: [TensorFlow - ReLU6](https://www.tensorflow.org/api_docs/python/tf/nn/relu6)
    relu6,

    /// **Leaky ReLU**: f(x) = x if x > 0, else α * x
    /// - **Derivative**: f'(x) = 1 if x > 0, else α
    /// - **Properties**: Allows a small, non-zero gradient when x < 0 to mitigate "dying ReLU" problem; α is a small constant (e.g., 0.01).
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Leaky_ReLU)
    leaky_relu,

    /// **Parametric ReLU (PReLU)**: f(x) = x if x > 0, else α * x
    /// - **Derivative**: f'(x) = 1 if x > 0, else α
    /// - **Properties**: Similar to Leaky ReLU but α is a learnable parameter, allowing the network to adapt the activation function.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Parametric_ReLU)
    parametric_relu,

    /// **Exponential Linear Unit (ELU)**: f(x) = x if x > 0, else α * (exp(x) - 1)
    /// - **Derivative**: f'(x) = 1 if x > 0, else f(x) + α
    /// - **Properties**: Smooth and differentiable; helps mitigate vanishing gradients; α is a positive constant.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#ELU)
    elu,

    /// **Scaled Exponential Linear Unit (SELU)**: f(x) = λ * (x if x > 0, else α * (exp(x) - 1))
    /// - **Derivative**: f'(x) = λ if x > 0, else λ * α * exp(x)
    /// - **Properties**: Self-normalizing; maintains mean and variance; λ and α are predefined constants.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#SELU)
    selu,

    /// **Continuously Differentiable Exponential Linear Unit (CELU)**: f(x) = x if x > 0, else α * (exp(x / α) - 1)
    /// - **Derivative**: f'(x) = 1 if x > 0, else exp(x / α)
    /// - **Properties**: Similar to ELU but ensures continuous differentiability; α is a positive constant.
    /// - **Reference**: [CELU Paper](https://arxiv.org/abs/1704.07483)
    celu,

    /// **Sigmoid**: f(x) = 1 / (1 + exp(-x))
    /// - **Derivative**: f'(x) = f(x) * (1 - f(x))
    /// - **Properties**: Outputs values between 0 and 1; prone to vanishing gradients; historically used in early neural networks.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Logistic)
    sigmoid,

    /// **Hard Sigmoid**: f(x) = max(0, min(1, 0.2 * x + 0.5))
    /// - **Derivative**: f'(x) = 0.2 if -2.5 < x < 2.5, else 0
    /// - **Properties**: Computationally efficient approximation of sigmoid; used in resource-constrained environments.
    /// - **Reference**: [Keras - Hard Sigmoid](https://keras.io/api/layers/activations/#hard_sigmoid-function)
    hard_sigmoid,

    /// **Hyperbolic Tangent (Tanh)**: f(x) = tanh(x)
    /// - **Derivative**: f'(x) = 1 - f(x)^2
    /// - **Properties**: Outputs values between -1 and 1; zero-centered; can suffer from vanishing gradients.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Hyperbolic_tangent)
    tanh,

    /// **Hard Tanh**: f(x) = max(-1, min(1, x))
    /// - **Derivative**: f'(x) = 1 if -1 < x < 1, else 0
    /// - **Properties**: Computationally efficient approximation of tanh; used in resource-constrained environments.
    /// - **Reference**: [PyTorch - Hard Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
    hard_tanh,

    /// **Softmax**: f(x_i) = exp(x_i) / Σ(exp(x_j) for j in inputs)
    /// - **Derivative**: Complex; involves Jacobian matrix where ∂f_i/∂x_j = f_i * (δ_ij - f_j)
    /// - **Properties**: Converts logits to probabilities; used in multi-class classification; outputs sum to 1.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Softmax)
    softmax,

    /// **Log Softmax**: f(x_i) = log(exp(x_i) / Σ(exp(x_j) for j in inputs))
    /// - **Derivative**: ∂f_i/∂x_j = δ_ij - exp(x_j - logsumexp(x))
    /// - **Properties**: Numerically stable version of softmax; used in combination with negative log-likelihood loss.
    /// - **Reference**: [PyTorch - Log Softmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)
    log_softmax,

    /// **Softmin**: f(x_i) = exp(-x_i) / Σ(exp(-x_j) for j in inputs)
    /// - **Derivative**: Complex; involves Jacobian matrix similar to softmax but with negated inputs
    /// - **Properties**: Emphasizes smaller input values; less common in practice.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Softmin)
    softmin,

    /// **Softplus**: f(x) = log(1 + exp(x))
    /// - **Derivative**: f'(x) = 1 / (1 + exp(-x)) = sigmoid(x)
    /// - **Properties**: Smooth approximation of ReLU; always positive; used in certain probabilistic models.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Softplus)
    softplus,

    /// **Softsign**: f(x) = x / (1 + |x|)
    /// - **Derivative**: f'(x) = 1 / (1 + |x|)^2
    /// - **Properties**: Similar to tanh but computationally simpler; outputs between -1 and 1.
    /// - **Reference**: [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function#Softsign)
    softsign,

    /// **Swish**: f(x) = x * sigmoid(β * x)
    /// - **Derivative**: f'(x) = β * sigmoid(β * x) + x * β * sigmoid(β * x) * (1 - sigmoid(β * x))
    /// - **Properties**: Smooth, non-monotonic; outperforms ReLU in some deep networks; β is a constant (often 1).
    /// - **Reference**: [Swish Paper](https://arxiv.org/abs/1710.05941)
    swish,

    /// **Hard Swish**: f(x) = x * max(0, min(1, (x + 3) / 6))
    /// - **Derivative**: f'(x) = (x / 6) + 0.5 if -3 < x < 3, else 1 if x > 3, else 0
    /// - **Properties**: Computationally efficient approximation of Swish; used in mobile networks.
    /// - **Reference**: [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
    hard_swish,

    /// **Mish**: f(x) = x * tanh(softplus(x))
    /// - **Derivative**: f'(x) = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh^2(softplus(x)))
    /// - **Properties**: Smooth, non-monotonic; outperforms ReLU in some tasks.
    /// - **Reference**: [Mish Paper](https://arxiv.org/abs/1908.08681)
    mish,

    /// **Gaussian Error Linear Unit (GELU)**: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    /// - **Derivative**: Complex; f'(x) = 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3))) + 0.5 * x * sech^2(...) * √(2/π) * (1 + 3 * 0.044715 * x^2)
    /// - **Properties**: Smooth; used in Transformers and BERT; probabilistically motivated activation.
    /// - **Reference**: [GELU Paper](https://arxiv.org/abs/1606.08415)
    gelu,

    /// **Quick GELU**: f(x) = x * sigmoid(1.702 * x)
    /// - **Derivative**: f'(x) = sigmoid(1.702 * x) + x * 1.702 * sigmoid(1.702 * x) * (1 - sigmoid(1.702 * x))
    /// - **Properties**: Faster approximation of GELU; maintains similar performance with reduced computational cost.
    /// - **Reference**: [Quick GELU approximation in various implementations]
    quick_gelu,

    /// **Linear**: f(x) = x
    /// - **Derivative**: f'(x) = 1
    /// - **Properties**: Identity function; no non-linearity; used for output layers in regression tasks.
    /// - **Reference**: [Wikipedia - Identity function](https://en.wikipedia.org/wiki/Identity_function)
    linear,

    /// **Step**: f(x) = 1 if x > 0, else 0
    /// - **Derivative**: f'(x) = 0 (undefined at x = 0)
    /// - **Properties**: Binary activation; historically used in perceptrons; not differentiable.
    /// - **Reference**: [Wikipedia - Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function)
    step,

    /// **Threshold**: f(x) = x if x > threshold, else 0
    /// - **Derivative**: f'(x) = 1 if x > threshold, else 0
    /// - **Properties**: Generalized step function with learnable threshold; creates sparse activations.
    /// - **Reference**: [Custom threshold activation for sparse neural networks]
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
            .parametric_relu => if (x > 0.0) 1.0 else self.config.alpha,
            .elu => if (x > 0.0) 1.0 else y + self.config.alpha,
            .selu => blk: {
                const alpha: f32 = 1.6732632423543772848170429916717;
                const scale: f32 = 1.0507009873554804934193349852946;
                if (x > 0.0) {
                    break :blk scale;
                } else {
                    break :blk scale * alpha * @exp(x);
                }
            },
            .celu => if (x > 0.0) 1.0 else @exp(x / self.config.alpha),
            .sigmoid => y * (1.0 - y),
            .hard_sigmoid => if (x > -2.5 and x < 2.5) 0.2 else 0.0,
            .tanh => 1.0 - y * y,
            .hard_tanh => if (x > -1.0 and x < 1.0) 1.0 else 0.0,
            .softplus => 1.0 / (1.0 + @exp(-x)),
            .softsign => {
                const abs_x = @abs(x);
                return 1.0 / ((1.0 + abs_x) * (1.0 + abs_x));
            },
            .linear => 1.0,
            .step => 0.0, // Derivative is 0 everywhere except at x=0 where it's undefined
            .threshold => if (x > self.config.threshold) 1.0 else 0.0,
            .swish => {
                const sigmoid_val = 1.0 / (1.0 + @exp(-self.config.beta * x));
                return sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val) * self.config.beta;
            },
            .hard_swish => blk: {
                if (x <= -3.0) break :blk 0.0;
                if (x >= 3.0) break :blk 1.0;
                break :blk (x / 6.0) + 0.5;
            },
            .mish => blk: {
                const softplus_x = @log(1.0 + @exp(x));
                const tanh_softplus = std.math.tanh(softplus_x);
                const sigmoid_x = 1.0 / (1.0 + @exp(-x));
                const sech2_softplus = 1.0 - tanh_softplus * tanh_softplus;
                break :blk tanh_softplus + x * sigmoid_x * sech2_softplus;
            },
            .gelu => blk: {
                const sqrt_2_pi = @sqrt(2.0 / std.math.pi);
                const pdf_factor = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
                const tanh_input = sqrt_2_pi * (x + 0.044715 * x * x * x);
                const sech2 = 1.0 - std.math.tanh(tanh_input) * std.math.tanh(tanh_input);
                break :blk 0.5 * (1.0 + std.math.tanh(tanh_input)) + 0.5 * x * sech2 * pdf_factor;
            },
            .quick_gelu => blk: {
                const sigmoid_val = 1.0 / (1.0 + @exp(-1.702 * x));
                break :blk sigmoid_val + x * 1.702 * sigmoid_val * (1.0 - sigmoid_val);
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
        return @hasDecl(std.simd, "f32x4");
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
        if (!@hasDecl(std.simd, "f32x4")) {
            return self.activateBatchScalar(output, input);
        }

        var index: usize = 0;
        if (@hasDecl(std.simd, "f32x16")) {
            index = self.sigmoidSimdApprox(output, input, index, std.simd.f32x16);
        }
        if (@hasDecl(std.simd, "f32x8")) {
            index = self.sigmoidSimdApprox(output, input, index, std.simd.f32x8);
        }
        if (@hasDecl(std.simd, "f32x4")) {
            index = self.sigmoidSimdApprox(output, input, index, std.simd.f32x4);
        }

        while (index < input.len) : (index += 1) {
            output[index] = self.sigmoid(input[index]);
        }
    }

    fn tanhSIMD(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
        if (!@hasDecl(std.simd, "f32x4")) {
            return self.activateBatchScalar(output, input);
        }

        var index: usize = 0;
        if (@hasDecl(std.simd, "f32x16")) {
            index = self.tanhSimdApprox(output, input, index, std.simd.f32x16);
        }
        if (@hasDecl(std.simd, "f32x8")) {
            index = self.tanhSimdApprox(output, input, index, std.simd.f32x8);
        }
        if (@hasDecl(std.simd, "f32x4")) {
            index = self.tanhSimdApprox(output, input, index, std.simd.f32x4);
        }

        while (index < input.len) : (index += 1) {
            output[index] = std.math.tanh(input[index]);
        }
    }

    fn sigmoidSimdApprox(self: *const ActivationProcessor, output: []f32, input: []const f32, start: usize, comptime V: type) usize {
        _ = self;
        const lanes = @typeInfo(V).Vector.len;
        var i = start;
        while (i + lanes <= input.len) : (i += lanes) {
            const slice = input[i .. i + lanes][0..lanes];
            const in_vec: V = @as(V, slice.*);
            const abs_vec: V = @abs(in_vec);
            const ones: V = @splat(@as(f32, 1.0));
            const half: V = @splat(@as(f32, 0.5));
            const approx_vec: V = half * ((in_vec / (ones + abs_vec)) + ones);
            var approx_arr = @as([lanes]f32, approx_vec);
            for (approx_arr[0..]) |*val| {
                val.* = std.math.clamp(val.*, 0.0, 1.0);
            }
            @memcpy(output[i .. i + lanes], approx_arr[0..]);
        }
        return i;
    }

    fn tanhSimdApprox(self: *const ActivationProcessor, output: []f32, input: []const f32, start: usize, comptime V: type) usize {
        _ = self;
        const lanes = @typeInfo(V).Vector.len;
        const c27: V = @splat(@as(f32, 27.0));
        const c9: V = @splat(@as(f32, 9.0));
        var i = start;
        while (i + lanes <= input.len) : (i += lanes) {
            const slice = input[i .. i + lanes][0..lanes];
            const x: V = @as(V, slice.*);
            const x2: V = x * x;
            const numerator: V = x * (c27 + x2);
            const denominator: V = c27 + (c9 * x2);
            var approx_arr = @as([lanes]f32, numerator / denominator);
            for (approx_arr[0..]) |*val| {
                val.* = std.math.clamp(val.*, -1.0, 1.0);
            }
            @memcpy(output[i .. i + lanes], approx_arr[0..]);
        }
        return i;
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

test "Activation sigmoid SIMD tracks scalar" {
    const input = [_]f32{ -6.0, -3.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, -2.5, 4.0, -4.0, 0.25 };
    var simd_output: [input.len]f32 = undefined;
    var scalar_output: [input.len]f32 = undefined;

    var simd_proc = ActivationProcessor.init(.{ .activation_type = .sigmoid, .enable_simd = true });
    var scalar_proc = ActivationProcessor.init(.{ .activation_type = .sigmoid, .enable_simd = false });

    simd_proc.activateBatch(simd_output[0..], input[0..]);
    scalar_proc.activateBatch(scalar_output[0..], input[0..]);

    const tolerance: f32 = 5.0e-2;
    for (simd_output, scalar_output) |simd_val, scalar_val| {
        try std.testing.expectApproxEqAbs(scalar_val, simd_val, tolerance);
    }
}

test "Activation tanh SIMD tracks scalar" {
    const input = [_]f32{ -5.0, -3.0, -1.5, -0.75, -0.1, 0.0, 0.1, 0.75, 1.5, 3.0, 5.0, -2.0, 2.0, -4.0, 4.0, 0.33 };
    var simd_output: [input.len]f32 = undefined;
    var scalar_output: [input.len]f32 = undefined;

    var simd_proc = ActivationProcessor.init(.{ .activation_type = .tanh, .enable_simd = true });
    var scalar_proc = ActivationProcessor.init(.{ .activation_type = .tanh, .enable_simd = false });

    simd_proc.activateBatch(simd_output[0..], input[0..]);
    scalar_proc.activateBatch(scalar_output[0..], input[0..]);

    const tolerance: f32 = 5.0e-2;
    for (simd_output, scalar_output) |simd_val, scalar_val| {
        try std.testing.expectApproxEqAbs(scalar_val, simd_val, tolerance);
    }
}
