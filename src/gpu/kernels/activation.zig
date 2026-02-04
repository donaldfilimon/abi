//! Neural Network Activation Kernel Definitions
//!
//! Pre-defined kernel IR for activation functions used in neural networks.
//!
//! ## Basic Activations
//! - softmax: Softmax normalization
//! - relu: Rectified Linear Unit
//! - sigmoid: Logistic sigmoid
//! - tanh: Hyperbolic tangent
//!
//! ## Advanced Activations
//! - gelu: Gaussian Error Linear Unit (BERT, GPT)
//! - gelu_fast: Fast GELU approximation
//! - silu: Sigmoid Linear Unit / Swish (EfficientNet, LLaMA)
//! - swiglu: Swish-Gated Linear Unit (LLaMA, Mixtral, PaLM)

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

/// Build softmax kernel: output[i] = exp(input[i] - max) / sum(exp)
pub fn buildSoftmaxKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "softmax");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const max_val = try builder.addUniform("max_val", Type.f32Type());
    const sum_exp = try builder.addUniform("sum_exp", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = exp(input[i] - max_val) / sum_exp
    const input_val = try input.at(idx);
    const shifted = try builder.sub(input_val, try max_val.toExpr());
    // exp(input[i] - max_val)
    const exp_shifted = try builder.exp(shifted);
    // exp(input[i] - max_val) / sum_exp
    const softmax_val = try builder.div(exp_shifted, try sum_exp.toExpr());
    const output_idx = try output.at(idx);

    const assign_stmt = try builder.assignStmt(output_idx, softmax_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build relu kernel: output[i] = max(0, input[i])
pub fn buildReluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "relu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const input_val = try input.at(idx);
    const zero = try builder.f32Lit(0.0);

    // max(0, input[i]) - use select
    const is_positive = try builder.gt(input_val, zero);
    const relu_val = try builder.select(is_positive, input_val, zero);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, relu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build sigmoid kernel: output[i] = 1 / (1 + exp(-input[i]))
pub fn buildSigmoidKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "sigmoid");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // sigmoid(x) = 1 / (1 + exp(-x))
    const x = try input.at(idx);

    // -x
    const neg_x = try builder.neg(x);

    // exp(-x)
    const exp_neg_x = try builder.exp(neg_x);

    // 1 + exp(-x)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg_x);

    // 1 / (1 + exp(-x))
    const sigmoid_val = try builder.div(one, denom);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, sigmoid_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build tanh kernel: output[i] = tanh(input[i])
pub fn buildTanhKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "tanh");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // tanh(x)
    const x = try input.at(idx);
    const tanh_val = try builder.tanh(x);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, tanh_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build GELU kernel: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// GELU (Gaussian Error Linear Unit) is used in BERT, GPT, and modern transformers.
pub fn buildGeluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "gelu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Constants: sqrt(2/pi) ~ 0.7978845608, coefficient = 0.044715
    const x = try input.at(idx);

    // x^3
    const x_sq = try builder.mul(x, x);
    const x_cubed = try builder.mul(x_sq, x);

    // 0.044715 * x^3
    const coef = try builder.f32Lit(0.044715);
    const coef_x3 = try builder.mul(coef, x_cubed);

    // x + 0.044715 * x^3
    const inner_sum = try builder.add(x, coef_x3);

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    const sqrt_2_pi = try builder.f32Lit(0.7978845608);
    const scaled = try builder.mul(sqrt_2_pi, inner_sum);

    // tanh(...)
    const tanh_val = try builder.tanh(scaled);

    // 1 + tanh(...)
    const one = try builder.f32Lit(1.0);
    const one_plus_tanh = try builder.add(one, tanh_val);

    // 0.5 * x
    const half = try builder.f32Lit(0.5);
    const half_x = try builder.mul(half, x);

    // 0.5 * x * (1 + tanh(...))
    const gelu_val = try builder.mul(half_x, one_plus_tanh);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fast GELU approximation kernel: output[i] = x * sigmoid(1.702 * x)
/// Faster than exact GELU, used in some production systems.
pub fn buildGeluFastKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "gelu_fast");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Fast GELU: x * sigmoid(1.702 * x)
    // sigmoid(y) = 1 / (1 + exp(-y))
    const x = try input.at(idx);

    // 1.702 * x
    const coef = try builder.f32Lit(1.702);
    const scaled_x = try builder.mul(coef, x);

    // -1.702 * x
    const neg_scaled = try builder.neg(scaled_x);

    // exp(-1.702 * x)
    const exp_neg = try builder.exp(neg_scaled);

    // 1 + exp(-1.702 * x)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg);

    // sigmoid = 1 / (1 + exp(-1.702 * x))
    const sigmoid = try builder.div(one, denom);

    // x * sigmoid(1.702 * x)
    const gelu_fast = try builder.mul(x, sigmoid);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_fast);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build SiLU (Swish) kernel: output[i] = x * sigmoid(x) = x / (1 + exp(-x))
/// SiLU (Sigmoid Linear Unit) is used in EfficientNet, LLaMA, and many modern models.
pub fn buildSiluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "silu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const x = try input.at(idx);

    // -x
    const neg_x = try builder.neg(x);

    // exp(-x)
    const exp_neg_x = try builder.exp(neg_x);

    // 1 + exp(-x)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg_x);

    // x / (1 + exp(-x))
    const silu_val = try builder.div(x, denom);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, silu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build SwiGLU kernel: output[i] = input[i] * silu(gate[i])
/// SwiGLU (Swish-Gated Linear Unit) is used in LLaMA, Mixtral, and PaLM.
/// Buffers: input (x), gate (g), output - computes x * SiLU(g)
pub fn buildSwigluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "swiglu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gate = try builder.addBuffer("gate", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // SwiGLU(x, g) = x * SiLU(g) = x * g / (1 + exp(-g))
    const x = try input.at(idx);
    const g = try gate.at(idx);

    // -g
    const neg_g = try builder.neg(g);

    // exp(-g)
    const exp_neg_g = try builder.exp(neg_g);

    // 1 + exp(-g)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg_g);

    // silu(g) = g / (1 + exp(-g))
    const silu_g = try builder.div(g, denom);

    // x * silu(g)
    const swiglu_val = try builder.mul(x, silu_g);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, swiglu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Tests
// ============================================================================

test "buildGeluKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildGeluKernel(allocator);
    try std.testing.expectEqualStrings("gelu", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len); // n
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}
