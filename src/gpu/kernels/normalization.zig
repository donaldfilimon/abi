//! Normalization Layer Kernel Definitions
//!
//! Pre-defined kernel IR for normalization operations.
//!
//! ## Operations
//! - layer_norm: Layer Normalization (transformers)
//! - rms_norm: RMS Normalization (LLaMA, T5)
//! - batch_norm: Batch Normalization (CNNs)
//! - fused_add_norm: Fused residual add + LayerNorm

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

/// Build LayerNorm kernel: output[i] = gamma[i] * (input[i] - mean) / sqrt(var + eps) + beta[i]
/// LayerNorm normalizes across the feature dimension for each sample.
/// Expects pre-computed mean and variance passed as uniforms.
pub fn buildLayerNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "layer_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const mean = try builder.addUniform("mean", Type.f32Type());
    const variance = try builder.addUniform("variance", Type.f32Type());
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = gamma[i] * (input[i] - mean) / sqrt(variance + epsilon) + beta[i]
    const x = try input.at(idx);
    const g = try gamma.at(idx);
    const b = try beta.at(idx);

    // input[i] - mean
    const centered = try builder.sub(x, try mean.toExpr());

    // variance + epsilon
    const var_eps = try builder.add(try variance.toExpr(), try epsilon.toExpr());

    // sqrt(variance + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // (input[i] - mean) / sqrt(variance + epsilon)
    const normalized = try builder.div(centered, std_dev);

    // gamma[i] * normalized
    const scaled = try builder.mul(g, normalized);

    // gamma[i] * normalized + beta[i]
    const layer_norm_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, layer_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build RMSNorm kernel: output[i] = gamma[i] * input[i] / sqrt(mean(x^2) + eps)
/// RMSNorm (Root Mean Square Normalization) is used in LLaMA, T5, and is simpler than LayerNorm.
/// Expects pre-computed RMS value passed as uniform.
pub fn buildRmsNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "rms_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const rms = try builder.addUniform("rms", Type.f32Type()); // pre-computed sqrt(mean(x^2))
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = gamma[i] * input[i] / (rms + epsilon)
    const x = try input.at(idx);
    const g = try gamma.at(idx);

    // rms + epsilon (for numerical stability)
    const rms_eps = try builder.add(try rms.toExpr(), try epsilon.toExpr());

    // input[i] / (rms + epsilon)
    const normalized = try builder.div(x, rms_eps);

    // gamma[i] * normalized
    const rms_norm_val = try builder.mul(g, normalized);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, rms_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build BatchNorm kernel (inference mode): output = gamma * (input - mean) / sqrt(var + eps) + beta
/// BatchNorm normalizes across the batch dimension for each feature.
/// Uses pre-computed running mean and variance from training.
pub fn buildBatchNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "batch_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const running_mean = try builder.addBuffer("running_mean", Type.f32Type(), .read_only);
    const running_var = try builder.addBuffer("running_var", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Compute channel index: c = idx % channels
    const c = try builder.mod(idx, try channels.toExpr());

    // Load per-channel parameters
    const x = try input.at(idx);
    const g = try gamma.at(c);
    const b = try beta.at(c);
    const mean_c = try running_mean.at(c);
    const var_c = try running_var.at(c);

    // (input - mean)
    const centered = try builder.sub(x, mean_c);

    // var + epsilon
    const var_eps = try builder.add(var_c, try epsilon.toExpr());

    // sqrt(var + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // (input - mean) / sqrt(var + epsilon)
    const normalized = try builder.div(centered, std_dev);

    // gamma * normalized + beta
    const scaled = try builder.mul(g, normalized);
    const batch_norm_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, batch_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fused add + LayerNorm kernel: output = LayerNorm(input + residual)
/// Commonly used in transformer architectures for residual connections.
pub fn buildFusedAddNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "fused_add_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const residual = try builder.addBuffer("residual", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const mean = try builder.addUniform("mean", Type.f32Type());
    const variance = try builder.addUniform("variance", Type.f32Type());
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Fused: output = LayerNorm(input + residual)
    const x = try input.at(idx);
    const r = try residual.at(idx);
    const g = try gamma.at(idx);
    const b = try beta.at(idx);

    // input + residual
    const added = try builder.add(x, r);

    // (added - mean)
    const centered = try builder.sub(added, try mean.toExpr());

    // variance + epsilon
    const var_eps = try builder.add(try variance.toExpr(), try epsilon.toExpr());

    // sqrt(variance + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // normalized
    const normalized = try builder.div(centered, std_dev);

    // gamma * normalized + beta
    const scaled = try builder.mul(g, normalized);
    const fused_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, fused_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fused linear + GELU kernel: output = GELU(input * weight + bias)
/// Fuses matrix-vector multiply with GELU activation for MLP layers.
/// Note: This is element-wise for pre-computed linear output.
pub fn buildFusedLinearGeluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "fused_linear_gelu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // For simplicity, this assumes linear output is pre-computed
    // and we just apply bias + GELU
    const linear_out = try builder.addBuffer("linear_out", Type.f32Type(), .read_only);
    const bias = try builder.addBuffer("bias", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());
    const hidden_dim = try builder.addUniform("hidden_dim", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // bias index = idx % hidden_dim
    const bias_idx = try builder.mod(idx, try hidden_dim.toExpr());

    // linear_out + bias
    const lo = try linear_out.at(idx);
    const b = try bias.at(bias_idx);
    const x = try builder.add(lo, b);

    // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const x_sq = try builder.mul(x, x);
    const x_cubed = try builder.mul(x_sq, x);
    const coef = try builder.f32Lit(0.044715);
    const coef_x3 = try builder.mul(coef, x_cubed);
    const inner_sum = try builder.add(x, coef_x3);
    const sqrt_2_pi = try builder.f32Lit(0.7978845608);
    const scaled = try builder.mul(sqrt_2_pi, inner_sum);
    const tanh_val = try builder.tanh(scaled);
    const one = try builder.f32Lit(1.0);
    const one_plus_tanh = try builder.add(one, tanh_val);
    const half = try builder.f32Lit(0.5);
    const half_x = try builder.mul(half, x);
    const gelu_val = try builder.mul(half_x, one_plus_tanh);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Tests
// ============================================================================

test "buildLayerNormKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildLayerNormKernel(allocator);
    try std.testing.expectEqualStrings("layer_norm", ir.name);
    try std.testing.expectEqual(@as(usize, 4), ir.buffers.len); // input, gamma, beta, output
    try std.testing.expectEqual(@as(usize, 4), ir.uniforms.len); // mean, variance, epsilon, n
}
