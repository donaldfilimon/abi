//! Kernel fusion types.

const std = @import("std");

pub const OpType = enum {
    add,
    sub,
    mul,
    div,
    neg,
    abs,
    sqrt,
    rsqrt,
    exp,
    log,
    pow,
    max,
    min,
    clamp,
    relu,
    leaky_relu,
    sigmoid,
    tanh,
    gelu,
    gelu_fast,
    silu,
    swiglu,
    softmax,
    layer_norm,
    rms_norm,
    batch_norm,
    reduce_sum,
    reduce_max,
    reduce_min,
    reduce_mean,
    matmul,
    batch_matmul,
    transpose,
    copy,
    broadcast,
    gather,
    scatter,
    fused_add_relu,
    fused_add_gelu,
    fused_mul_add,
    fused_layer_norm_gelu,
    fused_linear_relu,
    fused_linear_gelu,
    fused_linear_silu,
    dot_product,
    fused_attention_qk,
    fused_attention_sv,

    pub fn isElementWise(self: OpType) bool {
        return switch (self) {
            .add, .sub, .mul, .div, .neg, .abs, .sqrt, .rsqrt, .exp, .log, .pow, .max, .min, .clamp, .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu, .copy => true,
            else => false,
        };
    }

    pub fn isFusable(self: OpType) bool {
        return self.isElementWise() or self.isActivation() or self.isNormalization() or
            self == .matmul or self == .softmax;
    }

    pub fn isAttentionRelated(self: OpType) bool {
        return self == .matmul or self == .softmax or self == .fused_attention_qk or
            self == .fused_attention_sv;
    }

    pub fn isReduction(self: OpType) bool {
        return switch (self) {
            .reduce_sum, .reduce_max, .reduce_min, .reduce_mean, .softmax => true,
            else => false,
        };
    }

    pub fn isActivation(self: OpType) bool {
        return switch (self) {
            .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu, .swiglu => true,
            else => false,
        };
    }

    pub fn isNormalization(self: OpType) bool {
        return switch (self) {
            .layer_norm, .rms_norm, .batch_norm => true,
            else => false,
        };
    }

    pub fn flopsPerElement(self: OpType) u32 {
        return switch (self) {
            .add, .sub, .mul, .div, .neg, .abs, .max, .min, .copy => 1,
            .sqrt, .rsqrt => 4,
            .exp, .log => 8,
            .pow => 16,
            .relu, .leaky_relu => 1,
            .sigmoid => 4,
            .tanh => 8,
            .gelu => 16,
            .gelu_fast => 8,
            .silu => 5,
            .layer_norm => 8,
            .rms_norm => 6,
            .reduce_sum, .reduce_max, .reduce_min, .reduce_mean => 1,
            .matmul => 2,
            .fused_add_relu => 2,
            .fused_add_gelu => 17,
            .fused_mul_add => 2,
            .fused_linear_relu => 3,
            .fused_linear_gelu => 18,
            .fused_linear_silu => 6,
            .fused_attention_qk => 20,
            .fused_attention_sv => 10,
            .dot_product => 2,
            else => 4,
        };
    }

    pub fn memoryBytesPerElement(self: OpType) u32 {
        return switch (self) {
            .add, .sub, .mul, .div, .max, .min => 12,
            .neg, .abs, .sqrt, .rsqrt, .exp, .log, .copy => 8,
            .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu => 8,
            .pow => 12,
            .layer_norm => 16,
            .rms_norm => 12,
            .fused_add_relu, .fused_add_gelu => 12,
            .fused_mul_add => 16,
            .fused_linear_relu, .fused_linear_gelu, .fused_linear_silu => 20,
            .fused_attention_qk => 24,
            .fused_attention_sv => 16,
            else => 8,
        };
    }
};

pub const BufferHandle = u32;
pub const NO_BUFFER: BufferHandle = std.math.maxInt(BufferHandle);

pub const OpNode = struct {
    op: OpType,
    inputs: [4]BufferHandle = .{ NO_BUFFER, NO_BUFFER, NO_BUFFER, NO_BUFFER },
    num_inputs: u8 = 0,
    output: BufferHandle = NO_BUFFER,
    element_count: usize = 0,
    fused: bool = false,
    fused_into: ?u32 = null,
    scalar_params: [4]f32 = .{ 0, 0, 0, 0 },

    pub fn addInput(self: *OpNode, buf: BufferHandle) void {
        if (self.num_inputs < 4) {
            self.inputs[self.num_inputs] = buf;
            self.num_inputs += 1;
        }
    }

    pub fn getInputs(self: *const OpNode) []const BufferHandle {
        return self.inputs[0..self.num_inputs];
    }
};

pub const FusionPattern = struct {
    first_op_idx: u32,
    last_op_idx: u32,
    fused_op: OpType,
    speedup: f32,
    bandwidth_saved: usize,
    chain: [8]u32 = undefined,
    chain_len: u8 = 0,
};

pub const FusionStats = struct {
    ops_recorded: usize = 0,
    patterns_detected: usize = 0,
    fusions_applied: usize = 0,
    bandwidth_saved_bytes: usize = 0,
    estimated_speedup: f32 = 1.0,
};

pub fn calculateSpeedup(op1: OpType, op2: OpType, element_count: usize) f32 {
    const mem1 = @as(f32, @floatFromInt(op1.memoryBytesPerElement()));
    const mem2 = @as(f32, @floatFromInt(op2.memoryBytesPerElement()));
    const intermediate_saved: f32 = 8;

    const original_mem = mem1 + mem2;
    const fused_mem = original_mem - intermediate_saved;

    var speedup = original_mem / @max(fused_mem, 1);

    if (element_count < 10000) {
        speedup += 0.1;
    }

    return @max(speedup, 1.0);
}

test {
    std.testing.refAllDecls(@This());
}
