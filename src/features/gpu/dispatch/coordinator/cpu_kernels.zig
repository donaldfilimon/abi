//! CPU Fallback Kernel Implementations
//!
//! Pure CPU implementations of GPU kernels used as fallback when
//! GPU execution is unavailable or fails.

const std = @import("std");
const dispatch_types = @import("../types.zig");
const unified_buffer = @import("../../internal/unified_buffer.zig");

const DispatchError = dispatch_types.DispatchError;
const Buffer = unified_buffer.Buffer;

pub fn executeCpuVectorAdd(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] + b_data[i];
    }
}

pub fn executeCpuVectorSub(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] - b_data[i];
    }
}

pub fn executeCpuVectorMul(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(a_data.len, @min(b_data.len, r_data.len));
    for (0..len) |i| {
        r_data[i] = a_data[i] * b_data[i];
    }
}

pub fn executeCpuReduceSum(input: *Buffer, result: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    var sum: f32 = 0;
    for (in_data) |v| {
        sum += v;
    }

    if (r_data.len > 0) {
        r_data[0] = sum;
    }
}

pub fn executeCpuDotProduct(a: *Buffer, b: *Buffer, result: *Buffer) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    var sum: f32 = 0;
    const len = @min(a_data.len, b_data.len);
    for (0..len) |i| {
        sum += a_data[i] * b_data[i];
    }

    if (r_data.len > 0) {
        r_data[0] = sum;
    }
}

pub fn executeCpuSoftmax(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    if (in_data.len == 0) return;

    const len = @min(in_data.len, out_data.len);

    // Find max for numerical stability
    var max_val: f32 = in_data[0];
    for (in_data[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0;
    for (0..len) |i| {
        out_data[i] = @exp(in_data[i] - max_val);
        sum += out_data[i];
    }

    // Normalize
    for (0..len) |i| {
        out_data[i] /= sum;
    }
}

pub fn executeCpuMatrixMultiply(
    a: *Buffer,
    b: *Buffer,
    result: *Buffer,
    m: u32,
    n: u32,
    k: u32,
) DispatchError!void {
    const a_data = std.mem.bytesAsSlice(f32, a.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, b.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    // C[i,j] = sum(A[i,k] * B[k,j])
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += a_data[i * k + kk] * b_data[kk * n + j];
            }
            r_data[i * n + j] = sum;
        }
    }
}

pub fn executeCpuBatchCosineSimilarity(
    query: *Buffer,
    vectors: *Buffer,
    result: *Buffer,
    num_vectors: u32,
    dim: u32,
    query_norm: f32,
) DispatchError!void {
    const q_data = std.mem.bytesAsSlice(f32, query.host_data orelse return DispatchError.BufferNotReady);
    const v_data = std.mem.bytesAsSlice(f32, vectors.host_data orelse return DispatchError.BufferNotReady);
    var r_data = std.mem.bytesAsSlice(f32, result.host_data orelse return DispatchError.BufferNotReady);

    const d = @as(usize, dim);
    const n = @min(@as(usize, num_vectors), r_data.len);

    for (0..n) |i| {
        var dot_sum: f32 = 0;
        var vec_norm_sq: f32 = 0;
        const vec_offset = i * d;

        for (0..d) |j| {
            if (j < q_data.len and vec_offset + j < v_data.len) {
                const q_val = q_data[j];
                const v_val = v_data[vec_offset + j];
                dot_sum += q_val * v_val;
                vec_norm_sq += v_val * v_val;
            }
        }

        const vec_norm = @sqrt(vec_norm_sq);
        const norm_product = query_norm * vec_norm;

        r_data[i] = if (norm_product > 1e-8)
            dot_sum / norm_product
        else
            0;
    }
}

pub fn executeCpuGelu(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, out_data.len);
    const sqrt_2_pi: f32 = 0.7978845608;
    const coef: f32 = 0.044715;

    for (0..len) |i| {
        const x = in_data[i];
        const x_cubed = x * x * x;
        const inner = sqrt_2_pi * (x + coef * x_cubed);
        const tanh_val = std.math.tanh(inner);
        out_data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

pub fn executeCpuSilu(input: *Buffer, output: *Buffer) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, out_data.len);

    for (0..len) |i| {
        const x = in_data[i];
        // SiLU(x) = x / (1 + exp(-x))
        out_data[i] = x / (1.0 + @exp(-x));
    }
}

pub fn executeCpuLayerNorm(
    input: *Buffer,
    gamma: *Buffer,
    beta: *Buffer,
    output: *Buffer,
    mean: f32,
    variance: f32,
    epsilon: f32,
) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    const g_data = std.mem.bytesAsSlice(f32, gamma.host_data orelse return DispatchError.BufferNotReady);
    const b_data = std.mem.bytesAsSlice(f32, beta.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, @min(out_data.len, @min(g_data.len, b_data.len)));
    const std_dev = @sqrt(variance + epsilon);

    for (0..len) |i| {
        const normalized = (in_data[i] - mean) / std_dev;
        out_data[i] = g_data[i] * normalized + b_data[i];
    }
}

pub fn executeCpuRmsNorm(
    input: *Buffer,
    gamma: *Buffer,
    output: *Buffer,
    rms: f32,
    epsilon: f32,
) DispatchError!void {
    const in_data = std.mem.bytesAsSlice(f32, input.host_data orelse return DispatchError.BufferNotReady);
    const g_data = std.mem.bytesAsSlice(f32, gamma.host_data orelse return DispatchError.BufferNotReady);
    var out_data = std.mem.bytesAsSlice(f32, output.host_data orelse return DispatchError.BufferNotReady);

    const len = @min(in_data.len, @min(out_data.len, g_data.len));
    const rms_eps = rms + epsilon;

    for (0..len) |i| {
        out_data[i] = g_data[i] * in_data[i] / rms_eps;
    }
}
