//! FPGA Kernel Implementations
//!
//! Provides optimized kernel implementations for FPGA-accelerated operations.
//! In production, these would map to pre-compiled bitstream kernels via HLS.

const std = @import("std");
const interface = @import("../../interface.zig");

pub const KernelError = error{
    UnsupportedKernel,
    InvalidArguments,
    ArgumentCountMismatch,
    ExecutionFailed,
    ConfigurationError,
};

/// FPGA kernel types optimized for hardware acceleration
pub const FpgaKernelType = enum {
    // Vector operations
    vector_distance, // Cosine, L2, dot product
    vector_add,
    vector_scale,

    // Matrix operations
    quantized_matmul, // Q4/Q8 quantized matrix multiply
    matmul, // FP32 matrix multiply
    matmul_blocked, // Blocked/tiled matrix multiply

    // Neural network operations
    softmax,
    rmsnorm,
    silu_activation,
    rope_embedding,
    attention,

    // Database operations
    kmeans_assign, // K-means centroid assignment
    hnsw_search, // HNSW graph traversal
    pq_encode, // Product quantization encoding
    pq_decode, // Product quantization decoding

    // Reduction operations
    reduce_sum,
    reduce_max,
    reduce_min,

    // Unknown/generic
    unknown,
};

/// Map kernel name to type
pub fn kernelTypeFromName(name: []const u8) FpgaKernelType {
    const mappings = .{
        .{ "vector_distance", .vector_distance },
        .{ "vector_add", .vector_add },
        .{ "vector_scale", .vector_scale },
        .{ "quantized_matmul", .quantized_matmul },
        .{ "matmul", .matmul },
        .{ "matmul_blocked", .matmul_blocked },
        .{ "softmax", .softmax },
        .{ "rmsnorm", .rmsnorm },
        .{ "silu", .silu_activation },
        .{ "rope", .rope_embedding },
        .{ "attention", .attention },
        .{ "kmeans_assign", .kmeans_assign },
        .{ "hnsw_search", .hnsw_search },
        .{ "pq_encode", .pq_encode },
        .{ "pq_decode", .pq_decode },
        .{ "reduce_sum", .reduce_sum },
        .{ "reduce_max", .reduce_max },
        .{ "reduce_min", .reduce_min },
    };

    inline for (mappings) |mapping| {
        if (std.mem.eql(u8, name, mapping[0])) {
            return mapping[1];
        }
    }

    return .unknown;
}

/// Configuration for vector distance kernel
pub const VectorDistanceConfig = struct {
    dimension: u32,
    batch_size: u32 = 1,
    distance_type: DistanceType = .cosine,

    pub const DistanceType = enum {
        cosine,
        l2,
        dot,
        l2_squared,
    };
};

/// Configuration for quantized matrix multiplication
pub const QuantizedMatmulConfig = struct {
    m: u32, // Output rows
    n: u32, // Output columns
    k: u32, // Inner dimension
    quantization: QuantizationType = .q4_0,
    use_bias: bool = false,

    pub const QuantizationType = enum {
        q4_0, // 4-bit, block size 32
        q4_1, // 4-bit with min
        q5_0, // 5-bit
        q5_1, // 5-bit with min
        q8_0, // 8-bit
        fp16, // Half precision
    };
};

/// Configuration for K-means centroid assignment
pub const KMeansConfig = struct {
    num_vectors: u32,
    num_centroids: u32,
    dimension: u32,
    distance_type: VectorDistanceConfig.DistanceType = .l2_squared,
};

/// Configuration for HNSW search
pub const HnswSearchConfig = struct {
    dimension: u32,
    ef_search: u32 = 100,
    k: u32 = 10, // Number of results
    max_level: u32 = 16,
};

/// Execute an FPGA kernel (simulation for development)
pub fn executeKernel(
    kernel_type: FpgaKernelType,
    config: interface.LaunchConfig,
    args: []const *anyopaque,
) KernelError!void {
    return switch (kernel_type) {
        .vector_distance => executeVectorDistance(config, args),
        .vector_add => executeVectorAdd(config, args),
        .quantized_matmul => executeQuantizedMatmul(config, args),
        .matmul => executeMatmul(config, args),
        .softmax => executeSoftmax(config, args),
        .kmeans_assign => executeKMeansAssign(config, args),
        .reduce_sum => executeReduceSum(config, args),
        else => {
            std.log.warn("FPGA kernel not implemented: {t}", .{kernel_type});
            return error.UnsupportedKernel;
        },
    };
}

// ============================================================================
// Kernel Implementations (CPU simulation for development/testing)
// ============================================================================

fn executeVectorDistance(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 4) return error.ArgumentCountMismatch;

    // args: [query, database_vectors, results, dim_ptr]
    const query: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const db_vectors: [*]const f32 = @ptrCast(@alignCast(args[1]));
    const results: [*]f32 = @ptrCast(@alignCast(args[2]));
    const params: *const VectorDistanceParams = @ptrCast(@alignCast(args[3]));

    const dim = params.dimension;
    const num_vectors = params.num_vectors;

    // Compute cosine similarity for each vector
    for (0..num_vectors) |i| {
        var dot: f32 = 0.0;
        var norm_q: f32 = 0.0;
        var norm_v: f32 = 0.0;

        for (0..dim) |j| {
            const q = query[j];
            const v = db_vectors[i * dim + j];
            dot += q * v;
            norm_q += q * q;
            norm_v += v * v;
        }

        const denom = @sqrt(norm_q) * @sqrt(norm_v);
        results[i] = if (denom > 1e-8) dot / denom else 0.0;
    }
}

const VectorDistanceParams = struct {
    dimension: usize,
    num_vectors: usize,
};

fn executeVectorAdd(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 4) return error.ArgumentCountMismatch;

    const a: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const b: [*]const f32 = @ptrCast(@alignCast(args[1]));
    const c: [*]f32 = @ptrCast(@alignCast(args[2]));
    const n_ptr: *const u32 = @ptrCast(@alignCast(args[3]));
    const n = n_ptr.*;

    for (0..n) |i| {
        c[i] = a[i] + b[i];
    }
}

fn executeQuantizedMatmul(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 4) return error.ArgumentCountMismatch;

    // Simplified Q4 matmul simulation
    // In real FPGA, this would use dedicated dequantization + MAC units
    const a_quant: [*]const u8 = @ptrCast(@alignCast(args[0]));
    const b: [*]const f32 = @ptrCast(@alignCast(args[1]));
    const c: [*]f32 = @ptrCast(@alignCast(args[2]));
    const params: *const QuantizedMatmulParams = @ptrCast(@alignCast(args[3]));

    const m = params.m;
    const n = params.n;
    const k = params.k;

    // Dequantize and multiply (simplified)
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |idx| {
                // Simplified Q4 dequantization
                const quant_val = a_quant[row * k + idx];
                const dequant: f32 = @as(f32, @floatFromInt(@as(i8, @bitCast(quant_val)))) / 16.0;
                sum += dequant * b[idx * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

const QuantizedMatmulParams = struct {
    m: usize,
    n: usize,
    k: usize,
};

fn executeMatmul(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 4) return error.ArgumentCountMismatch;

    const a: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const b: [*]const f32 = @ptrCast(@alignCast(args[1]));
    const c: [*]f32 = @ptrCast(@alignCast(args[2]));
    const params: *const QuantizedMatmulParams = @ptrCast(@alignCast(args[3]));

    const m = params.m;
    const n = params.n;
    const k = params.k;

    // Standard matrix multiplication
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |idx| {
                sum += a[row * k + idx] * b[idx * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

fn executeSoftmax(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 3) return error.ArgumentCountMismatch;

    const input: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const output: [*]f32 = @ptrCast(@alignCast(args[1]));
    const params: *const SoftmaxParams = @ptrCast(@alignCast(args[2]));

    const batch_size = params.batch_size;
    const seq_len = params.seq_len;

    for (0..batch_size) |b| {
        const offset = b * seq_len;

        // Find max for numerical stability
        var max_val: f32 = input[offset];
        for (1..seq_len) |i| {
            max_val = @max(max_val, input[offset + i]);
        }

        // Compute exp and sum
        var sum: f32 = 0.0;
        for (0..seq_len) |i| {
            output[offset + i] = @exp(input[offset + i] - max_val);
            sum += output[offset + i];
        }

        // Normalize
        for (0..seq_len) |i| {
            output[offset + i] /= sum;
        }
    }
}

const SoftmaxParams = struct {
    batch_size: usize,
    seq_len: usize,
};

fn executeKMeansAssign(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 4) return error.ArgumentCountMismatch;

    const vectors: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const centroids: [*]const f32 = @ptrCast(@alignCast(args[1]));
    const assignments: [*]u32 = @ptrCast(@alignCast(args[2]));
    const params: *const KMeansParams = @ptrCast(@alignCast(args[3]));

    const num_vectors = params.num_vectors;
    const num_centroids = params.num_centroids;
    const dim = params.dimension;

    // Assign each vector to nearest centroid
    for (0..num_vectors) |v| {
        var min_dist: f32 = std.math.floatMax(f32);
        var best_centroid: u32 = 0;

        for (0..num_centroids) |c| {
            var dist: f32 = 0.0;
            for (0..dim) |d| {
                const diff = vectors[v * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_centroid = @intCast(c);
            }
        }

        assignments[v] = best_centroid;
    }
}

const KMeansParams = struct {
    num_vectors: usize,
    num_centroids: usize,
    dimension: usize,
};

fn executeReduceSum(config: interface.LaunchConfig, args: []const *anyopaque) KernelError!void {
    _ = config;
    if (args.len < 3) return error.ArgumentCountMismatch;

    const input: [*]const f32 = @ptrCast(@alignCast(args[0]));
    const output: [*]f32 = @ptrCast(@alignCast(args[1]));
    const n_ptr: *const u32 = @ptrCast(@alignCast(args[2]));
    const n = n_ptr.*;

    var sum: f32 = 0.0;
    for (0..n) |i| {
        sum += input[i];
    }
    output[0] = sum;
}

// ============================================================================
// HLS Code Templates (for reference - actual HLS would be in separate C++ files)
// ============================================================================

/// Example HLS pragma annotations that would be used in C++ HLS code
pub const hls_templates = struct {
    /// Vector distance HLS template
    pub const vector_distance_hls =
        \\// HLS C++ for vector distance computation
        \\void vector_distance(
        \\    const float* query,
        \\    const float* vectors,
        \\    float* results,
        \\    int dimension,
        \\    int num_vectors
        \\) {
        \\    #pragma HLS INTERFACE m_axi port=query offset=slave bundle=gmem0
        \\    #pragma HLS INTERFACE m_axi port=vectors offset=slave bundle=gmem1
        \\    #pragma HLS INTERFACE m_axi port=results offset=slave bundle=gmem2
        \\    #pragma HLS INTERFACE s_axilite port=dimension bundle=control
        \\    #pragma HLS INTERFACE s_axilite port=num_vectors bundle=control
        \\    #pragma HLS INTERFACE s_axilite port=return bundle=control
        \\
        \\    float query_local[MAX_DIM];
        \\    #pragma HLS ARRAY_PARTITION variable=query_local cyclic factor=16
        \\
        \\    // Load query vector
        \\    for (int i = 0; i < dimension; i++) {
        \\        #pragma HLS PIPELINE II=1
        \\        query_local[i] = query[i];
        \\    }
        \\
        \\    // Compute distances
        \\    for (int v = 0; v < num_vectors; v++) {
        \\        float dot = 0, norm_q = 0, norm_v = 0;
        \\
        \\        for (int i = 0; i < dimension; i++) {
        \\            #pragma HLS PIPELINE II=1
        \\            #pragma HLS UNROLL factor=16
        \\            float q = query_local[i];
        \\            float db = vectors[v * dimension + i];
        \\            dot += q * db;
        \\            norm_q += q * q;
        \\            norm_v += db * db;
        \\        }
        \\
        \\        results[v] = dot / (sqrt(norm_q) * sqrt(norm_v));
        \\    }
        \\}
    ;

    /// Quantized matmul HLS template
    pub const quantized_matmul_hls =
        \\// HLS C++ for Q4 quantized matrix multiplication
        \\void quantized_matmul(
        \\    const ap_uint<4>* A,  // Quantized weights
        \\    const float* scales,   // Dequantization scales
        \\    const float* B,        // Input activations
        \\    float* C,              // Output
        \\    int M, int N, int K
        \\) {
        \\    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
        \\    #pragma HLS INTERFACE m_axi port=scales offset=slave bundle=gmem0
        \\    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
        \\    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
        \\
        \\    // Block size for tiling
        \\    const int BLOCK = 64;
        \\
        \\    for (int m = 0; m < M; m += BLOCK) {
        \\        for (int n = 0; n < N; n += BLOCK) {
        \\            // Local accumulator
        \\            float acc[BLOCK][BLOCK];
        \\            #pragma HLS ARRAY_PARTITION variable=acc complete dim=2
        \\
        \\            // Initialize
        \\            for (int i = 0; i < BLOCK; i++) {
        \\                #pragma HLS UNROLL
        \\                for (int j = 0; j < BLOCK; j++) {
        \\                    #pragma HLS UNROLL
        \\                    acc[i][j] = 0;
        \\                }
        \\            }
        \\
        \\            // Compute block
        \\            for (int k = 0; k < K; k++) {
        \\                #pragma HLS PIPELINE II=1
        \\                // Dequantize and MAC
        \\                // ...
        \\            }
        \\
        \\            // Write back
        \\            for (int i = 0; i < BLOCK; i++) {
        \\                for (int j = 0; j < BLOCK; j++) {
        \\                    #pragma HLS PIPELINE II=1
        \\                    C[(m+i)*N + (n+j)] = acc[i][j];
        \\                }
        \\            }
        \\        }
        \\    }
        \\}
    ;
};

test "kernel type mapping" {
    try std.testing.expectEqual(FpgaKernelType.vector_distance, kernelTypeFromName("vector_distance"));
    try std.testing.expectEqual(FpgaKernelType.quantized_matmul, kernelTypeFromName("quantized_matmul"));
    try std.testing.expectEqual(FpgaKernelType.unknown, kernelTypeFromName("nonexistent"));
}
