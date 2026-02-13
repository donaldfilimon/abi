//! FPGA-optimized attention kernels for LLM inference
//!
//! Provides hardware-accelerated implementations for:
//! - Streaming softmax with online normalization
//! - Multi-head attention with parallel compute units
//! - Flash attention with tiled memory access
//!
//! Performance targets (per FPGA research roadmap):
//! - Attention softmax: 5-10x improvement over CPU
//! - Memory efficiency: O(N) instead of O(N^2) for flash attention
//! - Latency: <100us for typical sequence lengths

const std = @import("std");
const build_options = @import("build_options");
const matmul_kernels = @import("matmul_kernels.zig");

/// Configuration for FPGA attention kernels
pub const AttentionKernelConfig = struct {
    /// Number of attention heads
    num_heads: u32 = 8,
    /// Head dimension (d_k = d_v)
    head_dim: u32 = 64,
    /// Maximum sequence length supported
    max_seq_len: u32 = 2048,
    /// Block size for tiled attention
    block_size: u32 = 64,
    /// Enable causal masking (for decoder-only models)
    causal: bool = true,
    /// Enable flash attention algorithm
    flash_attention: bool = true,
    /// Number of parallel compute units
    compute_units: u32 = 8,
    /// Precision for attention computation
    precision: AttentionPrecision = .fp16,
    /// Use on-chip BRAM for intermediate results
    use_bram: bool = true,
    /// Enable KV-cache integration
    with_kv_cache: bool = true,
};

/// Precision modes for attention computation
pub const AttentionPrecision = enum {
    fp32, // Full precision
    fp16, // Half precision (recommended for FPGA)
    bf16, // Brain float (better dynamic range)
    mixed, // FP16 compute, FP32 accumulation

    pub fn bits(self: AttentionPrecision) u8 {
        return switch (self) {
            .fp32 => 32,
            .fp16, .bf16 => 16,
            .mixed => 16, // Compute precision
        };
    }
};

/// Attention mask types
pub const AttentionMask = union(enum) {
    /// No masking
    none,
    /// Causal mask (lower triangular)
    causal: void,
    /// Custom mask tensor [seq_len, seq_len]
    custom: []const f32,
    /// Padding mask [batch, seq_len]
    padding: []const bool,
};

/// Online softmax state for streaming computation
pub const OnlineSoftmaxState = struct {
    /// Running maximum for numerical stability
    max_val: f32 = -std.math.inf(f32),
    /// Running sum of exp(x - max)
    sum_exp: f32 = 0,
    /// Count of values processed
    count: usize = 0,

    /// Update state with new value
    pub fn update(self: *OnlineSoftmaxState, value: f32) void {
        if (value > self.max_val) {
            // Rescale existing sum
            if (self.count > 0) {
                self.sum_exp *= @exp(self.max_val - value);
            }
            self.max_val = value;
        }
        self.sum_exp += @exp(value - self.max_val);
        self.count += 1;
    }

    /// Get normalization factor
    pub fn getNormFactor(self: *const OnlineSoftmaxState) f32 {
        if (self.sum_exp == 0) return 0;
        return 1.0 / self.sum_exp;
    }

    /// Reset state for new sequence
    pub fn reset(self: *OnlineSoftmaxState) void {
        self.max_val = -std.math.inf(f32);
        self.sum_exp = 0;
        self.count = 0;
    }
};

/// FPGA-accelerated streaming softmax kernel
pub const StreamingSoftmaxKernel = struct {
    config: AttentionKernelConfig,
    allocator: std.mem.Allocator,

    // Per-row online states for streaming computation
    row_states: ?[]OnlineSoftmaxState = null,

    pub fn init(allocator: std.mem.Allocator, config: AttentionKernelConfig) !StreamingSoftmaxKernel {
        var kernel = StreamingSoftmaxKernel{
            .config = config,
            .allocator = allocator,
        };

        // Pre-allocate state for maximum sequence length
        kernel.row_states = try allocator.alloc(OnlineSoftmaxState, config.max_seq_len);
        for (kernel.row_states.?) |*state| {
            state.* = OnlineSoftmaxState{};
        }

        return kernel;
    }

    pub fn deinit(self: *StreamingSoftmaxKernel) void {
        if (self.row_states) |states| {
            self.allocator.free(states);
        }
    }

    /// Execute streaming softmax on attention scores
    /// Input: [batch, heads, seq_len, seq_len] or [seq_len, seq_len] per head
    pub fn execute(
        self: *StreamingSoftmaxKernel,
        scores: []const f32,
        output: []f32,
        seq_len: usize,
        mask: AttentionMask,
    ) !void {
        std.debug.assert(scores.len == seq_len * seq_len);
        std.debug.assert(output.len == scores.len);

        // FPGA implementation would:
        // 1. Stream scores row by row
        // 2. Apply online softmax normalization
        // 3. Handle masking in hardware
        // 4. Output normalized attention weights

        // Reset states
        for (self.row_states.?[0..seq_len]) |*state| {
            state.reset();
        }

        // Phase 1: Compute max and sum per row (streaming)
        for (0..seq_len) |i| {
            const row_start = i * seq_len;
            var state = &self.row_states.?[i];

            for (0..seq_len) |j| {
                const score = scores[row_start + j];

                // Apply mask
                const masked_score = switch (mask) {
                    .none => score,
                    .causal => if (j > i) -std.math.inf(f32) else score,
                    .custom => |m| score + m[i * seq_len + j],
                    .padding => |p| if (p[j]) -std.math.inf(f32) else score,
                };

                state.update(masked_score);
            }
        }

        // Phase 2: Compute normalized outputs
        for (0..seq_len) |i| {
            const row_start = i * seq_len;
            const state = &self.row_states.?[i];
            const norm_factor = state.getNormFactor();

            for (0..seq_len) |j| {
                const score = scores[row_start + j];

                const masked_score = switch (mask) {
                    .none => score,
                    .causal => if (j > i) -std.math.inf(f32) else score,
                    .custom => |m| score + m[i * seq_len + j],
                    .padding => |p| if (p[j]) -std.math.inf(f32) else score,
                };

                if (masked_score == -std.math.inf(f32)) {
                    output[row_start + j] = 0;
                } else {
                    output[row_start + j] = @exp(masked_score - state.max_val) * norm_factor;
                }
            }
        }
    }

    /// Execute fused softmax with scaling (for attention: softmax(QK^T / sqrt(d_k)))
    pub fn executeFused(
        self: *StreamingSoftmaxKernel,
        scores: []const f32,
        output: []f32,
        seq_len: usize,
        scale: f32,
        mask: AttentionMask,
    ) !void {
        // Apply scale during streaming - more efficient on FPGA
        for (0..seq_len) |i| {
            const row_start = i * seq_len;
            var state = &self.row_states.?[i];
            state.reset();

            for (0..seq_len) |j| {
                const scaled_score = scores[row_start + j] * scale;

                const masked_score = switch (mask) {
                    .none => scaled_score,
                    .causal => if (j > i) -std.math.inf(f32) else scaled_score,
                    .custom => |m| scaled_score + m[i * seq_len + j],
                    .padding => |p| if (p[j]) -std.math.inf(f32) else scaled_score,
                };

                state.update(masked_score);
            }
        }

        // Output phase
        for (0..seq_len) |i| {
            const row_start = i * seq_len;
            const state = &self.row_states.?[i];
            const norm_factor = state.getNormFactor();

            for (0..seq_len) |j| {
                const scaled_score = scores[row_start + j] * scale;

                const masked_score = switch (mask) {
                    .none => scaled_score,
                    .causal => if (j > i) -std.math.inf(f32) else scaled_score,
                    .custom => |m| scaled_score + m[i * seq_len + j],
                    .padding => |p| if (p[j]) -std.math.inf(f32) else scaled_score,
                };

                if (masked_score == -std.math.inf(f32)) {
                    output[row_start + j] = 0;
                } else {
                    output[row_start + j] = @exp(masked_score - state.max_val) * norm_factor;
                }
            }
        }
    }
};

/// FPGA-accelerated multi-head attention kernel
pub const MultiHeadAttentionKernel = struct {
    config: AttentionKernelConfig,
    allocator: std.mem.Allocator,

    // Sub-kernels
    softmax_kernel: StreamingSoftmaxKernel,

    // Pre-allocated buffers
    qk_scores: ?[]f32 = null,
    attn_weights: ?[]f32 = null,

    pub fn init(allocator: std.mem.Allocator, config: AttentionKernelConfig) !MultiHeadAttentionKernel {
        var kernel = MultiHeadAttentionKernel{
            .config = config,
            .allocator = allocator,
            .softmax_kernel = try StreamingSoftmaxKernel.init(allocator, config),
        };

        // Pre-allocate for max sequence
        const max_scores = config.max_seq_len * config.max_seq_len;
        kernel.qk_scores = try allocator.alloc(f32, max_scores);
        kernel.attn_weights = try allocator.alloc(f32, max_scores);

        return kernel;
    }

    pub fn deinit(self: *MultiHeadAttentionKernel) void {
        self.softmax_kernel.deinit();
        if (self.qk_scores) |buf| self.allocator.free(buf);
        if (self.attn_weights) |buf| self.allocator.free(buf);
    }

    /// Execute single-head attention: softmax(QK^T / sqrt(d_k)) @ V
    /// Q: [seq_len, head_dim]
    /// K: [seq_len, head_dim]
    /// V: [seq_len, head_dim]
    /// Output: [seq_len, head_dim]
    pub fn executeSingleHead(
        self: *MultiHeadAttentionKernel,
        query: []const f32,
        key: []const f32,
        value: []const f32,
        output: []f32,
        seq_len: usize,
    ) !void {
        const head_dim = self.config.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        std.debug.assert(query.len == seq_len * head_dim);
        std.debug.assert(key.len == seq_len * head_dim);
        std.debug.assert(value.len == seq_len * head_dim);
        std.debug.assert(output.len == seq_len * head_dim);

        // Step 1: Q @ K^T -> [seq_len, seq_len]
        const scores = self.qk_scores.?[0 .. seq_len * seq_len];
        try self.computeQKT(query, key, scores, seq_len, head_dim);

        // Step 2: softmax(scores / sqrt(d_k)) -> attn_weights
        const weights = self.attn_weights.?[0 .. seq_len * seq_len];
        const mask: AttentionMask = if (self.config.causal) .causal else .none;
        try self.softmax_kernel.executeFused(scores, weights, seq_len, scale, mask);

        // Step 3: attn_weights @ V -> output
        try self.computeWeightedSum(weights, value, output, seq_len, head_dim);
    }

    /// Execute multi-head attention across all heads in parallel
    pub fn executeMultiHead(
        self: *MultiHeadAttentionKernel,
        queries: []const []const f32,
        keys: []const []const f32,
        values: []const []const f32,
        outputs: [][]f32,
        seq_len: usize,
    ) !void {
        const num_heads = self.config.num_heads;
        std.debug.assert(queries.len == num_heads);
        std.debug.assert(keys.len == num_heads);
        std.debug.assert(values.len == num_heads);
        std.debug.assert(outputs.len == num_heads);

        // FPGA would process all heads in parallel using multiple compute units
        // For CPU fallback, process sequentially
        for (0..num_heads) |h| {
            try self.executeSingleHead(queries[h], keys[h], values[h], outputs[h], seq_len);
        }
    }

    fn computeQKT(
        self: *const MultiHeadAttentionKernel,
        query: []const f32,
        key: []const f32,
        scores: []f32,
        seq_len: usize,
        head_dim: usize,
    ) !void {
        _ = self;

        // Q @ K^T: [seq_len, head_dim] @ [head_dim, seq_len] -> [seq_len, seq_len]
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var sum: f32 = 0;
                for (0..head_dim) |k| {
                    sum += query[i * head_dim + k] * key[j * head_dim + k];
                }
                scores[i * seq_len + j] = sum;
            }
        }
    }

    fn computeWeightedSum(
        self: *const MultiHeadAttentionKernel,
        weights: []const f32,
        value: []const f32,
        output: []f32,
        seq_len: usize,
        head_dim: usize,
    ) !void {
        _ = self;

        // weights @ V: [seq_len, seq_len] @ [seq_len, head_dim] -> [seq_len, head_dim]
        for (0..seq_len) |i| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..seq_len) |j| {
                    sum += weights[i * seq_len + j] * value[j * head_dim + d];
                }
                output[i * head_dim + d] = sum;
            }
        }
    }
};

/// Flash Attention implementation for FPGA
/// Uses tiled computation for O(N) memory complexity
pub const FlashAttentionKernel = struct {
    config: AttentionKernelConfig,
    allocator: std.mem.Allocator,

    // Tile buffers for block-wise computation
    q_tile: ?[]f32 = null,
    k_tile: ?[]f32 = null,
    v_tile: ?[]f32 = null,
    s_tile: ?[]f32 = null,
    o_tile: ?[]f32 = null,

    // Per-row tracking for online softmax
    row_max: ?[]f32 = null,
    row_sum: ?[]f32 = null,

    pub fn init(allocator: std.mem.Allocator, config: AttentionKernelConfig) !FlashAttentionKernel {
        var kernel = FlashAttentionKernel{
            .config = config,
            .allocator = allocator,
        };

        const block_size: usize = config.block_size;
        const head_dim: usize = config.head_dim;

        // Allocate tile buffers (overflow-checked)
        const bh = try std.math.mul(usize, block_size, head_dim);
        const bb = try std.math.mul(usize, block_size, block_size);
        kernel.q_tile = try allocator.alloc(f32, bh);
        kernel.k_tile = try allocator.alloc(f32, bh);
        kernel.v_tile = try allocator.alloc(f32, bh);
        kernel.s_tile = try allocator.alloc(f32, bb);
        kernel.o_tile = try allocator.alloc(f32, bh);

        kernel.row_max = try allocator.alloc(f32, block_size);
        kernel.row_sum = try allocator.alloc(f32, block_size);

        return kernel;
    }

    pub fn deinit(self: *FlashAttentionKernel) void {
        if (self.q_tile) |buf| self.allocator.free(buf);
        if (self.k_tile) |buf| self.allocator.free(buf);
        if (self.v_tile) |buf| self.allocator.free(buf);
        if (self.s_tile) |buf| self.allocator.free(buf);
        if (self.o_tile) |buf| self.allocator.free(buf);
        if (self.row_max) |buf| self.allocator.free(buf);
        if (self.row_sum) |buf| self.allocator.free(buf);
    }

    /// Execute flash attention with tiled memory access
    /// Achieves O(N) memory complexity vs O(N^2) for standard attention
    pub fn execute(
        self: *FlashAttentionKernel,
        query: []const f32,
        key: []const f32,
        value: []const f32,
        output: []f32,
        seq_len: usize,
    ) !void {
        const block_size = self.config.block_size;
        const head_dim = self.config.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Initialize output to zero
        @memset(output, 0);

        // Initialize row tracking
        @memset(self.row_max.?, -std.math.inf(f32));
        @memset(self.row_sum.?, 0);

        // Outer loop: iterate over Q blocks
        var q_block: usize = 0;
        while (q_block * block_size < seq_len) : (q_block += 1) {
            const q_start = q_block * block_size;
            const q_end = @min(q_start + block_size, seq_len);
            const q_len = q_end - q_start;

            // Load Q tile
            self.loadTile(query, self.q_tile.?, q_start, q_len, head_dim);

            // Initialize output accumulator for this Q block
            @memset(self.o_tile.?[0 .. q_len * head_dim], 0);

            // Inner loop: iterate over K/V blocks
            var kv_block: usize = 0;
            const max_kv = if (self.config.causal) q_block + 1 else (seq_len + block_size - 1) / block_size;

            while (kv_block < max_kv) : (kv_block += 1) {
                const kv_start = kv_block * block_size;
                const kv_end = @min(kv_start + block_size, seq_len);
                const kv_len = kv_end - kv_start;

                // Load K and V tiles
                self.loadTile(key, self.k_tile.?, kv_start, kv_len, head_dim);
                self.loadTile(value, self.v_tile.?, kv_start, kv_len, head_dim);

                // Compute S = Q @ K^T * scale
                self.computeScores(q_len, kv_len, head_dim, scale, q_start, kv_start);

                // Online softmax update and accumulate output
                self.updateOnlineSoftmax(q_len, kv_len, head_dim);
            }

            // Normalize output by final softmax sum
            self.normalizeOutput(q_len, head_dim);

            // Write output tile back
            self.storeTile(output, self.o_tile.?, q_start, q_len, head_dim);
        }
    }

    fn loadTile(self: *const FlashAttentionKernel, src: []const f32, dst: []f32, start: usize, len: usize, dim: usize) void {
        _ = self;
        for (0..len) |i| {
            @memcpy(dst[i * dim ..][0..dim], src[(start + i) * dim ..][0..dim]);
        }
    }

    fn storeTile(self: *const FlashAttentionKernel, dst: []f32, src: []const f32, start: usize, len: usize, dim: usize) void {
        _ = self;
        for (0..len) |i| {
            @memcpy(dst[(start + i) * dim ..][0..dim], src[i * dim ..][0..dim]);
        }
    }

    fn computeScores(self: *FlashAttentionKernel, q_len: usize, kv_len: usize, head_dim: usize, scale: f32, q_offset: usize, kv_offset: usize) void {
        const q_tile = self.q_tile.?;
        const k_tile = self.k_tile.?;
        const s_tile = self.s_tile.?;

        for (0..q_len) |i| {
            for (0..kv_len) |j| {
                // Causal masking
                if (self.config.causal and (kv_offset + j) > (q_offset + i)) {
                    s_tile[i * kv_len + j] = -std.math.inf(f32);
                    continue;
                }

                var sum: f32 = 0;
                for (0..head_dim) |k| {
                    sum += q_tile[i * head_dim + k] * k_tile[j * head_dim + k];
                }
                s_tile[i * kv_len + j] = sum * scale;
            }
        }
    }

    fn updateOnlineSoftmax(self: *FlashAttentionKernel, q_len: usize, kv_len: usize, head_dim: usize) void {
        const s_tile = self.s_tile.?;
        const v_tile = self.v_tile.?;
        const o_tile = self.o_tile.?;
        const row_max = self.row_max.?;
        const row_sum = self.row_sum.?;

        for (0..q_len) |i| {
            // Find max in current block
            var block_max: f32 = -std.math.inf(f32);
            for (0..kv_len) |j| {
                block_max = @max(block_max, s_tile[i * kv_len + j]);
            }

            // Update running max
            const prev_max = row_max[i];
            const new_max = @max(prev_max, block_max);
            row_max[i] = new_max;

            // Compute correction factor for previous sum
            const correction = if (prev_max == -std.math.inf(f32)) 0 else @exp(prev_max - new_max);

            // Update output with correction
            for (0..head_dim) |d| {
                o_tile[i * head_dim + d] *= correction;
            }
            row_sum[i] *= correction;

            // Accumulate new block
            for (0..kv_len) |j| {
                const score = s_tile[i * kv_len + j];
                if (score == -std.math.inf(f32)) continue;

                const weight = @exp(score - new_max);
                row_sum[i] += weight;

                for (0..head_dim) |d| {
                    o_tile[i * head_dim + d] += weight * v_tile[j * head_dim + d];
                }
            }
        }
    }

    fn normalizeOutput(self: *FlashAttentionKernel, q_len: usize, head_dim: usize) void {
        const o_tile = self.o_tile.?;
        const row_sum = self.row_sum.?;

        for (0..q_len) |i| {
            const sum = row_sum[i];
            if (sum > 0) {
                const norm = 1.0 / sum;
                for (0..head_dim) |d| {
                    o_tile[i * head_dim + d] *= norm;
                }
            }
        }
    }
};

/// Performance metrics for attention operations
pub const AttentionMetrics = struct {
    total_flops: u64 = 0,
    execution_time_ns: u64 = 0,
    memory_bytes: u64 = 0,
    num_heads: u32 = 0,
    seq_len: u32 = 0,

    pub fn computeGflops(self: *const AttentionMetrics) f64 {
        if (self.execution_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.total_flops)) / @as(f64, @floatFromInt(self.execution_time_ns));
    }

    /// Compute theoretical FLOPS for attention
    /// 2 * seq_len^2 * d_k (for QK^T) + 2 * seq_len^2 * d_v (for attn @ V)
    pub fn computeTheoreticalFlops(seq_len: u32, head_dim: u32, num_heads: u32) u64 {
        const qk_flops: u64 = 2 * @as(u64, seq_len) * seq_len * head_dim;
        const av_flops: u64 = 2 * @as(u64, seq_len) * seq_len * head_dim;
        return (qk_flops + av_flops) * num_heads;
    }

    pub fn report(self: *const AttentionMetrics) void {
        std.log.info("Attention Performance:", .{});
        std.log.info("  Heads: {d}, Seq Len: {d}", .{ self.num_heads, self.seq_len });
        std.log.info("  GFLOPS: {d:.2}", .{self.computeGflops()});
        std.log.info("  Memory: {d:.2} MB", .{@as(f64, @floatFromInt(self.memory_bytes)) / (1024 * 1024)});
    }
};

// Tests

test "online softmax state" {
    var state = OnlineSoftmaxState{};

    state.update(1.0);
    state.update(2.0);
    state.update(3.0);

    try std.testing.expect(state.max_val == 3.0);
    try std.testing.expect(state.count == 3);

    const norm = state.getNormFactor();
    try std.testing.expect(norm > 0 and norm < 1);
}

test "streaming softmax basic" {
    const allocator = std.testing.allocator;

    const config = AttentionKernelConfig{
        .max_seq_len = 4,
        .causal = false,
    };

    var kernel = try StreamingSoftmaxKernel.init(allocator, config);
    defer kernel.deinit();

    const scores = [_]f32{ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 };
    var output: [16]f32 = undefined;

    try kernel.execute(&scores, &output, 4, .none);

    // Check each row sums to 1
    for (0..4) |i| {
        var sum: f32 = 0;
        for (0..4) |j| {
            sum += output[i * 4 + j];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    }
}

test "causal masking" {
    const allocator = std.testing.allocator;

    const config = AttentionKernelConfig{
        .max_seq_len = 4,
        .causal = true,
    };

    var kernel = try StreamingSoftmaxKernel.init(allocator, config);
    defer kernel.deinit();

    const scores = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var output: [16]f32 = undefined;

    try kernel.execute(&scores, &output, 4, .causal);

    // Upper triangle should be zero
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[7], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[11], 0.001);

    // First row should be [1, 0, 0, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.001);
}

test "flash attention memory efficiency" {
    const allocator = std.testing.allocator;

    const config = AttentionKernelConfig{
        .max_seq_len = 128,
        .head_dim = 64,
        .block_size = 32,
        .causal = true,
    };

    var kernel = try FlashAttentionKernel.init(allocator, config);
    defer kernel.deinit();

    // Verify buffer sizes are block-sized, not full sequence sized
    try std.testing.expect(kernel.q_tile.?.len == 32 * 64);
    try std.testing.expect(kernel.s_tile.?.len == 32 * 32);
}
