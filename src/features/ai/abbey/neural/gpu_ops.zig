//! GPU-accelerated Abbey Neural Operations
//!
//! Provides GPU acceleration for Abbey's neural network operations with
//! automatic fallback to CPU when GPU is unavailable. Implements adaptive
//! dispatch based on operation size thresholds to maximize efficiency.
//!
//! GPU access is provided through the centralized ai_ops interface, which
//! handles compile-time gating and provides stub implementations when GPU
//! is disabled.

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const build_options = @import("build_options");

// Centralized GPU interface - handles compile-time gating and stubs
// Import from the top-level gpu module
const ai_ops = @import("abi").gpu.ai_ops;

// Re-export GPU modules from ai_ops (stubs provided when GPU disabled)
const cuda_mod = struct {
    pub const llm_kernels = ai_ops.llm_kernels;
    pub const memory = ai_ops.memory;
};
const cublas = ai_ops.cublas;
const backend_mod = ai_ops.backend;

/// GPU operations context for Abbey neural computations.
/// Provides unified interface with automatic CPU fallback.
pub const GpuOpsContext = struct {
    allocator: std.mem.Allocator,
    gpu_available: bool,
    cublas_available: bool,
    kernels_available: bool,
    device_id: u32,

    /// cuBLAS context for accelerated GEMM operations
    cublas_ctx: ?cublas.CublasContext,

    /// CUDA kernel module for activation functions
    llm_kernels: ?cuda_mod.llm_kernels.LlmKernelModule,

    /// Scratch buffer for GPU operations
    scratch_buffer: ?[]f32,
    scratch_size: usize,

    /// Statistics tracking
    stats: GpuStats,

    /// Threshold for GPU dispatch (operations below this use CPU)
    /// Set to 4096 elements as smaller operations have overhead > benefit
    pub const GPU_THRESHOLD: usize = 4096;

    const Self = @This();

    /// Initialize GPU operations context.
    /// Automatically detects GPU availability and initializes backends.
    pub fn init(allocator: std.mem.Allocator) Self {
        var gpu_available = build_options.enable_gpu and checkGpuAvailability();
        const cublas_present = build_options.enable_gpu and cublas.isAvailable();
        const kernels_present = build_options.enable_gpu and cuda_mod.llm_kernels.isAvailable();

        var cublas_ctx: ?cublas.CublasContext = null;
        var cublas_available = false;
        var llm_kernels: ?cuda_mod.llm_kernels.LlmKernelModule = null;
        var kernels_available = false;
        var memory_ready = false;

        if (gpu_available and (cublas_present or kernels_present)) {
            if (cuda_mod.memory.init()) |_| {
                memory_ready = true;
            } else |err| {
                std.log.warn("CUDA memory init failed: {t}", .{err});
            }
        }

        if (memory_ready and cublas_present) {
            cublas_ctx = cublas.CublasContext.init() catch null;
            cublas_available = cublas_ctx != null;
            if (cublas_available) {
                std.log.info("Abbey GPU: cuBLAS initialized for matrix operations", .{});
            }
        }

        if (memory_ready and kernels_present) {
            llm_kernels = cuda_mod.llm_kernels.LlmKernelModule.init(allocator) catch null;
            kernels_available = llm_kernels != null;
            if (kernels_available) {
                std.log.info("Abbey GPU: CUDA kernels initialized (softmax, layerNorm, activations)", .{});
            }
        }

        gpu_available = gpu_available and memory_ready and (cublas_available or kernels_available);

        return .{
            .allocator = allocator,
            .gpu_available = gpu_available,
            .cublas_available = cublas_available,
            .kernels_available = kernels_available,
            .device_id = 0,
            .cublas_ctx = cublas_ctx,
            .llm_kernels = llm_kernels,
            .scratch_buffer = null,
            .scratch_size = 0,
            .stats = .{},
        };
    }

    /// Release GPU resources.
    pub fn deinit(self: *Self) void {
        if (self.llm_kernels) |*kernels| {
            kernels.deinit();
        }
        if (self.cublas_ctx) |*ctx| {
            ctx.deinit();
        }
        if (self.scratch_buffer) |buf| {
            self.allocator.free(buf);
        }
        self.* = undefined;
    }

    /// Ensure scratch buffer is at least the given size.
    pub fn ensureScratchBuffer(self: *Self, size: usize) ![]f32 {
        if (self.scratch_buffer == null or self.scratch_size < size) {
            if (self.scratch_buffer) |buf| {
                self.allocator.free(buf);
            }
            self.scratch_buffer = try self.allocator.alloc(f32, size);
            self.scratch_size = size;
        }
        return self.scratch_buffer.?;
    }

    /// Check if GPU operations are available.
    pub fn isGpuAvailable(self: *const Self) bool {
        return self.gpu_available;
    }

    // ========================================================================
    // Matrix Operations
    // ========================================================================

    /// Matrix multiplication: C = A @ B
    /// Automatically dispatches to GPU for large matrices, CPU for small.
    pub fn matmul(self: *Self, a: []const f32, b: []const f32, c: []f32, m: u32, n: u32, k: u32) void {
        const size = @as(usize, m) * n * k;
        if (self.gpu_available and size > GPU_THRESHOLD) {
            self.matmulGpu(a, b, c, m, n, k) catch {
                self.matmulCpu(a, b, c, m, n, k);
            };
        } else {
            self.matmulCpu(a, b, c, m, n, k);
        }
    }

    fn matmulGpu(self: *Self, a: []const f32, b: []const f32, c: []f32, m: u32, n: u32, k: u32) !void {
        var timer = time.Timer.start() catch null;

        if (self.cublas_ctx) |*ctx| {
            const a_size = @as(usize, m) * k * @sizeOf(f32);
            const b_size = @as(usize, k) * n * @sizeOf(f32);
            const c_size = @as(usize, m) * n * @sizeOf(f32);

            var a_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, a_size);
            defer a_dev.deinit();
            var b_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, b_size);
            defer b_dev.deinit();
            var c_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, c_size);
            defer c_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), a_size);
            try cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b_size);

            // cuBLAS uses column-major, we use row-major
            // C = A @ B in row-major = C^T = B^T @ A^T in col-major
            try ctx.sgemm(
                .no_trans,
                .no_trans,
                @intCast(n),
                @intCast(m),
                @intCast(k),
                1.0,
                @ptrCast(b_dev.ptr.?),
                @intCast(n),
                @ptrCast(a_dev.ptr.?),
                @intCast(k),
                0.0,
                @ptrCast(c_dev.ptr.?),
                @intCast(n),
            );

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(c.ptr), c_dev.ptr.?, c_size);
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn matmulCpu(self: *Self, a: []const f32, b: []const f32, c: []f32, m: u32, n: u32, k: u32) void {
        var timer = time.Timer.start() catch null;

        // Naive matmul - could use SIMD for larger matrices
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |l| {
                    acc += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = acc;
            }
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    // ========================================================================
    // Attention Operations
    // ========================================================================

    /// Scaled dot-product attention with GPU acceleration.
    /// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
    pub fn attention(
        self: *Self,
        q: []const f32,
        k_data: []const f32,
        v: []const f32,
        output: []f32,
        seq_len: u32,
        head_dim: u32,
    ) void {
        const size = @as(usize, seq_len) * seq_len * head_dim;
        if (self.gpu_available and size > GPU_THRESHOLD) {
            self.attentionGpu(q, k_data, v, output, seq_len, head_dim) catch {
                self.attentionCpu(q, k_data, v, output, seq_len, head_dim);
            };
        } else {
            self.attentionCpu(q, k_data, v, output, seq_len, head_dim);
        }
    }

    fn attentionGpu(
        self: *Self,
        q: []const f32,
        k_data: []const f32,
        v: []const f32,
        output: []f32,
        seq_len: u32,
        head_dim: u32,
    ) !void {
        const kernels = if (self.llm_kernels) |*value| value else return error.NotAvailable;
        const ctx = if (self.cublas_ctx) |*value| value else return error.NotAvailable;

        var timer = time.Timer.start() catch null;

        const seq_usize = @as(usize, seq_len);
        const head_usize = @as(usize, head_dim);
        const scores_size = seq_usize * seq_usize;
        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Allocate device memory
        const q_bytes = seq_usize * head_usize * @sizeOf(f32);
        const k_bytes = seq_usize * head_usize * @sizeOf(f32);
        const v_bytes = seq_usize * head_usize * @sizeOf(f32);
        const scores_bytes = scores_size * @sizeOf(f32);

        var q_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, q_bytes);
        defer q_dev.deinit();
        var k_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, k_bytes);
        defer k_dev.deinit();
        var v_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, v_bytes);
        defer v_dev.deinit();
        var scores_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, scores_bytes);
        defer scores_dev.deinit();
        var out_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, q_bytes);
        defer out_dev.deinit();

        // Copy to device
        try cuda_mod.memory.memcpyHostToDevice(q_dev.ptr.?, @ptrCast(q.ptr), q_bytes);
        try cuda_mod.memory.memcpyHostToDevice(k_dev.ptr.?, @ptrCast(k_data.ptr), k_bytes);
        try cuda_mod.memory.memcpyHostToDevice(v_dev.ptr.?, @ptrCast(v.ptr), v_bytes);

        // Q @ K^T -> scores
        try ctx.sgemm(
            .trans,
            .no_trans,
            @intCast(seq_len),
            @intCast(seq_len),
            @intCast(head_dim),
            attn_scale,
            @ptrCast(k_dev.ptr.?),
            @intCast(head_dim),
            @ptrCast(q_dev.ptr.?),
            @intCast(head_dim),
            0.0,
            @ptrCast(scores_dev.ptr.?),
            @intCast(seq_len),
        );

        // Softmax over scores (row-wise)
        for (0..seq_usize) |row| {
            const row_offset = row * seq_usize * @sizeOf(f32);
            try kernels.softmax(
                @intFromPtr(scores_dev.ptr.?) + row_offset,
                seq_len,
                null,
            );
        }

        // scores @ V -> output
        try ctx.sgemm(
            .no_trans,
            .no_trans,
            @intCast(head_dim),
            @intCast(seq_len),
            @intCast(seq_len),
            1.0,
            @ptrCast(v_dev.ptr.?),
            @intCast(head_dim),
            @ptrCast(scores_dev.ptr.?),
            @intCast(seq_len),
            0.0,
            @ptrCast(out_dev.ptr.?),
            @intCast(head_dim),
        );

        // Copy back
        try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(output.ptr), out_dev.ptr.?, q_bytes);
        self.stats.addOp(if (timer) |*t| t.read() else 0, true);
    }

    fn attentionCpu(
        self: *Self,
        q: []const f32,
        k_data: []const f32,
        v: []const f32,
        output: []f32,
        seq_len: u32,
        head_dim: u32,
    ) void {
        var timer = time.Timer.start() catch null;

        const seq_usize = @as(usize, seq_len);
        const head_usize = @as(usize, head_dim);
        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Allocate scores on stack if small enough, otherwise use scratch
        var scores_buf: [256]f32 = undefined;
        const scores = if (seq_usize * seq_usize <= 256)
            scores_buf[0 .. seq_usize * seq_usize]
        else
            self.ensureScratchBuffer(seq_usize * seq_usize) catch {
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

        // Q @ K^T
        for (0..seq_usize) |i| {
            for (0..seq_usize) |j| {
                var dot: f32 = 0;
                for (0..head_usize) |d| {
                    dot += q[i * head_usize + d] * k_data[j * head_usize + d];
                }
                scores[i * seq_usize + j] = dot * attn_scale;
            }
        }

        // Softmax over each row
        for (0..seq_usize) |i| {
            const row_start = i * seq_usize;
            const row = scores[row_start .. row_start + seq_usize];

            // Find max for numerical stability
            var max_val: f32 = row[0];
            for (row[1..]) |val| {
                if (val > max_val) max_val = val;
            }

            // Exp and sum
            var exp_sum: f32 = 0;
            for (row) |*val| {
                val.* = @exp(val.* - max_val);
                exp_sum += val.*;
            }

            // Normalize
            for (row) |*val| {
                val.* /= exp_sum;
            }
        }

        // scores @ V
        for (0..seq_usize) |i| {
            for (0..head_usize) |d| {
                var sum: f32 = 0;
                for (0..seq_usize) |j| {
                    sum += scores[i * seq_usize + j] * v[j * head_usize + d];
                }
                output[i * head_usize + d] = sum;
            }
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    // ========================================================================
    // Normalization Operations
    // ========================================================================

    /// Layer normalization with GPU acceleration.
    /// Normalizes input to zero mean and unit variance, then scales and shifts.
    pub fn layerNorm(
        self: *Self,
        x: []f32,
        gamma: []const f32,
        beta: []const f32,
        eps: f32,
    ) void {
        if (self.gpu_available and x.len > GPU_THRESHOLD) {
            self.layerNormGpu(x, gamma, beta, eps) catch {
                self.layerNormCpu(x, gamma, beta, eps);
            };
        } else {
            self.layerNormCpu(x, gamma, beta, eps);
        }
    }

    fn layerNormGpu(
        self: *Self,
        x: []f32,
        gamma: []const f32,
        beta: []const f32,
        eps: f32,
    ) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = x.len * @sizeOf(f32);
            var x_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer x_dev.deinit();
            var gamma_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, gamma.len * @sizeOf(f32));
            defer gamma_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size);
            try cuda_mod.memory.memcpyHostToDevice(gamma_dev.ptr.?, @ptrCast(gamma.ptr), gamma.len * @sizeOf(f32));

            try kernels.rmsnorm(
                @intFromPtr(x_dev.ptr.?),
                @intFromPtr(gamma_dev.ptr.?),
                @intCast(x.len),
                eps,
                null,
            );

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size);

            // Apply beta shift on CPU (not in RMSNorm kernel)
            for (x, 0..) |*val, i| {
                val.* += beta[i % beta.len];
            }

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn layerNormCpu(self: *Self, x: []f32, gamma: []const f32, beta: []const f32, eps: f32) void {
        var timer = time.Timer.start() catch null;

        const n = x.len;
        const dim = gamma.len;
        const batch_size = n / dim;

        for (0..batch_size) |b| {
            const start = b * dim;
            const end = start + dim;
            const slice = x[start..end];

            // Compute mean
            var sum: f32 = 0;
            for (slice) |v| sum += v;
            const mean_val = sum / @as(f32, @floatFromInt(dim));

            // Compute variance
            var var_sum: f32 = 0;
            for (slice) |v| {
                const diff = v - mean_val;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(dim));
            const std_dev = @sqrt(variance + eps);

            // Normalize and apply scale/shift
            for (0..dim) |i| {
                const normalized = (slice[i] - mean_val) / std_dev;
                x[start + i] = normalized * gamma[i] + beta[i];
            }
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    /// ReLU activation: max(0, x)
    pub fn relu(self: *Self, x: []f32) void {
        var timer = time.Timer.start() catch null;

        // ReLU is simple enough that GPU overhead isn't worth it for most sizes
        for (x) |*val| {
            val.* = @max(0, val.*);
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(self: *Self, x: []f32) void {
        if (self.gpu_available and x.len > GPU_THRESHOLD) {
            self.geluGpu(x) catch {
                self.geluCpu(x);
            };
        } else {
            self.geluCpu(x);
        }
    }

    fn geluGpu(self: *Self, x: []f32) !void {
        // Use dedicated GELU kernel for GPU acceleration
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = x.len * @sizeOf(f32);
            var x_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer x_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size);

            // Try dedicated GELU kernel first
            kernels.gelu(@intFromPtr(x_dev.ptr.?), @intCast(x.len), null) catch {
                // Fall back to CPU implementation if GELU kernel not available
                try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size);
                self.geluCpu(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size);

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn geluCpu(self: *Self, x: []f32) void {
        var timer = time.Timer.start() catch null;

        const sqrt_2_over_pi: f32 = 0.7978845608;
        for (x) |*val| {
            const v = val.*;
            const v3 = v * v * v;
            const inner = sqrt_2_over_pi * (v + 0.044715 * v3);
            val.* = 0.5 * v * (1.0 + std.math.tanh(inner));
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    /// Softmax activation over a vector.
    pub fn softmax(self: *Self, x: []f32) void {
        if (self.gpu_available and x.len > GPU_THRESHOLD) {
            self.softmaxGpu(x) catch {
                self.softmaxCpu(x);
            };
        } else {
            self.softmaxCpu(x);
        }
    }

    fn softmaxGpu(self: *Self, x: []f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = x.len * @sizeOf(f32);
            var x_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer x_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size);
            try kernels.softmax(@intFromPtr(x_dev.ptr.?), @intCast(x.len), null);
            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size);

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn softmaxCpu(self: *Self, x: []f32) void {
        var timer = time.Timer.start() catch null;

        // Find max for numerical stability
        var max_val: f32 = x[0];
        for (x[1..]) |val| {
            if (val > max_val) max_val = val;
        }

        // Exp and sum
        var exp_sum: f32 = 0;
        for (x) |*val| {
            val.* = @exp(val.* - max_val);
            exp_sum += val.*;
        }

        // Normalize
        for (x) |*val| {
            val.* /= exp_sum;
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    // ========================================================================
    // Element-wise Operations
    // ========================================================================

    /// Element-wise multiplication: a = a * b
    pub fn elementwiseMul(self: *Self, a: []f32, b: []const f32) void {
        if (self.gpu_available and a.len > GPU_THRESHOLD) {
            self.elementwiseMulGpu(a, b) catch {
                self.elementwiseMulCpu(a, b);
            };
        } else {
            self.elementwiseMulCpu(a, b);
        }
    }

    fn elementwiseMulGpu(self: *Self, a: []f32, b: []const f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = a.len * @sizeOf(f32);
            var a_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer a_dev.deinit();
            var b_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, b.len * @sizeOf(f32));
            defer b_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), size);
            try cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b.len * @sizeOf(f32));

            try kernels.elementwiseMul(
                @intFromPtr(a_dev.ptr.?),
                @intFromPtr(b_dev.ptr.?),
                @intCast(a.len),
                null,
            );

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(a.ptr), a_dev.ptr.?, size);
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn elementwiseMulCpu(self: *Self, a: []f32, b: []const f32) void {
        var timer = time.Timer.start() catch null;

        for (a, b) |*av, bv| {
            av.* *= bv;
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    /// Element-wise addition: a = a + b
    pub fn elementwiseAdd(self: *Self, a: []f32, b: []const f32) void {
        if (self.gpu_available and a.len > GPU_THRESHOLD) {
            self.elementwiseAddGpu(a, b) catch {
                self.elementwiseAddCpu(a, b);
            };
        } else {
            self.elementwiseAddCpu(a, b);
        }
    }

    fn elementwiseAddGpu(self: *Self, a: []f32, b: []const f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = a.len * @sizeOf(f32);
            var a_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer a_dev.deinit();
            var b_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, b.len * @sizeOf(f32));
            defer b_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), size);
            try cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b.len * @sizeOf(f32));

            try kernels.elementwiseAdd(
                @intFromPtr(a_dev.ptr.?),
                @intFromPtr(b_dev.ptr.?),
                @intCast(a.len),
                null,
            );

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(a.ptr), a_dev.ptr.?, size);
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn elementwiseAddCpu(self: *Self, a: []f32, b: []const f32) void {
        var timer = time.Timer.start() catch null;

        for (a, b) |*av, bv| {
            av.* += bv;
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    /// Scale vector by scalar: x = x * scalar
    pub fn scale(self: *Self, x: []f32, scalar: f32) void {
        if (self.gpu_available and x.len > GPU_THRESHOLD) {
            self.scaleGpu(x, scalar) catch {
                self.scaleCpu(x, scalar);
            };
        } else {
            self.scaleCpu(x, scalar);
        }
    }

    fn scaleGpu(self: *Self, x: []f32, scalar: f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = time.Timer.start() catch null;

            const size = x.len * @sizeOf(f32);
            var x_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, size);
            defer x_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size);
            try kernels.scale(@intFromPtr(x_dev.ptr.?), scalar, @intCast(x.len), null);
            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size);

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            return error.NotAvailable;
        }
    }

    fn scaleCpu(self: *Self, x: []f32, scalar: f32) void {
        var timer = time.Timer.start() catch null;

        for (x) |*val| {
            val.* *= scalar;
        }

        self.stats.addOp(if (timer) |*t| t.read() else 0, false);
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get current GPU statistics.
    pub fn getStats(self: *const Self) GpuStats {
        return self.stats;
    }

    /// Reset statistics counters.
    pub fn resetStats(self: *Self) void {
        self.stats = .{};
    }
};

/// Check if GPU is available at runtime.
fn checkGpuAvailability() bool {
    if (!build_options.enable_gpu) return false;

    const gpu_summary = backend_mod.summary();

    if (!gpu_summary.module_enabled) return false;
    if (gpu_summary.available_backend_count == 0) return false;
    if (gpu_summary.device_count == 0) return false;

    // Prefer real hardware over emulated devices
    return gpu_summary.device_count > gpu_summary.emulated_devices;
}

/// GPU execution statistics for monitoring and optimization.
pub const GpuStats = struct {
    /// Total operations executed
    total_ops: u64 = 0,
    /// Total execution time (nanoseconds)
    total_time_ns: u64 = 0,
    /// Operations that fell back to CPU
    fallback_ops: u64 = 0,
    /// Peak GPU memory used (bytes)
    peak_memory_bytes: u64 = 0,

    pub fn addOp(self: *GpuStats, time_ns: u64, used_gpu: bool) void {
        self.total_ops += 1;
        self.total_time_ns += time_ns;
        if (!used_gpu) {
            self.fallback_ops += 1;
        }
    }

    /// Calculate GPU utilization ratio (0.0 to 1.0).
    pub fn gpuUtilization(self: GpuStats) f64 {
        if (self.total_ops == 0) return 0;
        return 1.0 - (@as(f64, @floatFromInt(self.fallback_ops)) / @as(f64, @floatFromInt(self.total_ops)));
    }

    /// Get average operation time in microseconds.
    pub fn avgOpTimeMicros(self: GpuStats) f64 {
        if (self.total_ops == 0) return 0;
        return @as(f64, @floatFromInt(self.total_time_ns)) / @as(f64, @floatFromInt(self.total_ops)) / 1000.0;
    }
};

/// Create a GPU operations context.
pub fn createContext(allocator: std.mem.Allocator) GpuOpsContext {
    return GpuOpsContext.init(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "gpu ops context init" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    // Should be able to run matmul (falls back to CPU)
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    ctx.matmul(&a, &b, &c, 2, 2, 2);

    // Verify result: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    try std.testing.expectEqual(@as(f32, 19), c[0]);
    try std.testing.expectEqual(@as(f32, 22), c[1]);
    try std.testing.expectEqual(@as(f32, 43), c[2]);
    try std.testing.expectEqual(@as(f32, 50), c[3]);
}

test "gpu stats tracking" {
    var stats = GpuStats{};

    stats.addOp(1000, true);
    stats.addOp(2000, false);
    stats.addOp(3000, true);

    try std.testing.expectEqual(@as(u64, 3), stats.total_ops);
    try std.testing.expectEqual(@as(u64, 1), stats.fallback_ops);
    try std.testing.expectApproxEqAbs(@as(f64, 0.666), stats.gpuUtilization(), 0.01);
}

test "softmax cpu" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    var x = [_]f32{ 1, 2, 3 };
    ctx.softmax(&x);

    // Sum should be 1
    const total = x[0] + x[1] + x[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.0001);
}

test "relu activation" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    var x = [_]f32{ -2, -1, 0, 1, 2 };
    ctx.relu(&x);

    try std.testing.expectEqual(@as(f32, 0), x[0]);
    try std.testing.expectEqual(@as(f32, 0), x[1]);
    try std.testing.expectEqual(@as(f32, 0), x[2]);
    try std.testing.expectEqual(@as(f32, 1), x[3]);
    try std.testing.expectEqual(@as(f32, 2), x[4]);
}

test "gelu activation" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    var x = [_]f32{ 0, 1, -1 };
    ctx.gelu(&x);

    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), x[0], 0.0001);
    // GELU(1) approx 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), x[1], 0.01);
    // GELU(-1) approx -0.159
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), x[2], 0.01);
}

test "layer norm cpu" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    var x = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 1, 1 };
    const beta = [_]f32{ 0, 0, 0, 0 };

    ctx.layerNorm(&x, &gamma, &beta, 1e-5);

    // After layer norm, mean should be ~0
    const mean_val = (x[0] + x[1] + x[2] + x[3]) / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0), mean_val, 0.0001);
}
