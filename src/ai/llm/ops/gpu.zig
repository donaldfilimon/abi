//! GPU-accelerated LLM operations.
//!
//! Provides unified GPU inference path for LLM operations with automatic
//! fallback to CPU when GPU is unavailable. Supports cuBLAS acceleration
//! for matrix operations and CUDA kernels for activation functions.

const std = @import("std");
const build_options = @import("build_options");
const matmul = @import("matmul.zig");
const attention = @import("attention.zig");
const rmsnorm = @import("rmsnorm.zig");
const activations = @import("activations.zig");

// CUDA backend imports
const cuda_mod = if (build_options.enable_gpu)
    @import("../../../gpu/backends/cuda/mod.zig")
else
    struct {
        pub const llm_kernels = struct {
            pub fn isAvailable() bool {
                return false;
            }
            pub const LlmKernelModule = struct {
                pub fn init(_: std.mem.Allocator) !@This() {
                    return error.NotAvailable;
                }
                pub fn deinit(_: *@This()) void {}
                pub fn softmax(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
                pub fn rmsnorm(_: *@This(), _: u64, _: u64, _: u32, _: f32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
                pub fn silu(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
                pub fn elementwiseMul(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
                pub fn elementwiseAdd(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
                pub fn scale(_: *@This(), _: u64, _: f32, _: u32, _: ?*anyopaque) !void {
                    return error.NotAvailable;
                }
            };
        };
        pub const memory = struct {
            pub fn init() !void {
                return error.NotAvailable;
            }
            pub const DeviceMemory = struct {
                ptr: ?*anyopaque,
                size: usize,
                allocator: std.mem.Allocator,
                pub fn init(_: std.mem.Allocator, _: usize) !@This() {
                    return error.NotAvailable;
                }
                pub fn deinit(_: *@This()) void {}
            };
            pub fn memcpyHostToDevice(_: *anyopaque, _: *const anyopaque, _: usize) !void {
                return error.NotAvailable;
            }
            pub fn memcpyDeviceToHost(_: *anyopaque, _: *anyopaque, _: usize) !void {
                return error.NotAvailable;
            }
        };
    };

// cuBLAS support
const cublas = if (build_options.enable_gpu)
    @import("../../../gpu/backends/cuda/cublas.zig")
else
    struct {
        pub fn isAvailable() bool {
            return false;
        }
        pub const CublasOperation = enum { no_trans, trans };
        pub const CublasContext = struct {
            pub fn init() !@This() {
                return error.NotAvailable;
            }
            pub fn deinit(_: *@This()) void {}
            pub fn sgemm(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: *const anyopaque,
                _: i32,
                _: f32,
                _: *anyopaque,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }
            pub fn sgemmStridedBatched(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: f32,
                _: *anyopaque,
                _: i32,
                _: i64,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }
        };
        pub fn matmulRowMajor(
            _: *CublasContext,
            _: *const anyopaque,
            _: *const anyopaque,
            _: *anyopaque,
            _: i32,
            _: i32,
            _: i32,
        ) !void {
            return error.NotAvailable;
        }
    };

// GPU backend detection
const backend_mod = if (build_options.enable_gpu)
    @import("../../../gpu/backend.zig")
else
    struct {
        pub fn summary() Summary {
            return .{
                .module_enabled = false,
                .enabled_backend_count = 0,
                .available_backend_count = 0,
                .device_count = 0,
                .emulated_devices = 0,
            };
        }

        pub const Summary = struct {
            module_enabled: bool,
            enabled_backend_count: usize,
            available_backend_count: usize,
            device_count: usize,
            emulated_devices: usize,
        };
    };

/// GPU operation context for LLM inference.
pub const GpuOpsContext = struct {
    allocator: std.mem.Allocator,
    gpu_available: bool,
    cublas_available: bool,
    kernels_available: bool,
    device_id: u32,
    /// cuBLAS context for accelerated GEMM
    cublas_ctx: ?cublas.CublasContext,
    /// LLM kernel module for activation functions
    llm_kernels: ?cuda_mod.llm_kernels.LlmKernelModule,
    /// Scratch buffer for GPU operations
    scratch_buffer: ?[]f32,
    scratch_size: usize,
    /// Statistics tracking
    stats: GpuStats,

    pub fn init(allocator: std.mem.Allocator) GpuOpsContext {
        var gpu_available = build_options.enable_gpu and checkGpuAvailability();
        const cublas_present = build_options.enable_gpu and cublas.isAvailable();
        const kernels_present = build_options.enable_gpu and cuda_mod.llm_kernels.isAvailable();

        // Try to initialize cuBLAS
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
                std.log.info("cuBLAS initialized for GPU acceleration", .{});
            }
        }

        // Try to initialize LLM CUDA kernels
        if (memory_ready and kernels_present) {
            llm_kernels = cuda_mod.llm_kernels.LlmKernelModule.init(allocator) catch null;
            kernels_available = llm_kernels != null;
            if (kernels_available) {
                std.log.info("CUDA LLM kernels initialized (softmax, RMSNorm, SiLU)", .{});
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

    pub fn deinit(self: *GpuOpsContext) void {
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
    pub fn ensureScratchBuffer(self: *GpuOpsContext, size: usize) ![]f32 {
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
    pub fn isGpuAvailable(self: *const GpuOpsContext) bool {
        return self.gpu_available;
    }

    /// Matrix multiplication with GPU acceleration.
    pub fn matrixMultiply(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available) {
            // Try GPU path
            self.gpuMatmul(a, b, c, m, k, n) catch {
                // Fallback to CPU
                matmul.matrixMultiply(a, b, c, m, k, n);
            };
        } else {
            // CPU path
            matmul.matrixMultiply(a, b, c, m, k, n);
        }
    }

    /// Batched matrix multiplication with GPU acceleration.
    pub fn batchedMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available) {
            self.gpuBatchedMatmul(a, b, c, batch, m, k, n) catch {
                // Fallback: iterate batches on CPU
                const a_stride = @as(usize, m) * k;
                const b_stride = @as(usize, k) * n;
                const c_stride = @as(usize, m) * n;

                for (0..batch) |i| {
                    const a_offset = i * a_stride;
                    const b_offset = i * b_stride;
                    const c_offset = i * c_stride;
                    matmul.matrixMultiply(
                        a[a_offset..][0..a_stride],
                        b[b_offset..][0..b_stride],
                        c[c_offset..][0..c_stride],
                        m,
                        k,
                        n,
                    );
                }
            };
        } else {
            // CPU batched matmul
            const a_stride = @as(usize, m) * k;
            const b_stride = @as(usize, k) * n;
            const c_stride = @as(usize, m) * n;

            for (0..batch) |i| {
                const a_offset = i * a_stride;
                const b_offset = i * b_stride;
                const c_offset = i * c_stride;
                matmul.matrixMultiply(
                    a[a_offset..][0..a_stride],
                    b[b_offset..][0..b_stride],
                    c[c_offset..][0..c_stride],
                    m,
                    k,
                    n,
                );
            }
        }
    }

    /// Multi-head attention with GPU acceleration.
    pub fn multiHeadAttention(
        self: *GpuOpsContext,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        output: []f32,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        kv_len: u32,
    ) !void {
        if (self.gpu_available) {
            self.gpuAttention(q, k_cache, v_cache, output, seq_len, n_heads, head_dim, kv_len) catch {
                // Fallback to CPU attention
                try attention.multiHeadAttention(
                    self.allocator,
                    q,
                    k_cache,
                    v_cache,
                    output,
                    seq_len,
                    n_heads,
                    head_dim,
                    kv_len,
                );
            };
        } else {
            try attention.multiHeadAttention(
                self.allocator,
                q,
                k_cache,
                v_cache,
                output,
                seq_len,
                n_heads,
                head_dim,
                kv_len,
            );
        }
    }

    /// RMS normalization with GPU acceleration.
    pub fn rmsNorm(
        self: *GpuOpsContext,
        x: []f32,
        weight: []const f32,
        eps: f32,
    ) void {
        if (self.gpu_available) {
            self.gpuRmsNorm(x, weight, eps) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
            };
        } else {
            rmsnorm.rmsNormInPlace(x, weight, eps);
        }
    }

    /// Softmax with GPU acceleration.
    pub fn softmax(self: *GpuOpsContext, x: []f32) void {
        if (self.gpu_available) {
            self.gpuSoftmax(x) catch {
                activations.softmaxInPlace(x);
            };
        } else {
            activations.softmaxInPlace(x);
        }
    }

    /// SiLU activation with GPU acceleration.
    pub fn silu(self: *GpuOpsContext, x: []f32) void {
        if (self.gpu_available) {
            self.gpuSilu(x) catch {
                activations.siluInPlace(x);
            };
        } else {
            activations.siluInPlace(x);
        }
    }

    /// Element-wise multiply with GPU acceleration.
    pub fn elementwiseMul(self: *GpuOpsContext, a: []f32, b: []const f32) void {
        if (self.gpu_available) {
            self.gpuElementwiseMul(a, b) catch {
                for (a, b) |*av, bv| {
                    av.* *= bv;
                }
            };
        } else {
            for (a, b) |*av, bv| {
                av.* *= bv;
            }
        }
    }

    /// Vector add with GPU acceleration.
    pub fn vectorAdd(self: *GpuOpsContext, a: []f32, b: []const f32) void {
        if (self.gpu_available) {
            self.gpuVectorAdd(a, b) catch {
                for (a, b) |*av, bv| {
                    av.* += bv;
                }
            };
        } else {
            for (a, b) |*av, bv| {
                av.* += bv;
            }
        }
    }

    // GPU implementation using cuBLAS when available
    fn gpuMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        var timer = std.time.Timer.start() catch null;

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

            // Use cuBLAS SGEMM for GPU acceleration
            // Note: cuBLAS uses column-major, we use row-major
            // C = A @ B in row-major = C^T = B^T @ A^T in col-major
            // So we call sgemm(N, N, n, m, k, B, A, C) to get row-major result
            ctx.sgemm(
                .no_trans,
                .no_trans,
                @intCast(n), // rows of B^T
                @intCast(m), // cols of A^T
                @intCast(k), // inner dim
                1.0, // alpha
                @ptrCast(b_dev.ptr.?),
                @intCast(n), // ldb
                @ptrCast(a_dev.ptr.?),
                @intCast(k), // lda
                0.0, // beta
                @ptrCast(c_dev.ptr.?),
                @intCast(n), // ldc
            ) catch {
                // Fallback to CPU
                matmul.matrixMultiply(a, b, c, m, k, n);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(c.ptr), c_dev.ptr.?, c_size) catch {
                matmul.matrixMultiply(a, b, c, m, k, n);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            // CPU fallback
            matmul.matrixMultiply(a, b, c, m, k, n);
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }

    fn gpuBatchedMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        var timer = std.time.Timer.start() catch null;

        if (self.cublas_ctx) |*ctx| {
            // Use cuBLAS strided batched SGEMM
            const stride_a: i64 = @as(i64, m) * k;
            const stride_b: i64 = @as(i64, k) * n;
            const stride_c: i64 = @as(i64, m) * n;

            const batch_usize = @as(usize, batch);
            const a_size = batch_usize * @as(usize, @intCast(stride_a)) * @sizeOf(f32);
            const b_size = batch_usize * @as(usize, @intCast(stride_b)) * @sizeOf(f32);
            const c_size = batch_usize * @as(usize, @intCast(stride_c)) * @sizeOf(f32);

            var a_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, a_size);
            defer a_dev.deinit();
            var b_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, b_size);
            defer b_dev.deinit();
            var c_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, c_size);
            defer c_dev.deinit();

            try cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), a_size);
            try cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b_size);

            ctx.sgemmStridedBatched(
                .no_trans,
                .no_trans,
                @intCast(n),
                @intCast(m),
                @intCast(k),
                1.0,
                @ptrCast(b_dev.ptr.?),
                @intCast(n),
                stride_b,
                @ptrCast(a_dev.ptr.?),
                @intCast(k),
                stride_a,
                0.0,
                @ptrCast(c_dev.ptr.?),
                @intCast(n),
                stride_c,
                @intCast(batch),
            ) catch {
                // Fallback to iterative CPU
                const a_stride = @as(usize, m) * k;
                const b_stride = @as(usize, k) * n;
                const c_stride = @as(usize, m) * n;

                for (0..batch) |i| {
                    const a_offset = i * a_stride;
                    const b_offset = i * b_stride;
                    const c_offset = i * c_stride;
                    matmul.matrixMultiply(
                        a[a_offset..][0..a_stride],
                        b[b_offset..][0..b_stride],
                        c[c_offset..][0..c_stride],
                        m,
                        k,
                        n,
                    );
                }
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(c.ptr), c_dev.ptr.?, c_size) catch {
                const a_stride = @as(usize, m) * k;
                const b_stride = @as(usize, k) * n;
                const c_stride = @as(usize, m) * n;

                for (0..batch) |i| {
                    const a_offset = i * a_stride;
                    const b_offset = i * b_stride;
                    const c_offset = i * c_stride;
                    matmul.matrixMultiply(
                        a[a_offset..][0..a_stride],
                        b[b_offset..][0..b_stride],
                        c[c_offset..][0..c_stride],
                        m,
                        k,
                        n,
                    );
                }
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            // CPU batched matmul
            const a_stride = @as(usize, m) * k;
            const b_stride = @as(usize, k) * n;
            const c_stride = @as(usize, m) * n;

            for (0..batch) |i| {
                const a_offset = i * a_stride;
                const b_offset = i * b_stride;
                const c_offset = i * c_stride;
                matmul.matrixMultiply(
                    a[a_offset..][0..a_stride],
                    b[b_offset..][0..b_stride],
                    c[c_offset..][0..c_stride],
                    m,
                    k,
                    n,
                );
            }
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }

    fn gpuAttention(
        self: *GpuOpsContext,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        output: []f32,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        kv_len: u32,
    ) !void {
        // GPU attention is currently optimized for single-token decode.
        if (seq_len != 1) return error.NotAvailable;
        if (kv_len == 0 or n_heads == 0 or head_dim == 0) {
            @memset(output, 0);
            return;
        }

        const kernels = if (self.llm_kernels) |*value| value else return error.NotAvailable;
        const ctx = if (self.cublas_ctx) |*value| value else return error.NotAvailable;

        const head_dim_usize = @as(usize, head_dim);
        const kv_len_usize = @as(usize, kv_len);
        const n_heads_usize = @as(usize, n_heads);
        const kv_stride = n_heads_usize * head_dim_usize;
        const total_heads = n_heads_usize * head_dim_usize;

        if (q.len < total_heads or output.len < total_heads) return error.NotAvailable;
        if (k_cache.len < kv_len_usize * kv_stride or v_cache.len < kv_len_usize * kv_stride) {
            return error.NotAvailable;
        }

        const kv_head_len = kv_len_usize * head_dim_usize;
        const head_bytes = head_dim_usize * @sizeOf(f32);
        const kv_head_bytes = kv_head_len * @sizeOf(f32);
        const scores_bytes = kv_len_usize * @sizeOf(f32);

        var k_head = try self.allocator.alloc(f32, kv_head_len);
        defer self.allocator.free(k_head);
        var v_head = try self.allocator.alloc(f32, kv_head_len);
        defer self.allocator.free(v_head);
        var k_head_t = try self.allocator.alloc(f32, kv_head_len);
        defer self.allocator.free(k_head_t);
        const out_head = try self.allocator.alloc(f32, head_dim_usize);
        defer self.allocator.free(out_head);

        var q_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, head_bytes);
        defer q_dev.deinit();
        var k_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, kv_head_bytes);
        defer k_dev.deinit();
        var v_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, kv_head_bytes);
        defer v_dev.deinit();
        var scores_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, scores_bytes);
        defer scores_dev.deinit();
        var out_dev = try cuda_mod.memory.DeviceMemory.init(self.allocator, head_bytes);
        defer out_dev.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..n_heads_usize) |h| {
            const q_offset = h * head_dim_usize;
            const q_head = q[q_offset .. q_offset + head_dim_usize];

            for (0..kv_len_usize) |i| {
                const src_offset = i * kv_stride + q_offset;
                const dst_offset = i * head_dim_usize;
                @memcpy(
                    k_head[dst_offset .. dst_offset + head_dim_usize],
                    k_cache[src_offset .. src_offset + head_dim_usize],
                );
                @memcpy(
                    v_head[dst_offset .. dst_offset + head_dim_usize],
                    v_cache[src_offset .. src_offset + head_dim_usize],
                );
            }

            for (0..kv_len_usize) |i| {
                for (0..head_dim_usize) |j| {
                    k_head_t[j * kv_len_usize + i] =
                        k_head[i * head_dim_usize + j];
                }
            }

            try cuda_mod.memory.memcpyHostToDevice(q_dev.ptr.?, @ptrCast(q_head.ptr), head_bytes);
            try cuda_mod.memory.memcpyHostToDevice(k_dev.ptr.?, @ptrCast(k_head_t.ptr), kv_head_bytes);
            try cuda_mod.memory.memcpyHostToDevice(v_dev.ptr.?, @ptrCast(v_head.ptr), kv_head_bytes);

            try cublas.matmulRowMajor(
                ctx,
                @ptrCast(q_dev.ptr.?),
                @ptrCast(k_dev.ptr.?),
                @ptrCast(scores_dev.ptr.?),
                1,
                @intCast(head_dim),
                @intCast(kv_len),
            );

            try kernels.scale(@intFromPtr(scores_dev.ptr.?), scale, @intCast(kv_len), null);
            try kernels.softmax(@intFromPtr(scores_dev.ptr.?), @intCast(kv_len), null);

            try cublas.matmulRowMajor(
                ctx,
                @ptrCast(scores_dev.ptr.?),
                @ptrCast(v_dev.ptr.?),
                @ptrCast(out_dev.ptr.?),
                1,
                @intCast(kv_len),
                @intCast(head_dim),
            );

            try cuda_mod.memory.memcpyDeviceToHost(@ptrCast(out_head.ptr), out_dev.ptr.?, head_bytes);
            @memcpy(output[q_offset .. q_offset + head_dim_usize], out_head);
        }
    }

    fn gpuRmsNorm(
        self: *GpuOpsContext,
        x: []f32,
        weight: []const f32,
        eps: f32,
    ) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = std.time.Timer.start() catch null;

            // Allocate device memory
            const size = x.len * @sizeOf(f32);
            var x_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, size) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer x_dev.deinit();

            var weight_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, weight.len * @sizeOf(f32)) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer weight_dev.deinit();

            // Copy data to device
            cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            cuda_mod.memory.memcpyHostToDevice(weight_dev.ptr.?, @ptrCast(weight.ptr), weight.len * @sizeOf(f32)) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Run kernel
            kernels.rmsnorm(
                @intFromPtr(x_dev.ptr.?),
                @intFromPtr(weight_dev.ptr.?),
                @intCast(x.len),
                eps,
                null,
            ) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Copy result back
            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            rmsnorm.rmsNormInPlace(x, weight, eps);
        }
    }

    fn gpuSoftmax(self: *GpuOpsContext, x: []f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = std.time.Timer.start() catch null;

            // Allocate device memory
            const size = x.len * @sizeOf(f32);
            var x_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, size) catch {
                activations.softmaxInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer x_dev.deinit();

            // Copy to device
            cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size) catch {
                activations.softmaxInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Run kernel
            kernels.softmax(@intFromPtr(x_dev.ptr.?), @intCast(x.len), null) catch {
                activations.softmaxInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Copy back
            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size) catch {
                activations.softmaxInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            activations.softmaxInPlace(x);
        }
    }

    fn gpuSilu(self: *GpuOpsContext, x: []f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = std.time.Timer.start() catch null;

            // Allocate device memory
            const size = x.len * @sizeOf(f32);
            var x_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, size) catch {
                activations.siluInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer x_dev.deinit();

            // Copy to device
            cuda_mod.memory.memcpyHostToDevice(x_dev.ptr.?, @ptrCast(x.ptr), size) catch {
                activations.siluInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Run kernel
            kernels.silu(@intFromPtr(x_dev.ptr.?), @intCast(x.len), null) catch {
                activations.siluInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Copy back
            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(x.ptr), x_dev.ptr.?, size) catch {
                activations.siluInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            activations.siluInPlace(x);
        }
    }

    fn gpuElementwiseMul(self: *GpuOpsContext, a: []f32, b: []const f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = std.time.Timer.start() catch null;

            // Allocate device memory
            const size = a.len * @sizeOf(f32);
            var a_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, size) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer a_dev.deinit();

            var b_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, b.len * @sizeOf(f32)) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer b_dev.deinit();

            // Copy to device
            cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), size) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b.len * @sizeOf(f32)) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Run kernel
            kernels.elementwiseMul(
                @intFromPtr(a_dev.ptr.?),
                @intFromPtr(b_dev.ptr.?),
                @intCast(a.len),
                null,
            ) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Copy back
            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(a.ptr), a_dev.ptr.?, size) catch {
                for (a, b) |*av, bv| av.* *= bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            for (a, b) |*av, bv| {
                av.* *= bv;
            }
        }
    }

    fn gpuVectorAdd(self: *GpuOpsContext, a: []f32, b: []const f32) !void {
        if (self.llm_kernels) |*kernels| {
            var timer = std.time.Timer.start() catch null;

            // Allocate device memory
            const size = a.len * @sizeOf(f32);
            var a_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, size) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer a_dev.deinit();

            var b_dev = cuda_mod.memory.DeviceMemory.init(self.allocator, b.len * @sizeOf(f32)) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            defer b_dev.deinit();

            // Copy to device
            cuda_mod.memory.memcpyHostToDevice(a_dev.ptr.?, @ptrCast(a.ptr), size) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            cuda_mod.memory.memcpyHostToDevice(b_dev.ptr.?, @ptrCast(b.ptr), b.len * @sizeOf(f32)) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Run kernel
            kernels.elementwiseAdd(
                @intFromPtr(a_dev.ptr.?),
                @intFromPtr(b_dev.ptr.?),
                @intCast(a.len),
                null,
            ) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            // Copy back
            cuda_mod.memory.memcpyDeviceToHost(@ptrCast(a.ptr), a_dev.ptr.?, size) catch {
                for (a, b) |*av, bv| av.* += bv;
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };

            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            for (a, b) |*av, bv| {
                av.* += bv;
            }
        }
    }
};

/// Check if GPU is available at runtime.
/// Uses the GPU backend detection infrastructure to probe for real devices.
fn checkGpuAvailability() bool {
    if (!build_options.enable_gpu) return false;

    // Use the backend summary to check for available devices
    const gpu_summary = backend_mod.summary();

    // GPU is available if:
    // 1. The module is enabled
    // 2. At least one backend is available
    // 3. There are real (non-emulated) devices available
    if (!gpu_summary.module_enabled) return false;
    if (gpu_summary.available_backend_count == 0) return false;
    if (gpu_summary.device_count == 0) return false;

    // Prefer real hardware over emulated devices
    // Return true if there are any non-emulated devices
    return gpu_summary.device_count > gpu_summary.emulated_devices;
}

/// GPU execution statistics.
pub const GpuStats = struct {
    /// Total GPU operations executed
    total_ops: u64 = 0,
    /// Total GPU time (nanoseconds)
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

    pub fn gpuUtilization(self: GpuStats) f64 {
        if (self.total_ops == 0) return 0;
        return 1.0 - (@as(f64, @floatFromInt(self.fallback_ops)) / @as(f64, @floatFromInt(self.total_ops)));
    }
};

/// Create a GPU operations context for LLM inference.
pub fn createContext(allocator: std.mem.Allocator) GpuOpsContext {
    return GpuOpsContext.init(allocator);
}

test "gpu ops context init" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    // Should be able to run matmul (falls back to CPU)
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    ctx.matrixMultiply(&a, &b, &c, 2, 2, 2);
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
