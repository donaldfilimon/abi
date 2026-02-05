//! CUDA Implementation of AiOps Interface
//!
//! Implements the AiOps interface for CUDA backend using:
//! - cuBLAS for BLAS operations (sgemm, sgemmStridedBatched)
//! - LLM kernels for activations (softmax, rmsnorm, silu, gelu)
//! - CUDA memory API for device memory management
//!
//! This consolidates AI GPU operations into a single unified interface.

const std = @import("std");
const ai_ops = @import("../../ai_ops.zig");
const cublas_mod = @import("cublas.zig");
const memory_mod = @import("memory.zig");
const llm_kernels = @import("llm_kernels.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;

/// CUDA-backed implementation of AI operations.
pub const CudaAiOps = struct {
    allocator: std.mem.Allocator,
    cublas_ctx: ?cublas_mod.CublasContext,
    kernels: ?llm_kernels.LlmKernelModule,
    memory_initialized: bool,

    /// Initialize CUDA AI operations.
    /// Attempts to initialize cuBLAS and LLM kernels.
    /// Falls back gracefully if some components are unavailable.
    pub fn init(allocator: std.mem.Allocator) CudaAiOps {
        var self = CudaAiOps{
            .allocator = allocator,
            .cublas_ctx = null,
            .kernels = null,
            .memory_initialized = false,
        };

        // Try to initialize CUDA memory
        memory_mod.init() catch |err| {
            std.log.warn("CUDA memory init failed: {t}", .{err});
            return self;
        };
        self.memory_initialized = true;

        // Try to initialize cuBLAS
        if (cublas_mod.CublasContext.init()) |ctx| {
            self.cublas_ctx = ctx;
            std.log.info("cuBLAS initialized for AI operations", .{});
        } else |err| {
            std.log.warn("cuBLAS init failed: {t}", .{err});
        }

        // Try to initialize LLM kernels
        if (llm_kernels.LlmKernelModule.init(allocator)) |k| {
            self.kernels = k;
            std.log.info("CUDA LLM kernels initialized (softmax, RMSNorm, SiLU, GELU)", .{});
        } else |err| {
            std.log.warn("LLM kernels init failed: {t}", .{err});
        }

        return self;
    }

    /// Clean up resources.
    pub fn deinit(self: *CudaAiOps) void {
        if (self.kernels) |*k| {
            k.deinit();
        }
        if (self.cublas_ctx) |*c| {
            c.deinit();
        }
        if (self.memory_initialized) {
            memory_mod.deinit();
        }
        self.* = undefined;
    }

    /// Check if operations are available.
    pub fn isAvailable(self: *CudaAiOps) bool {
        return self.memory_initialized and (self.cublas_ctx != null or self.kernels != null);
    }

    // =========================================================================
    // BLAS Operations
    // =========================================================================

    pub fn sgemm(
        self: *CudaAiOps,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const anyopaque,
        lda: i32,
        b: *const anyopaque,
        ldb: i32,
        beta: f32,
        c: *anyopaque,
        ldc: i32,
    ) AiOpsError!void {
        var ctx = &(self.cublas_ctx orelse return error.NotAvailable);

        const cublas_trans_a: cublas_mod.CublasOperation = if (trans_a == .trans) .trans else .no_trans;
        const cublas_trans_b: cublas_mod.CublasOperation = if (trans_b == .trans) .trans else .no_trans;

        ctx.sgemm(
            cublas_trans_a,
            cublas_trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        ) catch return error.KernelFailed;
    }

    pub fn sgemmStridedBatched(
        self: *CudaAiOps,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const anyopaque,
        lda: i32,
        stride_a: i64,
        b: *const anyopaque,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: *anyopaque,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) AiOpsError!void {
        var ctx = &(self.cublas_ctx orelse return error.NotAvailable);

        const cublas_trans_a: cublas_mod.CublasOperation = if (trans_a == .trans) .trans else .no_trans;
        const cublas_trans_b: cublas_mod.CublasOperation = if (trans_b == .trans) .trans else .no_trans;

        ctx.sgemmStridedBatched(
            cublas_trans_a,
            cublas_trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        ) catch return error.KernelFailed;
    }

    // =========================================================================
    // Activation Operations
    // =========================================================================

    pub fn softmax(self: *CudaAiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const device_ptr: u64 = @intFromPtr(data);
        kernels.softmax(device_ptr, len, stream) catch return error.KernelFailed;
    }

    pub fn rmsnorm(
        self: *CudaAiOps,
        x: *anyopaque,
        weight: *const anyopaque,
        len: u32,
        eps: f32,
        stream: ?*anyopaque,
    ) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const x_ptr: u64 = @intFromPtr(x);
        const weight_ptr: u64 = @intFromPtr(weight);
        kernels.rmsnorm(x_ptr, weight_ptr, len, eps, stream) catch return error.KernelFailed;
    }

    pub fn silu(self: *CudaAiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const device_ptr: u64 = @intFromPtr(data);
        kernels.silu(device_ptr, len, stream) catch return error.KernelFailed;
    }

    pub fn gelu(self: *CudaAiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const device_ptr: u64 = @intFromPtr(data);
        kernels.gelu(device_ptr, len, stream) catch return error.KernelFailed;
    }

    pub fn scale(self: *CudaAiOps, data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const device_ptr: u64 = @intFromPtr(data);
        kernels.scale(device_ptr, scalar, len, stream) catch return error.KernelFailed;
    }

    pub fn elementwiseMul(
        self: *CudaAiOps,
        a: *anyopaque,
        b: *const anyopaque,
        len: u32,
        stream: ?*anyopaque,
    ) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const a_ptr: u64 = @intFromPtr(a);
        const b_ptr: u64 = @intFromPtr(b);
        kernels.elementwiseMul(a_ptr, b_ptr, len, stream) catch return error.KernelFailed;
    }

    pub fn elementwiseAdd(
        self: *CudaAiOps,
        a: *anyopaque,
        b: *const anyopaque,
        len: u32,
        stream: ?*anyopaque,
    ) AiOpsError!void {
        var kernels = &(self.kernels orelse return error.NotAvailable);
        const a_ptr: u64 = @intFromPtr(a);
        const b_ptr: u64 = @intFromPtr(b);
        kernels.elementwiseAdd(a_ptr, b_ptr, len, stream) catch return error.KernelFailed;
    }

    // =========================================================================
    // Memory Operations
    // =========================================================================

    pub fn allocDevice(self: *CudaAiOps, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
        _ = self;
        const mem = memory_mod.DeviceMemory.init(allocator, size) catch return error.OutOfMemory;

        // Create a DeviceBuffer - caller must manage cleanup via freeDevice
        // Note: DeviceBuffer.deinit won't work without ops reference; use freeDevice directly
        return DeviceBuffer{
            .ptr = mem.ptr,
            .size = size,
            .allocator = allocator,
            .ops = undefined, // Caller should use freeDevice() directly
        };
    }

    pub fn copyToDevice(self: *CudaAiOps, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
        if (!self.memory_initialized) return error.NotAvailable;
        memory_mod.memcpyHostToDevice(dst, @ptrCast(src), len) catch return error.TransferFailed;
    }

    pub fn copyFromDevice(self: *CudaAiOps, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
        if (!self.memory_initialized) return error.NotAvailable;
        memory_mod.memcpyDeviceToHost(@ptrCast(dst), @constCast(src), len) catch return error.TransferFailed;
    }

    pub fn freeDevice(self: *CudaAiOps, ptr: *anyopaque) void {
        _ = self;
        // Create a temporary DeviceMemory struct to use its deinit
        var mem = memory_mod.DeviceMemory{
            .ptr = ptr,
            .size = 0, // Size not needed for free
            .allocator = undefined, // Not used in deinit
        };
        mem.deinit();
    }
};

/// Check if CUDA AI operations are available.
pub fn isAvailable() bool {
    return cublas_mod.isAvailable() or llm_kernels.isAvailable();
}

// =============================================================================
// Tests
// =============================================================================

test "cuda ai ops init without hardware" {
    // This test should pass even without GPU hardware
    // by returning an unavailable instance
    var ops = CudaAiOps.init(std.testing.allocator);
    defer ops.deinit();

    // On systems without CUDA, isAvailable() should return false
    // On systems with CUDA, it might return true
    _ = ops.isAvailable();
}
