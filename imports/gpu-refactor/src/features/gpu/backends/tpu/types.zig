//! TPU Backend Types
//!
//! Core type definitions for the TPU (Tensor Processing Unit) backend.
//! TPUs are Google's custom ASICs designed for high-throughput matrix
//! operations and ML inference workloads.
//!
//! ## Overview
//!
//! TPUs differ from GPUs in several key ways:
//! - **Systolic array architecture**: Dedicated matrix multiply units (MXUs)
//! - **High-bandwidth memory (HBM)**: Large on-chip memory for model weights
//! - **bfloat16 native**: Optimized for reduced-precision ML workloads
//! - **ICI interconnect**: High-speed chip-to-chip links for multi-chip pods

const std = @import("std");

// ============================================================================
// Device Types
// ============================================================================

/// TPU hardware generation
pub const TpuGeneration = enum(u8) {
    /// TPU v2 (2017): 180 TFLOPS bf16, 64 GB HBM
    v2 = 2,
    /// TPU v3 (2018): 420 TFLOPS bf16, 128 GB HBM
    v3 = 3,
    /// TPU v4 (2021): 275 TFLOPS bf16 per chip, 32 GB HBM per chip
    v4 = 4,
    /// TPU v5e (2023): Cost-optimized inference
    v5e = 5,
    /// TPU v5p (2023): High-performance training
    v5p = 6,
    /// Simulated / unknown generation
    simulated = 0,

    pub fn name(self: TpuGeneration) []const u8 {
        return switch (self) {
            .v2 => "TPU v2",
            .v3 => "TPU v3",
            .v4 => "TPU v4",
            .v5e => "TPU v5e",
            .v5p => "TPU v5p",
            .simulated => "TPU (simulated)",
        };
    }

    /// Estimated peak bf16 TFLOPS per chip
    pub fn peakTflopsBf16(self: TpuGeneration) u32 {
        return switch (self) {
            .v2 => 180,
            .v3 => 420,
            .v4 => 275,
            .v5e => 197,
            .v5p => 459,
            .simulated => 0,
        };
    }

    /// HBM capacity per chip in GB
    pub fn hbmCapacityGb(self: TpuGeneration) u32 {
        return switch (self) {
            .v2 => 64,
            .v3 => 128,
            .v4 => 32,
            .v5e => 16,
            .v5p => 95,
            .simulated => 0,
        };
    }
};

/// TPU device capabilities
pub const DeviceCapabilities = struct {
    /// Hardware generation
    generation: TpuGeneration = .simulated,
    /// Number of MXU (Matrix Multiply Unit) cores
    mxu_count: u16 = 0,
    /// HBM capacity in bytes
    hbm_bytes: u64 = 0,
    /// HBM bandwidth in GB/s
    hbm_bandwidth_gbps: u32 = 0,
    /// Number of chips in the pod slice
    chip_count: u16 = 1,
    /// ICI interconnect bandwidth in GB/s (0 if single chip)
    ici_bandwidth_gbps: u32 = 0,
    /// Supported data types
    supported_types: SupportedTypes = .{},

    pub const SupportedTypes = packed struct {
        bf16: bool = true,
        fp32: bool = true,
        int8: bool = true,
        int4: bool = false,
        fp16: bool = false,
        fp8: bool = false,
        _padding: u2 = 0,
    };

    /// Estimated peak INT8 TOPS
    pub fn estimateInt8Tops(self: DeviceCapabilities) f32 {
        // INT8 throughput is roughly 2x bf16 TFLOPS
        return @as(f32, @floatFromInt(self.generation.peakTflopsBf16())) * 2.0;
    }
};

// ============================================================================
// Kernel Types
// ============================================================================

/// TPU kernel classification
pub const KernelClass = enum(u8) {
    /// Matrix multiply (the TPU's primary operation)
    matmul = 1,
    /// Fused matmul + bias + activation
    matmul_fused = 2,
    /// Multi-head attention
    attention = 10,
    /// Flash attention variant
    flash_attention = 11,
    /// Layer normalization
    layernorm = 20,
    /// RMS normalization
    rmsnorm = 21,
    /// Softmax
    softmax = 30,
    /// SiLU / Swish activation
    silu = 31,
    /// GELU activation
    gelu = 32,
    /// Embedding lookup
    embedding = 40,
    /// RoPE positional encoding
    rope = 41,
    /// Vector distance (cosine, L2, dot)
    vector_distance = 50,
    /// Reduce (sum, max, mean)
    reduce = 60,
    /// Custom / user-defined
    custom = 255,

    pub fn isMatmul(self: KernelClass) bool {
        const val = @intFromEnum(self);
        return val >= 1 and val < 10;
    }

    pub fn isAttention(self: KernelClass) bool {
        const val = @intFromEnum(self);
        return val >= 10 and val < 20;
    }
};

/// TPU kernel execution configuration
pub const KernelConfig = struct {
    /// Kernel class
    kernel_class: KernelClass = .custom,
    /// Batch size for batched operations
    batch_size: u32 = 1,
    /// Sequence length (for attention/transformer ops)
    sequence_length: u32 = 0,
    /// Hidden dimension
    hidden_dim: u32 = 0,
    /// Data type for computation
    compute_type: ComputeType = .bf16,
    /// Enable profiling
    profile: bool = false,

    pub const ComputeType = enum(u8) {
        bf16 = 0,
        fp32 = 1,
        int8 = 2,
        fp16 = 3,
    };
};

// ============================================================================
// Configuration
// ============================================================================

/// TPU backend configuration
pub const TpuConfig = struct {
    /// Target TPU generation (simulated if not available)
    generation: TpuGeneration = .simulated,
    /// Number of chips to simulate
    chip_count: u16 = 1,
    /// Enable verbose logging
    verbose: bool = false,
    /// Maximum memory budget in bytes (0 = unlimited)
    memory_budget: u64 = 0,
};

// ============================================================================
// Statistics
// ============================================================================

/// TPU execution statistics
pub const TpuStats = struct {
    /// Total kernel invocations
    kernel_invocations: u64 = 0,
    /// Total execution time (ns)
    total_execution_ns: u64 = 0,
    /// Total bytes transferred to device
    bytes_to_device: u64 = 0,
    /// Total bytes transferred from device
    bytes_from_device: u64 = 0,
    /// Number of memory allocations
    allocation_count: u64 = 0,
    /// Peak memory usage (bytes)
    peak_memory_bytes: u64 = 0,
    /// Current memory usage (bytes)
    current_memory_bytes: u64 = 0,

    pub fn averageKernelTimeNs(self: TpuStats) u64 {
        if (self.kernel_invocations == 0) return 0;
        return self.total_execution_ns / self.kernel_invocations;
    }

    pub fn totalTransferBytes(self: TpuStats) u64 {
        return self.bytes_to_device + self.bytes_from_device;
    }

    pub fn reset(self: *TpuStats) void {
        self.* = TpuStats{};
    }
};

// ============================================================================
// Error Types
// ============================================================================

/// TPU-specific errors
pub const TpuError = error{
    /// TPU device not found or not available
    DeviceNotFound,
    /// Driver or runtime not installed
    DriverNotFound,
    /// Initialization failed
    InitializationFailed,
    /// Out of device memory (HBM)
    OutOfDeviceMemory,
    /// Memory allocation failed
    AllocationFailed,
    /// Kernel compilation failed
    KernelCompileFailed,
    /// Kernel execution failed
    KernelExecutionFailed,
    /// Invalid configuration
    InvalidConfiguration,
    /// Operation not supported on this TPU generation
    UnsupportedOperation,
    /// Operation timed out
    Timeout,
    /// TPU backend is in simulation mode only
    SimulationOnly,
};

// ============================================================================
// Tests
// ============================================================================

test "TPU generation properties" {
    const v4 = TpuGeneration.v4;
    try std.testing.expectEqual(@as(u32, 275), v4.peakTflopsBf16());
    try std.testing.expectEqual(@as(u32, 32), v4.hbmCapacityGb());
    try std.testing.expectEqualStrings("TPU v4", v4.name());
}

test "device capabilities estimation" {
    const caps = DeviceCapabilities{
        .generation = .v5p,
        .mxu_count = 4,
        .hbm_bytes = 95 * 1024 * 1024 * 1024,
        .hbm_bandwidth_gbps = 2765,
    };

    const tops = caps.estimateInt8Tops();
    try std.testing.expect(tops > 900.0);
}

test "kernel class classification" {
    try std.testing.expect(KernelClass.matmul.isMatmul());
    try std.testing.expect(KernelClass.matmul_fused.isMatmul());
    try std.testing.expect(!KernelClass.attention.isMatmul());
    try std.testing.expect(KernelClass.attention.isAttention());
    try std.testing.expect(KernelClass.flash_attention.isAttention());
}

test "stats tracking" {
    var stats = TpuStats{};
    stats.kernel_invocations = 10;
    stats.total_execution_ns = 1000;
    stats.bytes_to_device = 500;
    stats.bytes_from_device = 300;

    try std.testing.expectEqual(@as(u64, 100), stats.averageKernelTimeNs());
    try std.testing.expectEqual(@as(u64, 800), stats.totalTransferBytes());

    stats.reset();
    try std.testing.expectEqual(@as(u64, 0), stats.kernel_invocations);
}

test {
    std.testing.refAllDecls(@This());
}
