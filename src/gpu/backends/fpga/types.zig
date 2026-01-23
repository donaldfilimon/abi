//! FPGA Backend Types
//!
//! Core type definitions for the FPGA backend, providing abstractions for
//! FPGA-specific concepts like compute units, memory banks, and kernel execution.
//!
//! ## Overview
//!
//! FPGAs differ from GPUs in several key ways:
//! - **Pre-compiled kernels**: Operations are defined in bitstreams, not runtime-compiled
//! - **Memory banks**: DDR/HBM organized in banks with explicit allocation
//! - **Compute units**: Fixed-function accelerators instantiated in fabric
//! - **Deterministic latency**: No warp scheduling or memory coalescing concerns
//!
//! ## Usage
//!
//! ```zig
//! const fpga_types = @import("types.zig");
//!
//! // Describe an FPGA device
//! var caps = fpga_types.DeviceCapabilities{
//!     .platform = .xilinx,
//!     .device_family = .alveo,
//!     .lut_count = 1_700_000,
//!     .dsp_count = 12_288,
//! };
//! ```

const std = @import("std");

// ============================================================================
// Platform and Device Types
// ============================================================================

/// FPGA vendor platform
pub const Platform = enum(u8) {
    /// Auto-detect available platform
    auto = 0,
    /// AMD/Xilinx devices (Alveo, Versal, etc.)
    xilinx = 1,
    /// Intel/Altera devices (Agilex, Stratix, etc.)
    intel = 2,
    /// Lattice devices (for edge applications)
    lattice = 3,
    /// Microchip/Microsemi devices
    microchip = 4,
    /// Unknown/unsupported platform
    unknown = 255,

    pub fn name(self: Platform) []const u8 {
        return switch (self) {
            .auto => "Auto-detect",
            .xilinx => "AMD/Xilinx",
            .intel => "Intel",
            .lattice => "Lattice",
            .microchip => "Microchip",
            .unknown => "Unknown",
        };
    }

    pub fn supportsHls(self: Platform) bool {
        return switch (self) {
            .xilinx, .intel => true,
            else => false,
        };
    }
};

/// FPGA device family (for capability detection)
pub const DeviceFamily = enum(u8) {
    // AMD/Xilinx families
    alveo = 1, // Data center accelerator cards
    versal = 2, // AI Core/AI Edge with AI Engines
    kintex = 3, // Mid-range FPGAs
    virtex = 4, // High-performance FPGAs
    artix = 5, // Cost-optimized FPGAs

    // Intel families
    agilex = 10, // High-performance with HBM
    stratix = 11, // High-performance
    arria = 12, // Mid-range
    cyclone = 13, // Cost-optimized

    // Other
    generic = 255,

    pub fn hasHbm(self: DeviceFamily) bool {
        return switch (self) {
            .alveo, .versal, .agilex => true,
            else => false,
        };
    }

    pub fn hasAiEngines(self: DeviceFamily) bool {
        return self == .versal;
    }
};

/// Detailed device capabilities
pub const DeviceCapabilities = struct {
    /// Vendor platform
    platform: Platform = .unknown,
    /// Device family
    device_family: DeviceFamily = .generic,

    // Fabric resources
    lut_count: u32 = 0, // Look-Up Tables
    dsp_count: u32 = 0, // DSP slices (multiply-accumulate)
    bram_kb: u32 = 0, // Block RAM in KB
    uram_kb: u32 = 0, // UltraRAM in KB (Xilinx)
    plram_kb: u32 = 0, // PL RAM / on-chip memory

    // Memory interfaces
    ddr_banks: u8 = 0, // Number of DDR memory banks
    ddr_size_gb: u16 = 0, // Total DDR in GB
    hbm_stacks: u8 = 0, // Number of HBM stacks
    hbm_size_gb: u16 = 0, // Total HBM in GB
    hbm_bandwidth_gbps: u16 = 0, // HBM bandwidth in GB/s

    // Compute units
    max_compute_units: u16 = 0, // Max instantiated kernels
    max_clock_mhz: u16 = 0, // Maximum kernel clock

    // Host interface
    pcie_gen: u8 = 0, // PCIe generation (3, 4, 5)
    pcie_lanes: u8 = 0, // PCIe lane count

    // Features
    features: FeatureFlags = .{},

    pub const FeatureFlags = packed struct {
        fp16: bool = false,
        fp64: bool = false,
        int4: bool = false,
        int8: bool = false,
        ai_engines: bool = false,
        dynamic_reconfig: bool = false,
        streaming_dma: bool = false,
        _padding: u1 = 0,
    };

    /// Estimate peak INT8 TOPS based on DSP count
    pub fn estimateInt8Tops(self: DeviceCapabilities) f32 {
        // DSPs typically do 2 INT8 MACs per cycle at ~300 MHz
        const macs_per_cycle: f32 = 2.0;
        const clock_ghz: f32 = @as(f32, @floatFromInt(self.max_clock_mhz)) / 1000.0;
        return @as(f32, @floatFromInt(self.dsp_count)) * macs_per_cycle * clock_ghz * 2.0 / 1000.0;
    }

    /// Estimate on-chip storage capacity
    pub fn totalOnChipKb(self: DeviceCapabilities) u32 {
        return self.bram_kb + self.uram_kb + self.plram_kb;
    }

    /// Check if device has enough resources for a given configuration
    pub fn meetsRequirements(self: DeviceCapabilities, req: ResourceRequirements) bool {
        return self.lut_count >= req.min_luts and
            self.dsp_count >= req.min_dsps and
            self.totalOnChipKb() >= req.min_sram_kb and
            (self.ddr_size_gb + self.hbm_size_gb) >= req.min_memory_gb;
    }
};

/// Resource requirements for a kernel or application
pub const ResourceRequirements = struct {
    min_luts: u32 = 0,
    min_dsps: u32 = 0,
    min_sram_kb: u32 = 0,
    min_memory_gb: u16 = 0,
    requires_hbm: bool = false,
    requires_ai_engines: bool = false,
};

// ============================================================================
// Memory Types
// ============================================================================

/// Memory bank identifier
pub const MemoryBank = enum(u8) {
    // DDR banks
    ddr0 = 0,
    ddr1 = 1,
    ddr2 = 2,
    ddr3 = 3,
    ddr4 = 4,
    ddr5 = 5,
    ddr6 = 6,
    ddr7 = 7,

    // HBM pseudo-channels (up to 32)
    hbm0 = 32,
    hbm1 = 33,
    hbm2 = 34,
    hbm3 = 35,
    hbm4 = 36,
    hbm5 = 37,
    hbm6 = 38,
    hbm7 = 39,
    // ... additional HBM channels follow pattern

    // On-chip memory
    plram0 = 64,
    plram1 = 65,
    plram2 = 66,
    plram3 = 67,

    // Special
    auto = 254, // Auto-select best bank
    host = 255, // Host memory (pinned)

    pub fn isHbm(self: MemoryBank) bool {
        const val = @intFromEnum(self);
        return val >= 32 and val < 64;
    }

    pub fn isDdr(self: MemoryBank) bool {
        const val = @intFromEnum(self);
        return val < 32;
    }

    pub fn isOnChip(self: MemoryBank) bool {
        const val = @intFromEnum(self);
        return val >= 64 and val < 254;
    }

    pub fn name(self: MemoryBank) []const u8 {
        return switch (self) {
            .ddr0 => "DDR[0]",
            .ddr1 => "DDR[1]",
            .ddr2 => "DDR[2]",
            .ddr3 => "DDR[3]",
            .hbm0 => "HBM[0]",
            .hbm1 => "HBM[1]",
            .plram0 => "PLRAM[0]",
            .auto => "AUTO",
            .host => "HOST",
            else => "BANK",
        };
    }
};

/// Memory allocation flags
pub const MemoryFlags = packed struct {
    /// Allocate in device memory
    device: bool = true,
    /// Host can read/write directly
    host_visible: bool = false,
    /// No explicit synchronization needed
    host_coherent: bool = false,
    /// Enable caching
    cached: bool = true,
    /// Read-only on device
    read_only: bool = false,
    /// Write-only on device
    write_only: bool = false,
    /// Use for streaming access pattern
    streaming: bool = false,
    /// Prefer high-bandwidth memory (HBM)
    high_bandwidth: bool = false,
};

/// Memory buffer descriptor
pub const BufferDescriptor = struct {
    /// Unique buffer identifier
    id: u64,
    /// Size in bytes
    size: usize,
    /// Alignment requirement (typically 64 bytes for FPGA)
    alignment: usize = 64,
    /// Memory bank allocation
    bank: MemoryBank,
    /// Allocation flags
    flags: MemoryFlags,
    /// Device pointer (opaque)
    device_ptr: ?*anyopaque = null,
    /// Host pointer (if mapped)
    host_ptr: ?[*]u8 = null,
    /// Current state
    state: BufferState = .unallocated,

    pub const BufferState = enum {
        unallocated,
        allocated,
        mapped,
        in_use,
        freed,
    };

    pub fn isValid(self: BufferDescriptor) bool {
        return self.state != .unallocated and self.state != .freed and self.device_ptr != null;
    }
};

// ============================================================================
// Kernel Types
// ============================================================================

/// Kernel type classification for FPGA-optimized operations
pub const KernelClass = enum(u8) {
    // Vector operations
    vector_distance = 1,
    vector_add = 2,
    vector_scale = 3,
    vector_normalize = 4,

    // Matrix operations
    matmul_fp32 = 10,
    matmul_fp16 = 11,
    matmul_int8 = 12,
    matmul_int4 = 13,
    matmul_q4_0 = 14,
    matmul_q4_1 = 15,
    matmul_q8_0 = 16,

    // Attention operations
    attention_qkv = 20,
    attention_softmax = 21,
    attention_output = 22,
    flash_attention = 23,

    // Normalization
    rmsnorm = 30,
    layernorm = 31,
    batchnorm = 32,

    // Activations
    silu = 40,
    gelu = 41,
    relu = 42,
    softmax = 43,

    // Embeddings
    rope = 50,
    alibi = 51,

    // Vector database
    hnsw_search = 60,
    ivf_search = 61,
    pq_encode = 62,
    pq_decode = 63,
    kmeans_assign = 64,

    // Reductions
    reduce_sum = 70,
    reduce_max = 71,
    reduce_mean = 72,

    // Custom/user-defined
    custom = 255,

    pub fn isMatmul(self: KernelClass) bool {
        const val = @intFromEnum(self);
        return val >= 10 and val < 20;
    }

    pub fn isAttention(self: KernelClass) bool {
        const val = @intFromEnum(self);
        return val >= 20 and val < 30;
    }

    pub fn isVectorDb(self: KernelClass) bool {
        const val = @intFromEnum(self);
        return val >= 60 and val < 70;
    }
};

/// Kernel execution configuration
pub const KernelConfig = struct {
    /// Kernel class/type
    kernel_class: KernelClass,
    /// Compute unit index to use (null = auto-select)
    compute_unit: ?u16 = null,
    /// Number of work items / iterations
    work_size: WorkSize = .{},
    /// Execution options
    options: ExecutionOptions = .{},

    pub const WorkSize = struct {
        global: [3]u32 = .{ 1, 1, 1 },
        local: [3]u32 = .{ 1, 1, 1 },
    };

    pub const ExecutionOptions = struct {
        /// Enable profiling for this kernel
        profile: bool = false,
        /// Timeout in microseconds (0 = no timeout)
        timeout_us: u32 = 0,
        /// Priority (higher = more urgent)
        priority: u8 = 128,
    };
};

/// Kernel execution result
pub const KernelResult = struct {
    /// Execution status
    status: Status,
    /// Execution time in nanoseconds
    execution_time_ns: u64 = 0,
    /// Compute unit that executed the kernel
    compute_unit: u16 = 0,
    /// Error message (if status is error)
    error_message: ?[]const u8 = null,

    pub const Status = enum {
        success,
        timeout,
        error,
        cancelled,
    };
};

// ============================================================================
// Bitstream Types
// ============================================================================

/// Bitstream metadata
pub const BitstreamInfo = struct {
    /// Unique identifier (UUID)
    uuid: [16]u8 = undefined,
    /// Platform this bitstream targets
    platform: Platform = .unknown,
    /// Device family
    device_family: DeviceFamily = .generic,
    /// Build timestamp (Unix epoch)
    build_timestamp: i64 = 0,
    /// Version string
    version: [32]u8 = undefined,
    version_len: usize = 0,
    /// List of kernel names
    kernel_names: []const []const u8 = &.{},
    /// Resource utilization
    resource_usage: ResourceUsage = .{},

    pub const ResourceUsage = struct {
        lut_percent: u8 = 0,
        dsp_percent: u8 = 0,
        bram_percent: u8 = 0,
        uram_percent: u8 = 0,
    };

    pub fn getVersion(self: *const BitstreamInfo) []const u8 {
        return self.version[0..self.version_len];
    }

    pub fn hasKernel(self: *const BitstreamInfo, name: []const u8) bool {
        for (self.kernel_names) |kernel_name| {
            if (std.mem.eql(u8, kernel_name, name)) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Data Types for Kernels
// ============================================================================

/// Quantization type for matrix operations
pub const QuantizationType = enum(u8) {
    fp32 = 0,
    fp16 = 1,
    bf16 = 2,
    int8 = 3,
    int4 = 4,
    q4_0 = 5, // GGML Q4_0: 4-bit with fp16 scale per 32 values
    q4_1 = 6, // GGML Q4_1: 4-bit with fp16 scale and min
    q5_0 = 7, // GGML Q5_0: 5-bit quantization
    q5_1 = 8, // GGML Q5_1: 5-bit with min
    q8_0 = 9, // GGML Q8_0: 8-bit quantization

    pub fn bitsPerValue(self: QuantizationType) u8 {
        return switch (self) {
            .fp32 => 32,
            .fp16, .bf16 => 16,
            .int8, .q8_0 => 8,
            .q5_0, .q5_1 => 5,
            .int4, .q4_0, .q4_1 => 4,
        };
    }

    pub fn blockSize(self: QuantizationType) u8 {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0 => 32,
            else => 1,
        };
    }
};

/// Distance metric for vector operations
pub const DistanceMetric = enum(u8) {
    cosine = 0,
    l2 = 1, // Euclidean
    l2_squared = 2,
    dot = 3, // Inner product
    manhattan = 4, // L1

    pub fn requiresNormalization(self: DistanceMetric) bool {
        return self == .cosine;
    }
};

// ============================================================================
// Statistics and Profiling
// ============================================================================

/// FPGA execution statistics
pub const FpgaStats = struct {
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
    /// Number of bitstream loads
    bitstream_loads: u32 = 0,

    pub fn averageKernelTimeNs(self: FpgaStats) u64 {
        if (self.kernel_invocations == 0) return 0;
        return self.total_execution_ns / self.kernel_invocations;
    }

    pub fn totalTransferBytes(self: FpgaStats) u64 {
        return self.bytes_to_device + self.bytes_from_device;
    }

    pub fn reset(self: *FpgaStats) void {
        self.* = FpgaStats{};
    }
};

// ============================================================================
// Error Types
// ============================================================================

/// FPGA-specific errors
pub const FpgaError = error{
    // Initialization errors
    PlatformNotSupported,
    DeviceNotFound,
    DriverNotInstalled,
    InitializationFailed,

    // Bitstream errors
    BitstreamLoadFailed,
    BitstreamInvalid,
    BitstreamIncompatible,
    KernelNotFound,

    // Memory errors
    OutOfMemory,
    OutOfDeviceMemory,
    AllocationFailed,
    InvalidPointer,
    TransferFailed,
    BankNotAvailable,

    // Execution errors
    KernelLaunchFailed,
    ExecutionTimeout,
    ExecutionFailed,
    InvalidConfiguration,

    // General errors
    NotInitialized,
    AlreadyInitialized,
    OperationCancelled,
    Disabled,
};

// ============================================================================
// Tests
// ============================================================================

test "device capabilities estimation" {
    const caps = DeviceCapabilities{
        .platform = .xilinx,
        .device_family = .alveo,
        .dsp_count = 12_288,
        .max_clock_mhz = 300,
        .bram_kb = 54 * 1024,
        .uram_kb = 90 * 1024,
    };

    // Estimate ~14 INT8 TOPS for Alveo U250
    const tops = caps.estimateInt8Tops();
    try std.testing.expect(tops > 10.0 and tops < 20.0);

    // Total on-chip: ~144 MB
    const sram = caps.totalOnChipKb();
    try std.testing.expect(sram > 100_000);
}

test "memory bank classification" {
    try std.testing.expect(MemoryBank.ddr0.isDdr());
    try std.testing.expect(!MemoryBank.ddr0.isHbm());

    try std.testing.expect(MemoryBank.hbm0.isHbm());
    try std.testing.expect(!MemoryBank.hbm0.isDdr());

    try std.testing.expect(MemoryBank.plram0.isOnChip());
}

test "kernel class classification" {
    try std.testing.expect(KernelClass.matmul_q4_0.isMatmul());
    try std.testing.expect(KernelClass.flash_attention.isAttention());
    try std.testing.expect(KernelClass.hnsw_search.isVectorDb());
}

test "quantization bits" {
    try std.testing.expectEqual(@as(u8, 4), QuantizationType.q4_0.bitsPerValue());
    try std.testing.expectEqual(@as(u8, 8), QuantizationType.q8_0.bitsPerValue());
    try std.testing.expectEqual(@as(u8, 32), QuantizationType.q4_0.blockSize());
}
