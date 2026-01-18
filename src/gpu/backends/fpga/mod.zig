//! FPGA Backend Module
//!
//! Provides hardware acceleration via FPGA devices for compute-intensive operations.
//! Supports AMD/Xilinx (via Vitis/XRT) and Intel (via oneAPI) FPGA platforms.
//!
//! ## Key Features
//!
//! - Bitstream loading and management
//! - DDR/HBM memory allocation and transfer
//! - Pre-compiled kernel execution
//! - Hardware-accelerated vector operations:
//!   - Vector distance (cosine, L2, dot product)
//!   - Quantized matrix multiplication (Q4, Q8)
//!   - HNSW graph traversal acceleration
//!   - K-means centroid assignment
//!
//! ## Usage
//!
//! ```zig
//! const fpga = @import("gpu/backends/fpga/mod.zig");
//!
//! var backend = try fpga.FpgaBackend.init(allocator, .{
//!     .platform = .xilinx,
//!     .device_index = 0,
//! });
//! defer backend.deinit();
//!
//! // Load pre-compiled bitstream
//! try backend.loadBitstream("kernels/vector_distance.xclbin");
//!
//! // Execute kernel
//! try backend.launchKernel("vector_distance", config, args);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");

pub const loader = @import("loader.zig");
pub const memory = @import("memory.zig");
pub const vtable = @import("vtable.zig");
pub const kernels = @import("kernels.zig");

// Re-export key types
pub const FpgaBackend = vtable.FpgaBackend;
pub const FpgaConfig = vtable.FpgaConfig;
pub const FpgaPlatform = vtable.FpgaPlatform;
pub const FpgaError = vtable.FpgaError;
pub const BitstreamHandle = loader.BitstreamHandle;
pub const FpgaMemory = memory.FpgaMemory;
pub const FpgaBuffer = memory.FpgaBuffer;

// Kernel types
pub const FpgaKernelType = kernels.FpgaKernelType;
pub const VectorDistanceConfig = kernels.VectorDistanceConfig;
pub const QuantizedMatmulConfig = kernels.QuantizedMatmulConfig;
pub const KMeansConfig = kernels.KMeansConfig;

/// Check if FPGA support is enabled at compile time
pub fn isEnabled() bool {
    return build_options.gpu_fpga;
}

/// Check if FPGA hardware is available at runtime
pub fn isAvailable() bool {
    if (!isEnabled()) return false;
    return loader.detectFpgaDevices() > 0;
}

/// Get the number of available FPGA devices
pub fn getDeviceCount() u32 {
    if (!isEnabled()) return 0;
    return loader.detectFpgaDevices();
}

/// Initialize the FPGA subsystem
pub fn init() !void {
    if (!isEnabled()) return error.FpgaDisabled;
    try loader.init();
}

/// Deinitialize the FPGA subsystem
pub fn deinit() void {
    if (isEnabled()) {
        loader.deinit();
    }
}

/// Create an FPGA backend instance that implements the GPU interface
pub fn createBackend(allocator: std.mem.Allocator, config: FpgaConfig) !interface.Backend {
    var impl = try FpgaBackend.init(allocator, config);
    return interface.createBackend(FpgaBackend, impl);
}

test "fpga module compilation" {
    // Basic compilation test
    const enabled = isEnabled();
    _ = enabled;
}
