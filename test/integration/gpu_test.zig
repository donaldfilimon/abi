//! Integration Tests: GPU Module
//!
//! Verifies GPU type availability, backend enum values, config struct
//! defaults, and basic API contracts without requiring real GPU hardware.

const std = @import("std");
const abi = @import("abi");

const gpu = abi.gpu;

// ============================================================================
// Type availability
// ============================================================================

test "gpu: Gpu type exists" {
    const GpuType = gpu.Gpu;
    _ = GpuType;
}

test "gpu: GpuConfig type exists" {
    const ConfigType = gpu.GpuConfig;
    _ = ConfigType;
}

test "gpu: UnifiedBuffer type exists" {
    const BufType = gpu.UnifiedBuffer;
    _ = BufType;
}

test "gpu: Device type exists" {
    const DevType = gpu.Device;
    _ = DevType;
}

test "gpu: DeviceType type exists" {
    const DT = gpu.DeviceType;
    _ = DT;
}

// ============================================================================
// Backend enum
// ============================================================================

test "gpu: Backend enum has expected variants" {
    const B = gpu.Backend;

    // Verify known backend values exist
    _ = B.cuda;
    _ = B.vulkan;
    _ = B.metal;
    _ = B.webgpu;
    _ = B.opengl;
    _ = B.stdgpu;
    _ = B.simulated;
}

test "gpu: Backend name returns string" {
    const B = gpu.Backend;
    const metal_name = B.metal.name();
    try std.testing.expectEqualStrings("metal", metal_name);

    const cuda_name = B.cuda.name();
    try std.testing.expectEqualStrings("cuda", cuda_name);
}

// ============================================================================
// Error types
// ============================================================================

test "gpu: GpuError type exists" {
    const E = gpu.GpuError;
    _ = E;
}

test "gpu: MemoryError type exists" {
    const E = gpu.MemoryError;
    _ = E;
}

test "gpu: KernelError type exists" {
    const E = gpu.KernelError;
    _ = E;
}

// ============================================================================
// Submodule availability
// ============================================================================

test "gpu: profiling submodule exists" {
    const P = gpu.profiling;
    _ = P;
}

test "gpu: execution_coordinator submodule exists" {
    const EC = gpu.execution_coordinator;
    _ = EC;
}

test "gpu: fusion submodule exists" {
    const F = gpu.fusion;
    _ = F;
}

test "gpu: occupancy submodule exists" {
    const O = gpu.occupancy;
    _ = O;
}

test "gpu: dsl submodule exists" {
    const D = gpu.dsl;
    _ = D;
}

test "gpu: backends submodule exists" {
    const B = gpu.backends;
    _ = B;
}

test "gpu: platform submodule exists" {
    const P = gpu.platform;
    _ = P;
}

// ============================================================================
// Buffer and stream types
// ============================================================================

test "gpu: BufferFlags type exists" {
    const BF = gpu.BufferFlags;
    _ = BF;
}

test "gpu: GpuBuffer type exists" {
    const GB = gpu.GpuBuffer;
    _ = GB;
}

test "gpu: StreamOptions type exists" {
    const SO = gpu.StreamOptions;
    _ = SO;
}

test "gpu: HealthStatus type exists" {
    const HS = gpu.HealthStatus;
    _ = HS;
}

test "gpu: ExecutionResult type exists" {
    const ER = gpu.ExecutionResult;
    _ = ER;
}

test "gpu: LaunchConfig type exists" {
    const LC = gpu.LaunchConfig;
    _ = LC;
}

test {
    std.testing.refAllDecls(@This());
}
