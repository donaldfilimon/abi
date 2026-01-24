//! FPGA Backend Module
//!
//! Provides support for FPGA-based acceleration of AI workloads.
//! Implementation follows the research in docs/research/fpga-inference-acceleration.md.
//! Currently supports simulation mode for development and testing.

const std = @import("std");

pub const vtable = @import("vtable.zig");
pub const kernels = @import("kernels.zig");

var initialized = false;

pub fn init() !void {
    if (initialized) return;

    // FPGA initialization
    // In real implementation, this would:
    // 1. Load XRT (Xilinx Runtime) or OpenCL libraries
    // 2. Query FPGA devices via platform/device APIs
    // 3. Initialize context and command queue

    // For now, mark as initialized even in simulation mode
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isAvailable() bool {
    // FPGA hardware detection strategy:
    // 1. Check for XRT (Xilinx) via xrt::device enumeration
    // 2. Check for OpenCL FPGA platforms via clGetPlatformIDs
    // 3. Check for Intel FPGA via oneAPI Level Zero
    // 4. Check simulation/test mode via environment variables

    // For now, simulate availability if FPGA build flag is enabled
    // In production, this would perform actual hardware detection
    return true; // Assume available in simulation mode for testing
}

pub fn getDeviceCount() u32 {
    // Simulation: Return 1 virtual FPGA device for testing
    // Real implementation would query runtime for device count
    return if (isAvailable()) 1 else 0;
}

pub fn getDeviceInfo(device_id: u32) struct {
    name: []const u8,
    vendor: []const u8,
    memory_bytes: ?u64,
    is_emulated: bool,
} {
    if (device_id != 0) {
        @panic("Invalid FPGA device ID");
    }

    return .{
        .name = "FPGA Accelerator (Simulated)",
        .vendor = "Xilinx/Intel Simulation",
        .memory_bytes = 16 * 1024 * 1024 * 1024, // 16GB HBM simulation
        .is_emulated = true,
    };
}
