//! GPU backend detection, kernel management, and memory utilities.
const std = @import("std");
const backend = @import("backend.zig");
const kernels = @import("kernels.zig");
const memory = @import("memory.zig");

pub const MemoryError = memory.MemoryError;
pub const BufferFlags = memory.BufferFlags;
pub const GPUBuffer = memory.GPUBuffer;
pub const GPUMemoryPool = memory.GPUMemoryPool;
pub const MemoryStats = memory.MemoryStats;
pub const AsyncTransfer = memory.AsyncTransfer;
pub const GpuError = memory.MemoryError || error{GpuDisabled};

pub const KernelSource = kernels.KernelSource;
pub const KernelConfig = kernels.KernelConfig;
pub const CompiledKernel = kernels.CompiledKernel;
pub const Stream = kernels.Stream;
pub const KernelError = kernels.KernelError;
pub const compileKernel = kernels.compileKernel;
pub const createDefaultKernels = kernels.createDefaultKernels;

pub const Backend = backend.Backend;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const BackendInfo = backend.BackendInfo;
pub const DeviceCapability = backend.DeviceCapability;
pub const DeviceInfo = backend.DeviceInfo;
pub const Summary = backend.Summary;
pub const backendName = backend.backendName;
pub const backendDisplayName = backend.backendDisplayName;
pub const backendDescription = backend.backendDescription;
pub const backendFlag = backend.backendFlag;
pub const backendFromString = backend.backendFromString;
pub const backendSupportsKernels = backend.backendSupportsKernels;
pub const backendAvailability = backend.backendAvailability;
pub const availableBackends = backend.availableBackends;
pub const listBackendInfo = backend.listBackendInfo;
pub const listDevices = backend.listDevices;
pub const defaultDevice = backend.defaultDevice;
pub const defaultDeviceLabel = backend.defaultDeviceLabel;
pub const summary = backend.summary;
pub const moduleEnabled = backend.moduleEnabled;
pub const isEnabled = backend.isEnabled;

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) GpuError!void {
    if (!moduleEnabled()) return error.GpuDisabled;
    initialized = true;
}

pub fn ensureInitialized(allocator: std.mem.Allocator) GpuError!void {
    if (!isInitialized()) {
        try init(allocator);
    }
}

pub fn deinit() void {
    initialized = false;
}

pub fn isInitialized() bool {
    return initialized;
}
