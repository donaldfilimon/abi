//! Shared types for the gpu feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

/// Errors returned by GPU memory operations.
pub const MemoryError = error{
    OutOfMemory,
    InvalidPointer,
    BufferTooSmall,
    HostAccessDisabled,
    DeviceMemoryMissing,
    SizeMismatch,
    InvalidOffset,
    TransferFailed,
};

/// Errors returned by GPU kernel operations.
pub const KernelError = error{
    CompilationFailed,
    InvalidKernel,
    InvalidArgument,
    LaunchFailed,
    BackendUnsupported,
};

/// Unified GPU error type combining memory errors and a disabled sentinel.
pub const GpuError = MemoryError || error{GpuDisabled};

/// Convenience alias for the primary GPU error set.
pub const Error = GpuError;

/// Errors returned when selecting a GPU backend.
pub const BackendSelectionError = error{
    RequestedBackendUnavailable,
    NoBackendsAvailable,
    OutOfMemory,
};

/// Summary of GPU memory usage.
pub const MemoryInfo = struct {
    total_bytes: u64 = 0,
    used_bytes: u64 = 0,
    free_bytes: u64 = 0,
    peak_used_bytes: u64 = 0,
};

/// Aggregate GPU execution statistics.
pub const GpuStats = struct {
    kernels_launched: u64 = 0,
    buffers_created: u64 = 0,
    bytes_allocated: u64 = 0,
    host_to_device_transfers: u64 = 0,
    device_to_host_transfers: u64 = 0,
    total_execution_time_ns: u64 = 0,
};

/// Aggregate metrics summary for GPU operations.
pub const MetricsSummary = struct {
    total_kernel_invocations: u64 = 0,
    avg_kernel_time_ns: f64 = 0,
    kernels_per_second: f64 = 0,
};
