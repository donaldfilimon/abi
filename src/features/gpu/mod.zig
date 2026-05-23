// GPU module entry point.
// Re-exports all sub-modules: backends, vector_ops, reporting, and shared Metal context.

pub const backends = @import("backends.zig");
pub const vector_ops = @import("vector_ops.zig");
pub const reporting = @import("reporting.zig");
const metal_shared = @import("metal_shared.zig");

// ---- Type re-exports from backends ----
pub const Backend = backends.Backend;
pub const BackendStatus = backends.BackendStatus;
pub const ExecutionMode = backends.ExecutionMode;
pub const KernelSpec = backends.KernelSpec;
pub const KernelResult = backends.KernelResult;
pub const NativeKernelStatus = backends.NativeKernelStatus;
pub const BackendCapabilities = backends.BackendCapabilities;

// ---- Function re-exports from backends ----
pub const backendName = backends.backendName;
pub const backendStatus = backends.backendStatus;
pub const backendCapabilities = backends.backendCapabilities;
pub const backendCapabilitiesList = backends.backendCapabilitiesList;
pub const detectBackend = backends.detectBackend;
pub const nativeKernelStatus = backends.nativeKernelStatus;
pub const threadsPerGroup = backends.threadsPerGroup;

// ---- Function re-exports from vector_ops ----
pub const VectorOps = vector_ops.VectorOps;
pub const executeKernel = vector_ops.executeKernel;
pub const vectorOps = vector_ops.vectorOps;

// ---- Function re-exports from reporting ----
pub const backendStatusReport = reporting.backendStatusReport;
pub const isAvailable = reporting.isAvailable;
pub const preferredBackend = reporting.preferredBackend;
