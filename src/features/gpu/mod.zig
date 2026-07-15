// GPU module entry point.
// Re-exports all sub-modules: backends, vector_ops, reporting, and shared Metal context.

const std = @import("std");

pub const backends = @import("backends.zig");
pub const vector_ops = @import("vector_ops.zig");
pub const reporting = @import("reporting.zig");

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

// Parent GPU compute API: one backend-agnostic facade (Metal-real + CPU
// fallback) with an honest per-backend availability matrix.
pub const compute_api = @import("compute_api.zig");
pub const GpuCompute = compute_api.GpuCompute;
pub const Kernel = compute_api.Kernel;
pub const BackendAvailability = compute_api.BackendAvailability;
pub const backendMatrix = compute_api.backendMatrix;

test {
    std.testing.refAllDecls(@This());
    _ = compute_api;
}

test "gpu module reexports safe vector operations" {
    const caps = backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), caps.len);
    try std.testing.expect(detectBackend().message.len > 0);

    const ops = vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));

    const report = try backendStatusReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "simulated:") != null);
}
