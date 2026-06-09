pub const Backend = enum {
    cpu,
    gpu_simulated,
    gpu_metal,
    gpu_vulkan,
    gpu_cuda,
    gpu_webgpu,
    gpu_opengl,
    gpu_webgl2,
    mlir,
};

pub const Workload = enum {
    inference,
    training,
    shader_compile,
    graph_lowering,
};

pub const Selection = struct {
    backend: Backend,
    workload: Workload,
    message: []const u8,
};

pub const SelectionReport = struct {
    workload: Workload,
    selected_backend: Backend,
    fallback_backend: Backend,
    native_available: bool,
    gpu_available: bool,
    gpu_accelerated: bool,
    message: []const u8,
};

pub fn backendName(backend: Backend) []const u8 {
    return switch (backend) {
        .cpu => "cpu",
        .gpu_simulated => "gpu-simulated",
        .gpu_metal => "gpu-metal",
        .gpu_vulkan => "gpu-vulkan",
        .gpu_cuda => "gpu-cuda",
        .gpu_webgpu => "gpu-webgpu",
        .gpu_opengl => "gpu-opengl",
        .gpu_webgl2 => "gpu-webgl2",
        .mlir => "mlir",
    };
}

pub fn workloadName(workload: Workload) []const u8 {
    return switch (workload) {
        .inference => "inference",
        .training => "training",
        .shader_compile => "shader-compile",
        .graph_lowering => "graph-lowering",
    };
}

pub fn selectBackend(workload: Workload) Selection {
    const report = selectionReport(workload);
    return .{
        .backend = report.selected_backend,
        .workload = report.workload,
        .message = report.message,
    };
}

pub fn selectionReport(workload: Workload) SelectionReport {
    return .{
        .workload = workload,
        .selected_backend = .cpu,
        .fallback_backend = .cpu,
        .native_available = false,
        .gpu_available = false,
        .gpu_accelerated = false,
        .message = "accelerator feature is disabled; using CPU",
    };
}

pub fn isAccelerated(selection: Selection) bool {
    _ = selection;
    return false;
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}

test "accelerator stub reports explicit disabled fallback" {
    const std = @import("std");
    const report = selectionReport(.training);
    try std.testing.expectEqual(Backend.cpu, report.selected_backend);
    try std.testing.expectEqual(Backend.cpu, report.fallback_backend);
    try std.testing.expect(!report.native_available);
    try std.testing.expect(!report.gpu_available);
    try std.testing.expect(!report.gpu_accelerated);
    try std.testing.expectEqualStrings("training", workloadName(report.workload));
}
