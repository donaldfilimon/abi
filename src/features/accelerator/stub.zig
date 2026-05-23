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

pub fn selectBackend(workload: Workload) Selection {
    return .{ .backend = .cpu, .workload = workload, .message = "accelerator feature is disabled; using CPU" };
}

pub fn isAccelerated(selection: Selection) bool {
    _ = selection;
    return false;
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
