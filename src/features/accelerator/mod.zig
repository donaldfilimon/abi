const std = @import("std");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

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
    const gpu_status = gpu.detectBackend();
    if (gpu_status.available and gpu_status.accelerated) {
        const gpu_backend: Backend = switch (gpu_status.backend) {
            .simulated => .gpu_simulated,
            .metal => .gpu_metal,
            .vulkan => .gpu_vulkan,
            .cuda => .gpu_cuda,
            .webgpu => .gpu_webgpu,
            .opengl => .gpu_opengl,
            .webgl2 => .gpu_webgl2,
        };
        return .{ .backend = gpu_backend, .workload = workload, .message = gpu_status.message };
    }

    if (gpu_status.available) {
        return .{ .backend = .gpu_simulated, .workload = workload, .message = gpu_status.message };
    }

    return switch (workload) {
        .graph_lowering => .{ .backend = .mlir, .workload = workload, .message = "MLIR textual lowering selected with CPU execution fallback" },
        .shader_compile => .{ .backend = .gpu_simulated, .workload = workload, .message = "Zig shader validation selected with deterministic GPU metadata" },
        .inference, .training => .{ .backend = .gpu_simulated, .workload = workload, .message = "Vectorized CPU accelerator selected for deterministic execution" },
    };
}

pub fn isAccelerated(selection: Selection) bool {
    return switch (selection.backend) {
        .gpu_metal, .gpu_vulkan, .gpu_cuda, .gpu_webgpu, .gpu_opengl, .gpu_webgl2 => true,
        else => false,
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "training selects a safe accelerator" {
    const selection = selectBackend(.training);
    try std.testing.expect(selection.message.len > 0);
}
