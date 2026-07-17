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

pub const SelectionReport = struct {
    workload: Workload,
    selected_backend: Backend,
    fallback_backend: Backend,
    native_available: bool,
    native_dispatch: bool,
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
    const gpu_status = gpu.detectBackend();
    const native_kernels = gpu.nativeKernelStatus();
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
        return .{
            .workload = workload,
            .selected_backend = gpu_backend,
            .fallback_backend = fallbackBackend(workload),
            .native_available = native_kernels.linked,
            .native_dispatch = false,
            .gpu_available = true,
            .gpu_accelerated = true,
            .message = if (native_kernels.linked)
                gpu_status.message
            else
                "GPU backend selected via feat-gpu; native accelerator dispatch is not linked in this module",
        };
    }

    if (gpu_status.available) {
        return .{
            .workload = workload,
            .selected_backend = .gpu_simulated,
            .fallback_backend = fallbackBackend(workload),
            .native_available = false,
            .native_dispatch = false,
            .gpu_available = true,
            .gpu_accelerated = false,
            .message = gpu_status.message,
        };
    }

    return switch (workload) {
        .graph_lowering => .{
            .workload = workload,
            .selected_backend = .mlir,
            .fallback_backend = .cpu,
            .native_available = false,
            .native_dispatch = false,
            .gpu_available = false,
            .gpu_accelerated = false,
            .message = "MLIR textual lowering selected; native accelerator dispatch is not linked",
        },
        .shader_compile => .{
            .workload = workload,
            .selected_backend = .gpu_simulated,
            .fallback_backend = .cpu,
            .native_available = false,
            .native_dispatch = false,
            .gpu_available = false,
            .gpu_accelerated = false,
            .message = "Shader validation selected; external compiler toolchains are not linked",
        },
        .inference, .training => .{
            .workload = workload,
            .selected_backend = .gpu_simulated,
            .fallback_backend = .cpu,
            .native_available = false,
            .native_dispatch = false,
            .gpu_available = false,
            .gpu_accelerated = false,
            .message = "Vectorized CPU accelerator selected; native CUDA/NPU/TPU dispatch is not linked",
        },
    };
}

fn fallbackBackend(workload: Workload) Backend {
    return switch (workload) {
        .graph_lowering => .mlir,
        .shader_compile, .inference, .training => .cpu,
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

test "accelerator selection report exposes fallback and native status" {
    const report = selectionReport(.graph_lowering);
    try std.testing.expectEqual(Workload.graph_lowering, report.workload);
    try std.testing.expect(report.message.len > 0);
    try std.testing.expect(backendName(report.selected_backend).len > 0);
    try std.testing.expect(backendName(report.fallback_backend).len > 0);
    try std.testing.expectEqualStrings("graph-lowering", workloadName(report.workload));
    try std.testing.expect(!report.native_dispatch);
    if (report.native_available) {
        try std.testing.expect(isAccelerated(.{ .backend = report.selected_backend, .workload = report.workload, .message = report.message }));
    } else {
        try std.testing.expect(!report.gpu_accelerated);
    }
}

test "accelerator never claims native dispatch in this module" {
    inline for (std.enums.values(Workload)) |workload| {
        const report = selectionReport(workload);
        try std.testing.expect(!report.native_dispatch);
        try std.testing.expect(report.message.len > 0);
    }
}

test "accelerator shader and graph workloads disclose no linked toolchain when gpu fallback" {
    const gpu_status = gpu.detectBackend();
    if (gpu_status.available and gpu_status.accelerated) return;

    const shader = selectionReport(.shader_compile);
    try std.testing.expect(std.mem.indexOf(u8, shader.message, "not linked") != null);

    const graph = selectionReport(.graph_lowering);
    try std.testing.expect(std.mem.indexOf(u8, graph.message, "not linked") != null);
}
