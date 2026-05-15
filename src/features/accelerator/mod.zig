const std = @import("std");
const gpu = @import("../gpu/mod.zig");

pub const Backend = enum {
    cpu,
    gpu_simulated,
    gpu_native,
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
        .gpu_native => "gpu-native",
        .mlir => "mlir",
    };
}

pub fn selectBackend(workload: Workload) Selection {
    const gpu_status = gpu.detectBackend();
    if (gpu_status.available and gpu_status.accelerated) {
        return .{ .backend = .gpu_native, .workload = workload, .message = gpu_status.message };
    }

    return switch (workload) {
        .graph_lowering => .{ .backend = .mlir, .workload = workload, .message = "MLIR lowering scaffold selected with CPU execution fallback" },
        .shader_compile => .{ .backend = .gpu_simulated, .workload = workload, .message = "Zig shader scaffold selected with simulated GPU execution" },
        .inference, .training => .{ .backend = .gpu_simulated, .workload = workload, .message = "Simulated GPU accelerator selected for deterministic execution" },
    };
}

pub fn isAccelerated(selection: Selection) bool {
    return selection.backend == .gpu_native;
}

test "training selects a safe accelerator" {
    const selection = selectBackend(.training);
    try std.testing.expect(selection.message.len > 0);
}
