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
    return .{ .backend = .cpu, .workload = workload, .message = "accelerator feature is disabled; using CPU" };
}

pub fn isAccelerated(selection: Selection) bool {
    _ = selection;
    return false;
}
