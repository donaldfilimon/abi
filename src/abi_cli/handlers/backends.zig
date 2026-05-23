const std = @import("std");
const features = @import("../../features/mod.zig");

pub fn handleBackends() !u8 {
    const gpu_status = features.gpu.detectBackend();
    const native_gpu = features.gpu.nativeKernelStatus();
    const gpu_report = try features.gpu.backendStatusReport(std.heap.page_allocator);
    defer std.heap.page_allocator.free(gpu_report);
    const training = features.accelerator.selectBackend(.training);
    const shader_status = features.shaders.compilerStatus();
    const shader = try features.shaders.compile(std.heap.page_allocator, .{
        .name = "status",
        .source = "fn main() void {}",
    });
    defer shader.deinit(std.heap.page_allocator);
    const mlir_status = features.mlir.toolchainStatus();
    const lowered = try features.mlir.lower(std.heap.page_allocator, .{
        .name = "status",
        .operations = &.{"matmul"},
    });
    defer lowered.deinit(std.heap.page_allocator);

    std.debug.print("GPU: {s} available={any} accelerated={any}\n", .{
        features.gpu.backendName(gpu_status.backend),
        gpu_status.available,
        gpu_status.accelerated,
    });
    std.debug.print("GPU backend report:\n{s}\n", .{gpu_report});
    std.debug.print("Native GPU kernels: linked={any} ({s})\n", .{ native_gpu.linked, native_gpu.message });
    std.debug.print("Accelerator: {s} ({s})\n", .{ features.accelerator.backendName(training.backend), training.message });
    std.debug.print("Shader: {s} backend={s} compiler_available={any} ({s})\n", .{ features.shaders.languageName(shader.language), shader.backend, shader_status.available, shader_status.message });
    std.debug.print("MLIR: {s} backend={s} toolchain_available={any} ({s})\n", .{ features.mlir.dialectName(lowered.dialect), lowered.target_backend, mlir_status.available, mlir_status.message });
    return 0;
}
