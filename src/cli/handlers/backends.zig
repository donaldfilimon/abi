const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");

/// `abi backends`: report the detected compute backends — GPU backend and
/// native-kernel status, the accelerator training selection, and the shader
/// compiler status with a sample compile — plus build-time feature flags
/// and framework version info. Returns the process exit code.
pub fn handleBackends() !u8 {
    // ── Version / build header ──
    std.debug.print("ABI Framework  0.1.0\n", .{});
    std.debug.print("Zig {s}  {s}  {s}\n\n", .{
        builtin.zig_version_string,
        @tagName(builtin.mode),
        @tagName(builtin.os.tag),
    });

    // ── Feature flags ──
    const features_list = [_]struct {
        name: []const u8,
        enabled: bool,
        desc: []const u8,
    }{
        .{ .name = "ai", .enabled = build_options.feat_ai, .desc = "AI profiles, routing, constitution" },
        .{ .name = "wdbx", .enabled = build_options.feat_wdbx, .desc = "Vector store, HNSW index, persistence" },
        .{ .name = "sea", .enabled = build_options.feat_sea, .desc = "Self-learning evidence loop" },
        .{ .name = "nn", .enabled = build_options.feat_nn, .desc = "Neural net demo trainer" },
        .{ .name = "gpu", .enabled = build_options.feat_gpu, .desc = "GPU acceleration (Metal)" },
        .{ .name = "accelerator", .enabled = build_options.feat_accelerator, .desc = "Accelerator backend routing" },
        .{ .name = "shaders", .enabled = build_options.feat_shader, .desc = "Shader validation" },
        .{ .name = "mlir", .enabled = build_options.feat_mlir, .desc = "Textual MLIR lowering" },
        .{ .name = "tui", .enabled = build_options.feat_tui, .desc = "TUI dashboard" },
        .{ .name = "os_control", .enabled = build_options.feat_os_control, .desc = "OS command policy" },
        .{ .name = "telemetry", .enabled = build_options.feat_telemetry, .desc = "Event telemetry" },
        .{ .name = "foundationmodels", .enabled = build_options.feat_foundationmodels, .desc = "Apple Foundation Models" },
        .{ .name = "hash", .enabled = build_options.feat_hash, .desc = "Portable hashing utilities" },
        .{ .name = "metrics", .enabled = build_options.feat_metrics, .desc = "In-process observability" },
        .{ .name = "mobile", .enabled = build_options.feat_mobile, .desc = "Mobile platform detection" },
    };

    std.debug.print("Features:\n", .{});
    for (features_list) |f| {
        const status = if (f.enabled) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m";
        std.debug.print("  {s:<18} {s}  {s}\n", .{ f.name, status, f.desc });
    }
    std.debug.print("\n", .{});

    // ── Compute backends ──
    // Probe Metal (or CPU fallback) before reporting so status reflects a real
    // init attempt, not a cold uninitialized context.
    _ = features.gpu.vectorOps();
    const gpu_status = features.gpu.detectBackend();
    const native_gpu = features.gpu.nativeKernelStatus();
    const gpu_report = try features.gpu.backendStatusReport(std.heap.page_allocator);
    defer std.heap.page_allocator.free(gpu_report);
    const training = features.accelerator.selectionReport(.training);
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

    std.debug.print("Compute Backends:\n", .{});
    std.debug.print("  GPU:            {s}  {s}  accelerated={s}\n", .{
        features.gpu.backendName(gpu_status.backend),
        if (gpu_status.available) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m",
        if (gpu_status.accelerated) "\x1b[32myes\x1b[0m" else "\x1b[90mno\x1b[0m",
    });
    std.debug.print("  Native kernels  {s}  {s}\n", .{
        if (native_gpu.linked) "\x1b[32mlinked\x1b[0m" else "\x1b[90mnot linked\x1b[0m",
        native_gpu.message,
    });
    std.debug.print("  Accelerator     training \x1b[90m→\x1b[0m {s}  (gpu={s} native={s})\n", .{
        features.accelerator.backendName(training.selected_backend),
        if (training.gpu_available) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m",
        if (training.native_available) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m",
    });
    std.debug.print("  Shaders         {s}  {s}\n", .{
        features.shaders.languageName(shader.language),
        if (shader_status.available) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m",
    });
    std.debug.print("  MLIR            {s} \x1b[90m→\x1b[0m {s}  {s}\n", .{
        features.mlir.dialectName(lowered.dialect),
        lowered.target_backend,
        if (mlir_status.available) "\x1b[32m✓\x1b[0m" else "\x1b[90m○\x1b[0m",
    });
    return 0;
}

test "backends handler runs the full status path and returns success" {
    // Smoke coverage: the GPU/accelerator/shader/MLIR report path executes
    // end-to-end on the deterministic CPU-fallback surface and exits 0 without
    // crashing. Guards against a regression in any of the four status reporters.
    try std.testing.expectEqual(@as(u8, 0), try handleBackends());
}

test {
    std.testing.refAllDecls(@This());
}
