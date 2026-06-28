const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "feature namespaces are stable across flags" {
    const features = abi.features;
    inline for (.{
        "ai",
        "accelerator",
        "gpu",
        "mlir",
        "os_control",
        "shaders",
        "tui",
        "wdbx",
        "mobile",
        "hash",
        "metrics",
        "telemetry",
        "sea",
        "nn",
    }) |decl_name| {
        try std.testing.expect(@hasDecl(features, decl_name));
    }
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

test "feature modules expose safe runtime contracts" {
    const features = abi.features;
    try std.testing.expect(@hasDecl(features.wdbx.persistence, "CHECKSUM_PREFIX"));
    try std.testing.expect(@hasDecl(features.wdbx.Store, "restoreBlock"));
    try std.testing.expect(@hasDecl(features.wdbx.storage.BlockChain, "appendAt"));
    try std.testing.expect(@hasDecl(features.wdbx.spatial_3d.SpatialIndex3D, "initWithPool"));

    const gpu_caps = features.gpu.backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), gpu_caps.len);
    try std.testing.expect(features.gpu.detectBackend().message.len > 0);
    const ops = features.gpu.vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectError(error.DimensionMismatch, ops.dot(&.{1}, &.{ 1, 2 }));
    try std.testing.expect(@hasDecl(features.gpu.VectorOps, "batchCosineSimilarity"));

    // Exercise the distance primitive used by HNSW/WDBX. VectorOps.dot /
    // squaredL2 / cosineSimilarity / batchCosineSimilarity select the native GPU
    // kernel path only when the backend reports initialized native kernels; otherwise
    // they fall back to vectorized CPU. HNSW routes cosine distance through this
    // same abstraction and retains its SIMD fallback for deterministic disabled paths.
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    const cos_ident = try ops.cosineSimilarity(&.{ 1.0, 0.0, 0.0 }, &.{ 1.0, 0.0, 0.0 });
    try std.testing.expectEqual(@as(f32, 1.0), cos_ident);
    const cos_ortho = try ops.cosineSimilarity(&.{ 1.0, 0.0 }, &.{ 0.0, 1.0 });
    try std.testing.expectEqual(@as(f32, 0.0), cos_ortho);
    const batch_candidates = [_][]const f32{
        &.{ 1.0, 0.0, 0.0 },
        &.{ 0.0, 1.0, 0.0 },
    };
    var batch_out: [2]f32 = undefined;
    try ops.batchCosineSimilarity(&.{ 1.0, 0.0, 0.0 }, &batch_candidates, &batch_out);
    try std.testing.expectEqual(@as(f32, 1.0), batch_out[0]);
    try std.testing.expectEqual(@as(f32, 0.0), batch_out[1]);
    var bad_batch_out: [1]f32 = undefined;
    try std.testing.expectError(error.DimensionMismatch, ops.batchCosineSimilarity(&.{ 1.0, 0.0, 0.0 }, &batch_candidates, &bad_batch_out));

    const accelerator_selection = features.accelerator.selectBackend(.training);
    try std.testing.expect(accelerator_selection.message.len > 0);
    try std.testing.expect(@hasDecl(features.accelerator, "SelectionReport"));
    try std.testing.expect(@hasDecl(features.accelerator, "selectionReport"));
    try std.testing.expect(@hasDecl(features.accelerator, "workloadName"));
    const accelerator_report = features.accelerator.selectionReport(.training);
    try std.testing.expectEqual(features.accelerator.Workload.training, accelerator_report.workload);
    try std.testing.expectEqualStrings("training", features.accelerator.workloadName(accelerator_report.workload));
    try std.testing.expect(features.accelerator.backendName(accelerator_report.selected_backend).len > 0);
    try std.testing.expect(features.accelerator.backendName(accelerator_report.fallback_backend).len > 0);
    try std.testing.expect(accelerator_report.message.len > 0);
    if (accelerator_report.native_available) {
        try std.testing.expect(features.accelerator.isAccelerated(accelerator_selection));
    } else {
        try std.testing.expect(!accelerator_report.gpu_accelerated);
    }

    try std.testing.expect(features.shaders.compilerStatus().message.len > 0);
    try std.testing.expect(@hasDecl(features.shaders, "ValidationReport"));
    try std.testing.expect(@hasDecl(features.shaders, "validateDetailed"));
    const shader_report = try features.shaders.validateDetailed(.{ .name = "contract", .source = "fn main() void {}" });
    try std.testing.expectEqualStrings("main", shader_report.entry_point);
    try std.testing.expectEqual(@as(usize, "fn main() void {}".len), shader_report.source_bytes);
    try std.testing.expect(shader_report.checksum != 0);
    try std.testing.expectError(error.UnbalancedShaderDelimiters, features.shaders.validate(.{ .name = "contract", .source = "fn main() void {" }));

    try std.testing.expect(features.mlir.toolchainStatus().message.len > 0);
    try std.testing.expect(@hasDecl(features.mlir, "ModuleAnalysis"));
    try std.testing.expect(@hasDecl(features.mlir, "analyze"));
    const mlir_analysis = try features.mlir.analyze(.{ .name = "contract", .operations = &.{ "matmul", "relu" } });
    try std.testing.expectEqual(@as(usize, 2), mlir_analysis.operation_count);
    try std.testing.expect(mlir_analysis.checksum != 0);
    try std.testing.expectError(error.InvalidMlirModuleName, features.mlir.analyze(.{ .name = "bad name" }));
    const mlir_lowered = try features.mlir.lower(std.testing.allocator, .{ .name = "contract", .operations = &.{"quote \" op"} });
    defer mlir_lowered.deinit(std.testing.allocator);
    if (build_options.feat_mlir) {
        try expectContains(mlir_lowered.ir, "quote \\22 op");
    }

    try std.testing.expect(features.mobile.detectPlatform().message.len > 0);
    try std.testing.expect(@hasDecl(features.mobile, "RuntimeMode"));
    try std.testing.expect(@hasDecl(features.mobile, "MobileProfile"));
    try std.testing.expect(@hasDecl(features.mobile, "DeviceProfile"));
    try std.testing.expect(@hasDecl(features.mobile, "profile"));
    try std.testing.expect(@hasDecl(features.mobile, "deviceProfile"));
    try std.testing.expect(@hasDecl(features.mobile, "layoutSummary"));
    try std.testing.expect(@hasDecl(features.mobile, "runtimeModeName"));
    const mobile_runtime = features.mobile.profile();
    try std.testing.expect(mobile_runtime.message.len > 0);
    try std.testing.expect(mobile_runtime.hardware_model.len > 0);
    try std.testing.expect(!mobile_runtime.native_dispatch);
    try std.testing.expect(features.mobile.runtimeModeName(mobile_runtime.mode).len > 0);
    var mobile_profile = features.mobile.deviceProfile();
    mobile_profile.item_count = 2;
    if (build_options.feat_mobile) {
        try std.testing.expect(mobile_runtime.mode == .native_platform or mobile_runtime.mode == .simulated_profile);
        const summary = try features.mobile.layoutSummary(mobile_profile);
        try std.testing.expect(summary.width > 0);
        try std.testing.expect(summary.height > 0);
        try std.testing.expect(summary.density > 0);
        const mobile_view = try features.mobile.renderMobileView(std.testing.allocator, "ABI", &.{ "one", "two" });
        defer std.testing.allocator.free(mobile_view);
        try expectContains(mobile_view, "items=2");
        try expectContains(mobile_view, "native_dispatch=false");
    } else {
        try std.testing.expectEqual(features.mobile.RuntimeMode.disabled, mobile_runtime.mode);
        try std.testing.expectError(error.InvalidMobileView, features.mobile.layoutSummary(mobile_profile));
    }
    const mobile_task = try features.mobile.executeMobileTask(std.testing.allocator, "sync");
    defer std.testing.allocator.free(mobile_task);
    try expectContains(mobile_task, "native_dispatch=false");
    try std.testing.expectError(error.InvalidMobileView, features.mobile.renderMobileView(std.testing.allocator, "ABI", &.{""}));
    try std.testing.expectError(error.InvalidTaskName, features.mobile.executeMobileTask(std.testing.allocator, "bad\x00task"));

    const dashboard = try features.tui.renderDashboard(std.testing.allocator, .{ .title = "ABI" });
    defer std.testing.allocator.free(dashboard);
    try std.testing.expect(dashboard.len > 0);

    // TUI output sanitizer (escape-injection hardening): strips ESC/NUL from
    // attacker-influenced fields before they reach ANSI render output. Available
    // and stripping in both builds (the stub mirrors the real strip), like the
    // unconditional renderDashboard contract above.
    const tui_sanitized = try features.tui.sanitizeControlBytes(std.testing.allocator, "\x1b[2J\x00danger");
    defer std.testing.allocator.free(tui_sanitized);
    try std.testing.expect(std.mem.indexOfScalar(u8, tui_sanitized, 0x1b) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, tui_sanitized, 0x00) == null);

    // C1 closure (both builds): the UTF-8-aware sanitizer neutralizes a lone raw
    // CSI (0x9B) and an encoded C1 (0xC2 0x9B) so neither byte survives. Length is
    // preserved. The stub mirrors this behavior, so this runs feat-on and feat-off.
    const tui_c1 = try features.tui.sanitizeControlBytes(std.testing.allocator, "x\x9b\xc2\x9by");
    defer std.testing.allocator.free(tui_c1);
    try std.testing.expect(std.mem.indexOfScalar(u8, tui_c1, 0x9b) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, tui_c1, 0xc2) == null);
    try std.testing.expectEqual(@as(usize, 5), tui_c1.len);

    const command_decision = features.os_control.validateCommand(.{ .argv = &.{"ls"} }, .{ .workspace_root = "/tmp/work" });
    try std.testing.expect(command_decision.message.len > 0);

    // New utilities + observability features (hash + metrics)
    const h = features.hash.wyhash("contract test", 0);
    try std.testing.expect(h != 0);
    try std.testing.expect(features.hash.isEnabled() or !build_options.feat_hash);

    // nn char-LM trainer: when enabled, a tiny training run must strictly reduce
    // cross-entropy loss (the feature's hard success contract).
    if (build_options.feat_nn) {
        try std.testing.expect(features.nn.isEnabled());
        const report = try features.nn.trainOnText(std.testing.allocator, "hello world ", .{
            .seq_len = 2,
            .epochs = 120,
            .lr = 0.5,
            .seed = 99,
        });
        try std.testing.expect(report.improved);
        try std.testing.expect(report.final_loss < report.initial_loss);
    }

    if (build_options.feat_metrics) {
        var m = features.metrics.Metrics.init(std.testing.allocator);
        defer m.deinit();
        try m.increment("contract.test", 1);
        try std.testing.expectEqual(@as(u64, 1), m.getCounter("contract.test").?);
    }

    if (build_options.feat_telemetry) {
        features.telemetry.reset();
        defer features.telemetry.reset();
        features.telemetry.record("contract.event");
        features.telemetry.increment("contract.event", 2);
        try std.testing.expectEqual(@as(u64, 3), features.telemetry.counterValue("contract.event"));
        try std.testing.expectEqual(@as(u64, 3), features.telemetry.totalEvents());
        try std.testing.expectEqual(@as(usize, 1), features.telemetry.distinctCounters());
    }

    if (build_options.feat_wdbx) {
        var store = features.wdbx.Store.init(std.testing.allocator);
        defer store.deinit();
        const stats = store.stats();
        try std.testing.expect(stats.acceleration.message.len > 0);
        const manifest = try store.exportManifest(std.testing.allocator);
        defer std.testing.allocator.free(manifest);
        try std.testing.expect(manifest.len > 0);

        // Exercise HNSW index path (via Store.putVector + search), which now routes
        // distance calculations through gpu.vectorOps and drives acceleration status
        // updates through gpu.executeKernel (reflecting native_gpu / simulated_gpu mode).
        // Asserts that acceleration status reflects the mode selected by the GPU path
        // without fabricating success when disabled/fallback.
        const vid1 = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
        _ = try store.putVector(&.{ 0.9, 0.1, 0.0, 0.0 });
        const hits = try store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 2);
        defer std.testing.allocator.free(hits);
        try std.testing.expect(hits.len >= 1);
        try std.testing.expectEqual(vid1, hits[0].id);
        if (hits.len > 1) {
            try std.testing.expect(hits[0].score >= hits[1].score);
        }

        const accel = store.accelerationStatus();
        try std.testing.expect(accel.message.len > 0);
        if (build_options.feat_gpu) {
            try std.testing.expect(accel.mode == features.gpu.ExecutionMode.native_gpu or accel.mode == features.gpu.ExecutionMode.simulated_gpu);
        } else {
            try std.testing.expect(accel.mode == features.gpu.ExecutionMode.cpu_fallback);
        }

        // Persistence feature: a serialize/deserialize round-trip locks the public
        // snapshot surface and verifies the store is reconstructed without widening
        // vector dimensionality (padded-width regression guard).
        const snapshot = try features.wdbx.persistence.serialize(std.testing.allocator, &store);
        defer std.testing.allocator.free(snapshot);
        try std.testing.expect(snapshot.len > 0);
        var restored = try features.wdbx.persistence.deserialize(std.testing.allocator, snapshot);
        defer restored.deinit();
        try std.testing.expectEqual(store.vectorCount(), restored.vectorCount());
        const restored_hits = try restored.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 1);
        defer std.testing.allocator.free(restored_hits);
        try std.testing.expect(restored_hits.len == 1);
    } else {
        // Disabled wdbx must degrade cleanly (per AGENTS.md + feature stub contract)
        var store = features.wdbx.Store.init(std.testing.allocator);
        defer store.deinit();
        try std.testing.expectError(error.FeatureDisabled, store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 }));
        try std.testing.expectError(error.FeatureDisabled, store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 1));
        // Disabled persistence must also refuse, never fabricate a snapshot.
        try std.testing.expectError(error.FeatureDisabled, features.wdbx.persistence.serialize(std.testing.allocator, &store));
        const stats = store.stats();
        try std.testing.expect(stats.acceleration.message.len > 0);
        // Stub reports disabled message
        try std.testing.expect(std.mem.indexOf(u8, stats.acceleration.message, "disabled") != null or stats.vectors == 0);
    }

    try std.testing.expect(@hasDecl(features.ai, "CompletionTaskContext"));
    try std.testing.expect(@hasDecl(features.ai, "TrainingTaskContext"));
    try std.testing.expect(@hasDecl(features.ai, "submitCompletionTask"));
    try std.testing.expect(@hasDecl(features.ai, "submitTrainingTask"));
    try std.testing.expect(@hasDecl(features.ai, "completeWithScheduler"));

    var completion = try features.ai.complete(std.testing.allocator, .{ .input = "contract surface", .model = "abi-contract" });
    defer completion.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("abi-contract", completion.model);
    try std.testing.expect(completion.output.len > 0);
}

test "disabled feature modules expose explicit degraded behavior" {
    const features = abi.features;

    if (!build_options.feat_accelerator) {
        const selection = features.accelerator.selectBackend(.training);
        try std.testing.expectEqual(features.accelerator.Backend.cpu, selection.backend);
        try std.testing.expect(!features.accelerator.isAccelerated(selection));
        try expectContains(selection.message, "disabled");
        const report = features.accelerator.selectionReport(.training);
        try std.testing.expectEqual(features.accelerator.Backend.cpu, report.selected_backend);
        try std.testing.expectEqual(features.accelerator.Backend.cpu, report.fallback_backend);
        try std.testing.expect(!report.native_available);
        try std.testing.expect(!report.gpu_available);
        try std.testing.expect(!report.gpu_accelerated);
    }

    if (!build_options.feat_mlir) {
        const status = features.mlir.toolchainStatus();
        try std.testing.expect(!status.available);
        try std.testing.expectEqualStrings("disabled", status.backend);
        var lowered = try features.mlir.lower(std.testing.allocator, .{ .name = "contract" });
        defer lowered.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("disabled", lowered.target_backend);
        try expectContains(lowered.ir, "disabled");
        const analysis = try features.mlir.analyze(.{ .name = "contract", .operations = &.{"matmul"} });
        try std.testing.expectEqual(@as(usize, 1), analysis.operation_count);
        try std.testing.expect(analysis.checksum != 0);
        try std.testing.expectError(error.InvalidMlirModuleName, features.mlir.analyze(.{ .name = "bad name" }));
    }

    if (!build_options.feat_shader) {
        const status = features.shaders.compilerStatus();
        try std.testing.expect(!status.available);
        try std.testing.expectEqualStrings("disabled", status.backend);
        try std.testing.expectError(error.MissingShaderEntryPoint, features.shaders.validate(.{ .name = "contract", .source = "kernel" }));
        try std.testing.expectError(error.UnbalancedShaderDelimiters, features.shaders.validate(.{ .name = "contract", .source = "fn main() void {" }));
        try features.shaders.validate(.{ .name = "contract", .source = "fn main() void {}" });
        const report = try features.shaders.validateDetailed(.{ .name = "contract", .source = "fn main() void {}" });
        try std.testing.expectEqualStrings("main", report.entry_point);
        const artifact = try features.shaders.compile(std.testing.allocator, .{ .name = "contract", .source = "fn main() void {}" });
        defer artifact.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("disabled", artifact.backend);
        try std.testing.expectEqualStrings("main", artifact.entry_point);
        try expectContains(artifact.bytes, "disabled");
    }

    if (!build_options.feat_tui) {
        const dashboard = try features.tui.renderDashboard(std.testing.allocator, .{ .title = "ABI" });
        defer std.testing.allocator.free(dashboard);
        try expectContains(dashboard, "disabled");
        const diagnostics = try features.tui.renderDiagnostics(std.testing.allocator, .{});
        defer std.testing.allocator.free(diagnostics);
        try expectContains(diagnostics, "disabled");

        // The disabled stub still hardens output: sanitizeControlBytes strips
        // ESC/NUL identically (a pure safety utility degrades, it does not refuse).
        const sanitized = try features.tui.sanitizeControlBytes(std.testing.allocator, "\x1b]0;title\x07\x00");
        defer std.testing.allocator.free(sanitized);
        try std.testing.expect(std.mem.indexOfScalar(u8, sanitized, 0x1b) == null);
        try std.testing.expect(std.mem.indexOfScalar(u8, sanitized, 0x00) == null);
    }

    if (!build_options.feat_metrics) {
        var m = features.metrics.Metrics.init(std.testing.allocator);
        defer m.deinit();
        try std.testing.expectError(error.FeatureDisabled, m.increment("x", 1));
    }

    if (!build_options.feat_nn) {
        try std.testing.expect(!features.nn.isEnabled());
        try std.testing.expectError(error.FeatureDisabled, features.nn.trainOnText(std.testing.allocator, "hello world ", .{}));
        try std.testing.expectError(error.FeatureDisabled, features.nn.extractCorpusFromJsonl(std.testing.allocator, "{\"text\":\"x\"}", "text"));
        try std.testing.expectError(error.FeatureDisabled, features.nn.trainOnJsonl(std.testing.allocator, "nonexistent.jsonl", "text", .{}));
        var model = features.nn.Model{};
        try std.testing.expectError(error.FeatureDisabled, features.nn.sample(std.testing.allocator, &model, 'h', 4));
    }

    if (!build_options.feat_telemetry) {
        features.telemetry.record("contract.event");
        features.telemetry.increment("contract.event", 2);
        try std.testing.expectEqual(@as(u64, 0), features.telemetry.counterValue("contract.event"));
        try std.testing.expectEqual(@as(u64, 0), features.telemetry.totalEvents());
        try std.testing.expectEqual(@as(usize, 0), features.telemetry.distinctCounters());
        try std.testing.expectEqual(@as(u64, 0), features.telemetry.droppedEvents());
    }

    if (!build_options.feat_os_control) {
        const request = features.os_control.CommandRequest{ .argv = &.{"ls"} };
        const policy = features.os_control.Policy{ .workspace_root = "/tmp/work" };
        const decision = features.os_control.validateCommand(request, policy);
        try std.testing.expectEqual(features.os_control.Decision.deny, decision.decision);
        try expectContains(decision.message, "disabled");
        const io: std.Io = undefined;
        try std.testing.expectError(error.CommandDenied, features.os_control.executeConfirmed(std.testing.allocator, io, request, policy));
    }

    if (!build_options.feat_mobile) {
        const status = features.mobile.detectPlatform();
        try std.testing.expect(!status.available);
        try std.testing.expect(!status.accelerated);
        try std.testing.expectEqual(features.mobile.Platform.unknown, status.platform);
        const mobile_runtime = features.mobile.profile();
        try std.testing.expectEqual(features.mobile.RuntimeMode.disabled, mobile_runtime.mode);
        try std.testing.expectEqual(@as(u32, 0), mobile_runtime.screen.width);
        try std.testing.expect(!mobile_runtime.native_dispatch);
        const profile = features.mobile.deviceProfile();
        try std.testing.expectEqual(features.mobile.Platform.unknown, profile.platform);
        try std.testing.expectEqual(features.mobile.RuntimeMode.disabled, profile.mode);
        try std.testing.expect(!profile.native_dispatch);
        try std.testing.expect(!profile.simulated);
        try std.testing.expectEqual(@as(u32, 0), profile.width);
        var info = try features.mobile.getDeviceInfo(std.testing.allocator);
        defer info.deinit(std.testing.allocator);
        try std.testing.expectEqual(features.mobile.Platform.unknown, info.platform);
        const view = try features.mobile.renderMobileView(std.testing.allocator, "ABI", &.{"item"});
        defer std.testing.allocator.free(view);
        try expectContains(view, "disabled");
        try expectContains(view, "mode=disabled");
        const task = try features.mobile.executeMobileTask(std.testing.allocator, "sync");
        defer std.testing.allocator.free(task);
        try expectContains(task, "disabled");
        try expectContains(task, "mode=disabled");
    }
}
