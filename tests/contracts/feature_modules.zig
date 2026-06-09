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

    // Exercise the distance primitive used by HNSW/WDBX. VectorOps.dot /
    // squaredL2 / cosineSimilarity select the native GPU kernel path (Metal on
    // macOS when feat-gpu + backend.accelerated + metal context initialized) or
    // fall back to vectorized CPU. HNSW routes cosine distance through this same
    // abstraction and retains its SIMD fallback for deterministic disabled paths.
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    const cos_ident = try ops.cosineSimilarity(&.{ 1.0, 0.0, 0.0 }, &.{ 1.0, 0.0, 0.0 });
    try std.testing.expectEqual(@as(f32, 1.0), cos_ident);
    const cos_ortho = try ops.cosineSimilarity(&.{ 1.0, 0.0 }, &.{ 0.0, 1.0 });
    try std.testing.expectEqual(@as(f32, 0.0), cos_ortho);

    const accelerator_selection = features.accelerator.selectBackend(.training);
    try std.testing.expect(accelerator_selection.message.len > 0);

    try std.testing.expect(features.shaders.compilerStatus().message.len > 0);
    try std.testing.expect(features.mlir.toolchainStatus().message.len > 0);
    try std.testing.expect(features.mobile.detectPlatform().message.len > 0);

    const dashboard = try features.tui.renderDashboard(std.testing.allocator, .{ .title = "ABI" });
    defer std.testing.allocator.free(dashboard);
    try std.testing.expect(dashboard.len > 0);

    const command_decision = features.os_control.validateCommand(.{ .argv = &.{"ls"} }, .{ .workspace_root = "/tmp/work" });
    try std.testing.expect(command_decision.message.len > 0);

    // New utilities + observability features (hash + metrics)
    const h = features.hash.wyhash("contract test", 0);
    try std.testing.expect(h != 0);
    try std.testing.expect(features.hash.isEnabled() or !build_options.feat_hash);

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
    }

    if (!build_options.feat_mlir) {
        const status = features.mlir.toolchainStatus();
        try std.testing.expect(!status.available);
        try std.testing.expectEqualStrings("disabled", status.backend);
        var lowered = try features.mlir.lower(std.testing.allocator, .{ .name = "contract" });
        defer lowered.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("disabled", lowered.target_backend);
        try expectContains(lowered.ir, "disabled");
    }

    if (!build_options.feat_shader) {
        const status = features.shaders.compilerStatus();
        try std.testing.expect(!status.available);
        try std.testing.expectEqualStrings("disabled", status.backend);
        try std.testing.expectError(error.MissingShaderEntryPoint, features.shaders.validate(.{ .name = "contract", .source = "kernel" }));
        try features.shaders.validate(.{ .name = "contract", .source = "fn main() void {}" });
        const artifact = try features.shaders.compile(std.testing.allocator, .{ .name = "contract", .source = "fn main() void {}" });
        defer artifact.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("disabled", artifact.backend);
        try expectContains(artifact.bytes, "disabled");
    }

    if (!build_options.feat_tui) {
        const dashboard = try features.tui.renderDashboard(std.testing.allocator, .{ .title = "ABI" });
        defer std.testing.allocator.free(dashboard);
        try expectContains(dashboard, "disabled");
        const diagnostics = try features.tui.renderDiagnostics(std.testing.allocator, .{});
        defer std.testing.allocator.free(diagnostics);
        try expectContains(diagnostics, "disabled");
    }

    if (!build_options.feat_metrics) {
        var m = features.metrics.Metrics.init(std.testing.allocator);
        defer m.deinit();
        try std.testing.expectError(error.FeatureDisabled, m.increment("x", 1));
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
        var info = try features.mobile.getDeviceInfo(std.testing.allocator);
        defer info.deinit(std.testing.allocator);
        try std.testing.expectEqual(features.mobile.Platform.unknown, info.platform);
        const view = try features.mobile.renderMobileView(std.testing.allocator, "ABI", &.{"item"});
        defer std.testing.allocator.free(view);
        try expectContains(view, "disabled");
        const task = try features.mobile.executeMobileTask(std.testing.allocator, "sync");
        defer std.testing.allocator.free(task);
        try expectContains(task, "disabled");
    }
}
