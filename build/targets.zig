const std = @import("std");

/// Descriptor for a build target (example, tool, etc.).
pub const BuildTarget = struct {
    name: []const u8,
    step_name: []const u8,
    description: []const u8,
    source_path: []const u8,
    optimize: ?std.builtin.OptimizeMode = null,
};

/// All example programs shipped with the project.
pub const example_targets = [_]BuildTarget{
    .{ .name = "example-hello", .step_name = "run-hello", .description = "Run hello example", .source_path = "examples/hello.zig" },
    .{ .name = "example-database", .step_name = "run-database", .description = "Run database example", .source_path = "examples/database.zig" },
    .{ .name = "example-agent", .step_name = "run-agent", .description = "Run agent example", .source_path = "examples/agent.zig" },
    .{ .name = "example-compute", .step_name = "run-compute", .description = "Run compute example", .source_path = "examples/compute.zig" },
    .{ .name = "example-network", .step_name = "run-network", .description = "Run network example", .source_path = "examples/network.zig" },
    .{ .name = "example-discord", .step_name = "run-discord", .description = "Run discord example", .source_path = "examples/discord.zig" },
    .{ .name = "example-llm", .step_name = "run-llm", .description = "Run LLM example", .source_path = "examples/llm.zig" },
    .{ .name = "example-llm-real", .step_name = "run-llm-real", .description = "Run real LLM via connectors (Ollama/LM Studio/vLLM)", .source_path = "examples/llm_real.zig" },
    .{ .name = "example-training", .step_name = "run-training", .description = "Run training example", .source_path = "examples/training.zig" },
    .{ .name = "example-ha", .step_name = "run-ha", .description = "Run HA example", .source_path = "examples/ha.zig" },
    .{ .name = "example-train-demo", .step_name = "run-train-demo", .description = "Run LLM training demo", .source_path = "examples/training/train_demo.zig" },
    .{ .name = "example-orchestration", .step_name = "run-orchestration", .description = "Run multi-model orchestration example", .source_path = "examples/orchestration.zig" },
    .{ .name = "example-train-ava", .step_name = "run-train-ava", .description = "Train Ava assistant from gpt-oss", .source_path = "examples/train_ava.zig" },
    .{ .name = "example-concurrency", .step_name = "run-concurrency", .description = "Run concurrency primitives example", .source_path = "examples/concurrency.zig" },
    .{ .name = "example-observability", .step_name = "run-observability", .description = "Run observability example", .source_path = "examples/observability.zig" },
    .{ .name = "example-gpu", .step_name = "run-gpu", .description = "Run GPU example", .source_path = "examples/gpu.zig" },
    .{ .name = "example-streaming", .step_name = "run-streaming", .description = "Run streaming API example", .source_path = "examples/streaming.zig" },
    .{ .name = "example-registry", .step_name = "run-registry", .description = "Run feature registry example", .source_path = "examples/registry.zig" },
    .{ .name = "example-embeddings", .step_name = "run-embeddings", .description = "Run embeddings example", .source_path = "examples/embeddings.zig" },
    .{ .name = "example-config", .step_name = "run-config", .description = "Run configuration example", .source_path = "examples/config.zig" },
    .{ .name = "example-tensor-ops", .step_name = "run-tensor-ops", .description = "Run tensor + matrix + SIMD example", .source_path = "examples/tensor_ops.zig" },
    .{ .name = "example-concurrent-pipeline", .step_name = "run-concurrent-pipeline", .description = "Run channel + thread pool + DAG pipeline example", .source_path = "examples/concurrent_pipeline.zig" },
    .{ .name = "example-cache", .step_name = "run-cache", .description = "Run cache example", .source_path = "examples/cache.zig" },
    .{ .name = "example-search", .step_name = "run-search", .description = "Run search example", .source_path = "examples/search.zig" },
    .{ .name = "example-messaging", .step_name = "run-messaging", .description = "Run messaging example", .source_path = "examples/messaging.zig" },
    .{ .name = "example-storage", .step_name = "run-storage", .description = "Run storage example", .source_path = "examples/storage.zig" },
    .{ .name = "example-gateway", .step_name = "run-gateway", .description = "Run gateway example", .source_path = "examples/gateway.zig" },
    .{ .name = "example-pages", .step_name = "run-pages", .description = "Run pages example", .source_path = "examples/pages.zig" },
    .{ .name = "example-auth", .step_name = "run-auth", .description = "Run auth example", .source_path = "examples/auth.zig" },
    .{ .name = "example-analytics", .step_name = "run-analytics", .description = "Run analytics example", .source_path = "examples/analytics.zig" },
    .{ .name = "example-cloud", .step_name = "run-cloud", .description = "Run cloud example", .source_path = "examples/cloud.zig" },
    .{ .name = "example-web", .step_name = "run-web", .description = "Run web example", .source_path = "examples/web.zig" },
    .{ .name = "example-ai-core", .step_name = "run-ai-core", .description = "Run AI core example", .source_path = "examples/ai_core.zig" },
    .{ .name = "example-ai-inference", .step_name = "run-ai-inference", .description = "Run AI inference example", .source_path = "examples/ai_inference.zig" },
    .{ .name = "example-ai-training", .step_name = "run-ai-training", .description = "Run AI training example", .source_path = "examples/ai_training.zig" },
    .{ .name = "example-ai-reasoning", .step_name = "run-ai-reasoning", .description = "Run AI reasoning example", .source_path = "examples/ai_reasoning.zig" },
    .{ .name = "example-mobile", .step_name = "run-mobile", .description = "Run mobile example", .source_path = "examples/mobile.zig" },
    .{ .name = "example-gpu-training", .step_name = "run-gpu-training", .description = "Run GPU + training integration example", .source_path = "examples/gpu_training.zig" },
    .{ .name = "example-distributed-db", .step_name = "run-distributed-db", .description = "Run distributed database integration example", .source_path = "examples/distributed_db.zig" },
    .{ .name = "example-web-observability", .step_name = "run-web-observability", .description = "Run web + observability integration example", .source_path = "examples/web_observability.zig" },
};

/// Check whether a path exists within the build root.
pub fn pathExists(b: *std.Build, path: []const u8) bool {
    b.build_root.handle.access(b.graph.io, path, .{}) catch return false;
    return true;
}

/// Register build steps for a table of `BuildTarget` entries.  Skips
/// targets whose source files are missing.
pub fn buildTargets(
    b: *std.Build,
    table: []const BuildTarget,
    abi_module: *std.Build.Module,
    build_opts: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aggregate: ?*std.Build.Step,
    aggregate_runs: bool,
) void {
    for (table) |t| {
        if (!pathExists(b, t.source_path)) continue;
        const exe_optimize = t.optimize orelse optimize;
        const exe = b.addExecutable(.{
            .name = t.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.source_path),
                .target = target,
                .optimize = exe_optimize,
                .link_libc = true,
            }),
        });
        exe.root_module.addImport("abi", abi_module);
        exe.root_module.addImport("build_options", build_opts);
        applyPerformanceTweaks(exe, exe_optimize);
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        const step = b.step(t.step_name, t.description);
        step.dependOn(&run.step);
        if (aggregate) |agg| {
            if (aggregate_runs) {
                agg.dependOn(&run.step);
            } else {
                agg.dependOn(&exe.step);
            }
        }
    }
}

/// Strip release builds by default.
pub fn applyPerformanceTweaks(exe: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    if (optimize == .ReleaseFast or optimize == .ReleaseSmall) {
        if (exe.root_module.strip == null)
            exe.root_module.strip = true;
    }
}
