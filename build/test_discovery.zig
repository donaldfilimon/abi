const std = @import("std");
const options_mod = @import("options.zig");
const link = @import("link.zig");
const BuildOptions = options_mod.BuildOptions;

/// A single entry in the feature-test manifest.
///
/// `flag` is the name of a `BuildOptions` bool field (e.g. "feat_ai").
/// When `flag` is `null` the import is unconditional (always-on services).
pub const FeatureTestEntry = struct {
    flag: ?[]const u8,
    path: []const u8,
};

/// Canonical manifest of every source file whose inline tests should be
/// discovered by the feature-test binary.
///
/// This table is the **single source of truth** for feature test coverage.
/// The actual discovery root is generated from these entries during the
/// build, so there is no tracked mirror file to keep in sync.
pub const feature_test_manifest = [_]FeatureTestEntry{
    // ── AI modules (gated on feat_ai) ─────────────────────────────────
    .{ .flag = "feat_ai", .path = "features/ai/eval/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/rag/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/templates/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/orchestration/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/constitution/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/documents/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/memory/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/tools/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/streaming/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/abbey/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/multi_agent/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/explore/explore_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/llm/ops/gpu_memory_pool_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/database/wdbx.zig" },

    // ── LLM module ──────────────────────────────────────────────────────
    .{ .flag = "feat_llm", .path = "features/ai/llm/mod.zig" },

    // ── Feature modules (flag-gated) ────────────────────────────────────
    .{ .flag = "feat_cache", .path = "features/cache/mod.zig" },
    .{ .flag = "feat_gateway", .path = "features/gateway/mod.zig" },
    .{ .flag = "feat_messaging", .path = "features/messaging/mod.zig" },
    .{ .flag = "feat_search", .path = "features/search/mod.zig" },
    .{ .flag = "feat_storage", .path = "features/storage/mod.zig" },
    .{ .flag = "feat_pages", .path = "features/observability/pages/mod.zig" },
    .{ .flag = "feat_analytics", .path = "features/analytics/mod.zig" },
    .{ .flag = "feat_profiling", .path = "features/observability/mod.zig" },
    .{ .flag = "feat_mobile", .path = "features/mobile/mod.zig" },
    .{ .flag = "feat_benchmarks", .path = "features/benchmarks/mod.zig" },
    .{ .flag = "feat_network", .path = "features/network/mod.zig" },
    .{ .flag = "feat_network", .path = "features/network/heartbeat.zig" },
    .{ .flag = "feat_network", .path = "features/network/rpc_protocol.zig" },
    .{ .flag = "feat_network", .path = "features/network/raft_test.zig" },
    .{ .flag = "feat_network", .path = "features/network/tests/distributed_computation_test.zig" },
    .{ .flag = "feat_web", .path = "features/web/mod.zig" },
    .{ .flag = "feat_cloud", .path = "features/cloud/mod.zig" },

    // ── AI facade modules ───────────────────────────────────────────────
    .{ .flag = "feat_ai", .path = "features/ai/facades/core.zig" },
    .{ .flag = "feat_llm", .path = "features/ai/facades/inference.zig" },
    .{ .flag = "feat_training", .path = "features/ai/facades/training.zig" },
    .{ .flag = "feat_reasoning", .path = "features/ai/facades/reasoning.zig" },

    // ── GPU and database ────────────────────────────────────────────────
    .{ .flag = "feat_gpu", .path = "features/gpu/mod.zig" },
    .{ .flag = "feat_database", .path = "features/database/mod.zig" },

    // ── Auth ────────────────────────────────────────────────────────────
    .{ .flag = "feat_auth", .path = "features/auth/auth_test.zig" },

    // ── Compute, documents, desktop ────────────────────────────────────
    .{ .flag = "feat_compute", .path = "features/compute/mod.zig" },
    .{ .flag = "feat_documents", .path = "features/documents/mod.zig" },
    .{ .flag = "feat_desktop", .path = "features/desktop/mod.zig" },

    // ── AI sub-modules with test blocks ────────────────────────────────
    .{ .flag = "feat_ai", .path = "features/ai/coordination/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/profiles/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/personas/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/agents/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/embeddings/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/vision/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/training/mod.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/training/self_learning_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/training/trainable_model_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/reasoning/mod.zig" },

    // ── AI persona tests ───────────────────────────────────────────────
    .{ .flag = "feat_ai", .path = "features/ai/personas/tests/abbey_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/personas/tests/abi_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/personas/tests/aviva_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/personas/tests/benchmark_test.zig" },
    .{ .flag = "feat_ai", .path = "features/ai/personas/tests/integration_test.zig" },

    // ── Database sub-modules and tests ─────────────────────────────────
    .{ .flag = "feat_database", .path = "features/database/semantic_store/mod.zig" },
    .{ .flag = "feat_database", .path = "features/database/database_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/hnsw_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/batch_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/quantization_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/distributed/conflict_resolution_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/distributed/integration_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/distributed/shard_assignment_test.zig" },
    .{ .flag = "feat_database", .path = "features/database/distributed/version_vector_test.zig" },

    // ── Observability sub-modules ──────────────────────────────────────
    .{ .flag = "feat_profiling", .path = "features/observability/tracing.zig" },
    .{ .flag = "feat_profiling", .path = "features/observability/metrics/primitives.zig" },

    // ── Always-on services ──────────────────────────────────────────────
    .{ .flag = null, .path = "services/mcp/mod.zig" },
    .{ .flag = null, .path = "services/mcp/server.zig" },
    .{ .flag = null, .path = "services/mcp/types.zig" },
    .{ .flag = null, .path = "services/acp/mod.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/channel.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/thread_pool.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/dag_pipeline.zig" },
    .{ .flag = null, .path = "services/shared/utils/swiss_map.zig" },
    .{ .flag = null, .path = "services/shared/utils/abix_serialize.zig" },
    .{ .flag = null, .path = "services/shared/utils/primitives.zig" },
    .{ .flag = null, .path = "services/shared/utils/profiler.zig" },
    .{ .flag = null, .path = "services/shared/utils/structured_error.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/arena_pool.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/combinators.zig" },
    .{ .flag = null, .path = "services/shared/sync.zig" },
    .{ .flag = null, .path = "services/shared/tensor.zig" },
    .{ .flag = null, .path = "services/shared/matrix.zig" },
    .{ .flag = null, .path = "services/shared/simd/vector_ops.zig" },
    .{ .flag = null, .path = "services/shared/simd/distances.zig" },
    .{ .flag = null, .path = "services/shared/simd/activations.zig" },
    .{ .flag = null, .path = "services/shared/simd/integer_ops.zig" },
    .{ .flag = null, .path = "services/shared/simd/extras.zig" },
    .{ .flag = null, .path = "services/shared/utils/radix_tree.zig" },
    .{ .flag = null, .path = "services/connectors/mod.zig" },
    .{ .flag = null, .path = "services/connectors/shared.zig" },
    .{ .flag = null, .path = "services/connectors/openai.zig" },
    .{ .flag = null, .path = "services/connectors/anthropic.zig" },
    .{ .flag = null, .path = "services/connectors/ollama.zig" },
    .{ .flag = null, .path = "services/connectors/huggingface.zig" },
    .{ .flag = null, .path = "services/connectors/mistral.zig" },
    .{ .flag = null, .path = "services/connectors/cohere.zig" },
    .{ .flag = null, .path = "services/connectors/lm_studio.zig" },
    .{ .flag = null, .path = "services/connectors/vllm.zig" },
    .{ .flag = null, .path = "services/connectors/mlx.zig" },
    .{ .flag = null, .path = "services/connectors/local_scheduler.zig" },
    .{ .flag = null, .path = "services/connectors/stub.zig" },
    .{ .flag = null, .path = "services/connectors/discord/mod.zig" },
    .{ .flag = null, .path = "services/connectors/discord/utils.zig" },
    .{ .flag = null, .path = "services/connectors/discord/rest_encoders.zig" },
    .{ .flag = null, .path = "services/connectors/claude.zig" },
    .{ .flag = null, .path = "services/connectors/codex.zig" },
    .{ .flag = null, .path = "services/connectors/gemini.zig" },
    .{ .flag = null, .path = "services/connectors/ollama_passthrough.zig" },
    .{ .flag = null, .path = "services/connectors/opencode.zig" },
    .{ .flag = null, .path = "services/ha/mod.zig" },
    .{ .flag = null, .path = "services/ha/replication.zig" },
    .{ .flag = null, .path = "services/ha/backup.zig" },
    .{ .flag = null, .path = "services/ha/pitr.zig" },
    .{ .flag = null, .path = "services/ha/stub.zig" },
    .{ .flag = null, .path = "services/platform/mod.zig" },
    .{ .flag = null, .path = "services/platform/detection.zig" },
    .{ .flag = null, .path = "services/shared/utils/benchmark.zig" },
    .{ .flag = null, .path = "services/shared/time.zig" },
    .{ .flag = null, .path = "services/shared/utils/json/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/retry.zig" },
    .{ .flag = null, .path = "services/shared/logging.zig" },
    .{ .flag = null, .path = "services/shared/plugins.zig" },
    .{ .flag = null, .path = "services/shared/utils.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/aligned.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/stack.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/tracking.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/ring.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/pool.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/zerocopy.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/mod.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/chase_lev.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/lockfree.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/mpmc_queue.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/priority_queue.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/epoch.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/mod.zig" },
    // NOTE: async.zig spawns OS threads + Io.Threaded backend — hangs in test runner
    .{ .flag = null, .path = "services/runtime/scheduling/cancellation.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/future.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/task_group.zig" },
    .{ .flag = null, .path = "services/runtime/workload.zig" },
    .{ .flag = null, .path = "services/tasks/types.zig" },
    .{ .flag = null, .path = "services/tasks/mod.zig" },
    .{ .flag = null, .path = "core/errors.zig" },
    .{ .flag = null, .path = "core/feature_catalog.zig" },
    .{ .flag = null, .path = "wdbx/dist/mod.zig" },
    .{ .flag = null, .path = "wdbx/dist/rpc.zig" },
    .{ .flag = null, .path = "wdbx/dist/replication.zig" },
    .{ .flag = null, .path = "core/registry/mod.zig" },
    .{ .flag = null, .path = "core/registry/stub.zig" },
    .{ .flag = null, .path = "services/shared/security/mod.zig" },
    .{ .flag = null, .path = "services/shared/security/csprng.zig" },
    .{ .flag = null, .path = "services/runtime/engine/result_cache.zig" },
    .{ .flag = null, .path = "services/runtime/engine/steal_policy.zig" },
    .{ .flag = null, .path = "services/runtime/engine/numa.zig" },
    .{ .flag = null, .path = "services/runtime/engine/mod.zig" },
    .{ .flag = null, .path = "services/runtime/memory/mod.zig" },
    .{ .flag = null, .path = "services/shared/errors.zig" },
    .{ .flag = null, .path = "services/shared/utils/binary.zig" },
    .{ .flag = null, .path = "services/shared/utils/encoding/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/config.zig" },
    .{ .flag = null, .path = "services/shared/utils/crypto/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/http/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/http/async_http.zig" },
    .{ .flag = null, .path = "services/shared/utils/fs/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/net/mod.zig" },
    .{ .flag = null, .path = "services/shared/utils/memory/thread_cache.zig" },
    .{ .flag = null, .path = "services/shared/os.zig" },
    .{ .flag = null, .path = "services/shared/simd/simd_test.zig" },
    .{ .flag = null, .path = "services/shared/resilience/circuit_breaker.zig" },
    .{ .flag = null, .path = "services/shared/app_paths.zig" },
    .{ .flag = null, .path = "services/tasks/roadmap.zig" },
    .{ .flag = null, .path = "services/lsp/mod.zig" },
    .{ .flag = null, .path = "services/lsp/jsonrpc.zig" },
    .{ .flag = null, .path = "services/lsp/types.zig" },
    .{ .flag = null, .path = "services/lsp/client.zig" },
};

fn renderFeatureTestRoot(allocator: std.mem.Allocator) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try out.writer.writeAll(
        "//! Generated feature module test discovery root.\n" ++
            "//! Source of truth: build/test_discovery.zig:feature_test_manifest.\n\n" ++
            "const build_options = @import(\"build_options\");\n\n" ++
            "test {\n",
    );

    for (feature_test_manifest) |entry| {
        if (manifestEntryImportExpr(entry)) |expr| {
            if (entry.flag) |flag| {
                try out.writer.print("    if (build_options.{s}) _ = {s};\n", .{ flag, expr });
            } else {
                try out.writer.print("    _ = {s};\n", .{expr});
            }
            continue;
        }

        if (entry.flag) |flag| {
            try out.writer.print("    if (build_options.{s}) _ = @import(\"{s}\");\n", .{ flag, entry.path });
        } else {
            try out.writer.print("    _ = @import(\"{s}\");\n", .{entry.path});
        }
    }

    try out.writer.writeAll("}\n");
    return try out.toOwnedSlice();
}

fn manifestEntryImportExpr(entry: FeatureTestEntry) ?[]const u8 {
    if (std.mem.eql(u8, entry.path, "wdbx/dist/mod.zig")) return "@import(\"wdbx\").dist";
    if (std.mem.eql(u8, entry.path, "wdbx/dist/rpc.zig")) return "@import(\"wdbx\").dist.rpc";
    if (std.mem.eql(u8, entry.path, "wdbx/dist/replication.zig")) return "@import(\"wdbx\").dist.replication";
    return null;
}

/// Wire up the feature-test binary and return its run step.
///
/// The tracked manifest table in this file is the only source of truth for
/// feature test coverage. The actual discovery root is generated during the
/// build and imports each manifest entry through named module imports.
pub fn addFeatureTests(
    b: *std.Build,
    options: BuildOptions,
    build_opts: *std.Build.Module,
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const root_path = "src/generated_feature_tests.zig";
    const root_source = renderFeatureTestRoot(b.allocator) catch @panic("renderFeatureTestRoot failed");
    var root_file = std.Io.Dir.cwd().createFile(b.graph.io, root_path, .{ .truncate = true }) catch @panic("Failed to create generated feature test root");
    defer root_file.close(b.graph.io);
    root_file.writeStreamingAll(b.graph.io, root_source) catch @panic("Failed to write generated feature test root");

    const feature_test_root = b.createModule(.{
        .root_source_file = b.path(root_path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    feature_test_root.addImport("abi", abi_module);
    feature_test_root.addImport("build_options", build_opts);

    var it = abi_module.import_table.iterator();
    while (it.next()) |import_entry| {
        feature_test_root.addImport(import_entry.key_ptr.*, import_entry.value_ptr.*);
    }

    const is_blocked_darwin = @import("builtin").os.tag == .macos and @import("builtin").os.version_range.semver.min.major >= 26;
    const ft_step = b.step("feature-tests", "Run feature module inline tests");

    if (is_blocked_darwin) {
        const feature_tests = b.addObject(.{
            .name = "feature_tests",
            .root_module = feature_test_root,
        });
        feature_tests.use_llvm = true;
        ft_step.dependOn(&feature_tests.step);
    } else {
        const feature_tests = b.addTest(.{
            .root_module = feature_test_root,
        });
        link.applyAllPlatformLinks(
            feature_tests.root_module,
            target.result.os.tag,
            options.gpu_metal(),
            options.gpu_backends,
        );
        const run_feature_tests = b.addRunArtifact(feature_tests);
        run_feature_tests.skip_foreign_checks = true;
        ft_step.dependOn(&run_feature_tests.step);
    }

    return ft_step;
}
