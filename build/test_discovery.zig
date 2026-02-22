const std = @import("std");
const options_mod = @import("options.zig");
const link = @import("link.zig");
const targets = @import("targets.zig");
const BuildOptions = options_mod.BuildOptions;

/// A single entry in the feature-test manifest.
///
/// `flag` is the name of a `BuildOptions` bool field (e.g. "enable_ai").
/// When `flag` is `null` the import is unconditional (always-on services).
pub const FeatureTestEntry = struct {
    flag: ?[]const u8,
    path: []const u8,
};

/// Canonical manifest of every source file whose inline tests should be
/// discovered by the feature-test binary.
///
/// This table is the **single source of truth** for feature test coverage.
/// The actual test root (`src/feature_test_root.zig`) must mirror these
/// entries.  When adding a new test source:
///   1. Add an entry to this manifest
///   2. Add the corresponding `@import` to `src/feature_test_root.zig`
pub const feature_test_manifest = [_]FeatureTestEntry{
    // ── AI modules (gated on enable_ai) ─────────────────────────────────
    .{ .flag = "enable_ai", .path = "features/ai/eval/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/rag/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/templates/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/orchestration/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/documents/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/memory/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/tools/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/streaming/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/abbey/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/multi_agent/mod.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/explore/explore_test.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/llm/ops/gpu_memory_pool_test.zig" },
    .{ .flag = "enable_ai", .path = "features/ai/database/wdbx.zig" },

    // ── LLM module ──────────────────────────────────────────────────────
    .{ .flag = "enable_llm", .path = "features/ai/llm/mod.zig" },

    // ── Feature modules (flag-gated) ────────────────────────────────────
    .{ .flag = "enable_cache", .path = "features/cache/mod.zig" },
    .{ .flag = "enable_gateway", .path = "features/gateway/mod.zig" },
    .{ .flag = "enable_messaging", .path = "features/messaging/mod.zig" },
    .{ .flag = "enable_search", .path = "features/search/mod.zig" },
    .{ .flag = "enable_storage", .path = "features/storage/mod.zig" },
    .{ .flag = "enable_pages", .path = "features/pages/mod.zig" },
    .{ .flag = "enable_analytics", .path = "features/analytics/mod.zig" },
    .{ .flag = "enable_profiling", .path = "features/observability/mod.zig" },
    .{ .flag = "enable_mobile", .path = "features/mobile/mod.zig" },
    .{ .flag = "enable_benchmarks", .path = "features/benchmarks/mod.zig" },
    .{ .flag = "enable_network", .path = "features/network/mod.zig" },
    .{ .flag = "enable_web", .path = "features/web/mod.zig" },
    .{ .flag = "enable_cloud", .path = "features/cloud/mod.zig" },

    // ── AI facade modules ───────────────────────────────────────────────
    .{ .flag = "enable_ai", .path = "features/ai/facades/core.zig" },
    .{ .flag = "enable_llm", .path = "features/ai/facades/inference.zig" },
    .{ .flag = "enable_training", .path = "features/ai/facades/training.zig" },
    .{ .flag = "enable_reasoning", .path = "features/ai/facades/reasoning.zig" },

    // ── GPU and database ────────────────────────────────────────────────
    .{ .flag = "enable_gpu", .path = "features/gpu/mod.zig" },
    .{ .flag = "enable_database", .path = "features/database/mod.zig" },

    // ── Auth ────────────────────────────────────────────────────────────
    .{ .flag = "enable_auth", .path = "features/auth/auth_test.zig" },

    // ── Always-on services ──────────────────────────────────────────────
    .{ .flag = null, .path = "services/mcp/server.zig" },
    .{ .flag = null, .path = "services/mcp/types.zig" },
    .{ .flag = null, .path = "services/acp/mod.zig" },
    .{ .flag = null, .path = "services/runtime/concurrency/channel.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/thread_pool.zig" },
    .{ .flag = null, .path = "services/runtime/scheduling/dag_pipeline.zig" },
    .{ .flag = null, .path = "services/shared/utils/swiss_map.zig" },
    .{ .flag = null, .path = "services/shared/utils/abix_serialize.zig" },
    .{ .flag = null, .path = "services/shared/utils/v2_primitives.zig" },
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
    .{ .flag = null, .path = "core/registry/mod.zig" },
    .{ .flag = null, .path = "core/registry/stub.zig" },
    .{ .flag = null, .path = "vnext/capability.zig" },
    .{ .flag = null, .path = "vnext/config.zig" },
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
    .{ .flag = null, .path = "services/tasks/roadmap.zig" },
};

/// Wire up the feature-test binary and return its run step.
///
/// Uses `src/feature_test_root.zig` as the test root (required for correct
/// `@import` path resolution within `src/`).  Applies Metal framework links
/// when targeting macOS with the Metal backend.
///
/// Returns `null` if `src/feature_test_root.zig` does not exist.
pub fn addFeatureTests(
    b: *std.Build,
    options: BuildOptions,
    build_opts: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) ?*std.Build.Step {
    if (!targets.pathExists(b, "src/feature_test_root.zig")) return null;

    const feature_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/feature_test_root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    feature_tests.root_module.addImport("build_options", build_opts);
    link.applyFrameworkLinks(
        feature_tests.root_module,
        target.result.os.tag,
        options.gpu_metal(),
    );

    const run_feature_tests = b.addRunArtifact(feature_tests);
    run_feature_tests.skip_foreign_checks = true;
    const ft_step = b.step("feature-tests", "Run feature module inline tests");
    ft_step.dependOn(&run_feature_tests.step);

    return &run_feature_tests.step;
}
