//! Feature module test discovery root.
//!
//! Separate test binary for discovering inline test blocks inside feature
//! module source files. Zig 0.16 restricts each source file to one module,
//! so these cannot be imported by the main test binary (which already uses
//! them via the 'abi' named module).

const build_options = @import("build_options");

test {
    if (build_options.enable_ai) {
        // Modules verified to compile cleanly in standalone context
        _ = @import("features/ai/eval/mod.zig");
        _ = @import("features/ai/rag/mod.zig");
        _ = @import("features/ai/templates/mod.zig");
        _ = @import("features/ai/orchestration/mod.zig");
        _ = @import("features/ai/documents/mod.zig");

        // Fixed: persistence.zig coercion, edit_tools I/O API, generator init signature
        _ = @import("features/ai/memory/mod.zig");
        _ = @import("features/ai/tools/mod.zig");
        _ = @import("features/ai/streaming/mod.zig");
        _ = @import("features/ai/abbey/mod.zig");

        // Multi-agent coordination (coordinator + aggregation + messaging)
        _ = @import("features/ai/multi_agent/mod.zig");

        // Explore module standalone tests
        _ = @import("features/ai/explore/explore_test.zig");

        // LLM GPU memory pool tests
        _ = @import("features/ai/llm/ops/gpu_memory_pool_test.zig");

        // AI database submodule (wdbx token dataset tests)
        _ = @import("features/ai/database/wdbx.zig");
    }

    // LLM module (152 test blocks across 47 files)
    if (@hasDecl(build_options, "enable_llm") and build_options.enable_llm) {
        _ = @import("features/ai/llm/mod.zig");
    }

    // Phase 9 feature modules (inline tests)
    if (build_options.enable_cache) _ = @import("features/cache/mod.zig");
    if (build_options.enable_gateway) _ = @import("features/gateway/mod.zig");
    if (build_options.enable_messaging) _ = @import("features/messaging/mod.zig");
    if (build_options.enable_search) _ = @import("features/search/mod.zig");
    if (build_options.enable_storage) _ = @import("features/storage/mod.zig");
    if (build_options.enable_pages) _ = @import("features/pages/mod.zig");

    // Analytics, observability, mobile, and benchmarks inline tests
    if (build_options.enable_analytics) _ = @import("features/analytics/mod.zig");
    if (build_options.enable_profiling) _ = @import("features/observability/mod.zig");
    if (build_options.enable_mobile) _ = @import("features/mobile/mod.zig");
    if (build_options.enable_benchmarks) _ = @import("features/benchmarks/mod.zig");

    // Network and web inline tests
    if (build_options.enable_network) _ = @import("features/network/mod.zig");
    if (build_options.enable_web) _ = @import("features/web/mod.zig");

    // Cloud module
    if (build_options.enable_cloud) _ = @import("features/cloud/mod.zig");

    // AI facade modules (inference, training, reasoning, ai_core)
    if (build_options.enable_ai)
        _ = @import("features/ai_core/mod.zig");
    if (@hasDecl(build_options, "enable_llm") and build_options.enable_llm)
        _ = @import("features/ai_inference/mod.zig");
    if (@hasDecl(build_options, "enable_training") and build_options.enable_training)
        _ = @import("features/ai_training/mod.zig");
    if (@hasDecl(build_options, "enable_reasoning") and build_options.enable_reasoning)
        _ = @import("features/ai_reasoning/mod.zig");

    // GPU and database modules (Zig 0.16 migration completed)
    if (build_options.enable_gpu) _ = @import("features/gpu/mod.zig");
    if (build_options.enable_database) _ = @import("features/database/mod.zig");

    if (build_options.enable_auth) _ = @import("features/auth/auth_test.zig");

    // MCP/ACP service tests (types + server only — mod.zig has database dep)
    _ = @import("services/mcp/server.zig");
    _ = @import("services/mcp/types.zig");
    _ = @import("services/acp/mod.zig");

    // Runtime concurrency and scheduling tests
    _ = @import("services/runtime/concurrency/channel.zig");
    _ = @import("services/runtime/scheduling/thread_pool.zig");
    _ = @import("services/runtime/scheduling/dag_pipeline.zig");

    // Shared utility inline tests
    _ = @import("services/shared/utils/swiss_map.zig");
    _ = @import("services/shared/utils/abix_serialize.zig");
    _ = @import("services/shared/utils/v2_primitives.zig");
    _ = @import("services/shared/utils/profiler.zig");
    _ = @import("services/shared/utils/structured_error.zig");
    _ = @import("services/shared/utils/memory/arena_pool.zig");
    _ = @import("services/shared/utils/memory/combinators.zig");
    _ = @import("services/shared/sync.zig");
    _ = @import("services/shared/tensor.zig");
    _ = @import("services/shared/matrix.zig");

    // SIMD module inline tests
    _ = @import("services/shared/simd/vector_ops.zig");
    _ = @import("services/shared/simd/distances.zig");
    _ = @import("services/shared/simd/activations.zig");
    _ = @import("services/shared/simd/integer_ops.zig");
    _ = @import("services/shared/simd/extras.zig");
    _ = @import("services/shared/utils/radix_tree.zig");

    // Connector inline tests (not discovered through named abi module)
    _ = @import("services/connectors/mod.zig");
    _ = @import("services/connectors/shared.zig");
    _ = @import("services/connectors/openai.zig");
    _ = @import("services/connectors/anthropic.zig");
    _ = @import("services/connectors/ollama.zig");
    _ = @import("services/connectors/huggingface.zig");
    _ = @import("services/connectors/mistral.zig");
    _ = @import("services/connectors/cohere.zig");
    _ = @import("services/connectors/lm_studio.zig");
    _ = @import("services/connectors/vllm.zig");
    _ = @import("services/connectors/mlx.zig");
    _ = @import("services/connectors/local_scheduler.zig");
    _ = @import("services/connectors/stub.zig");
    _ = @import("services/connectors/discord/mod.zig");
    _ = @import("services/connectors/discord/utils.zig");
    _ = @import("services/connectors/discord/rest_encoders.zig");

    // HA (high availability) sub-modules
    _ = @import("services/ha/mod.zig");
    _ = @import("services/ha/replication.zig");
    _ = @import("services/ha/backup.zig");
    _ = @import("services/ha/pitr.zig");
    _ = @import("services/ha/stub.zig");

    // Platform detection
    _ = @import("services/platform/mod.zig");

    // Additional shared utilities
    _ = @import("services/shared/utils/benchmark.zig");
    _ = @import("services/shared/time.zig");
    _ = @import("services/shared/utils/json/mod.zig");
    _ = @import("services/shared/utils/retry.zig");
    _ = @import("services/shared/logging.zig");
    _ = @import("services/shared/plugins.zig");
    _ = @import("services/shared/utils.zig");
    _ = @import("services/shared/platform.zig");

    // Shared memory utilities
    _ = @import("services/shared/utils/memory/aligned.zig");
    _ = @import("services/shared/utils/memory/stack.zig");
    _ = @import("services/shared/utils/memory/tracking.zig");
    _ = @import("services/shared/utils/memory/ring.zig");
    _ = @import("services/shared/utils/memory/pool.zig");
    _ = @import("services/shared/utils/memory/zerocopy.zig");

    // Runtime concurrency
    _ = @import("services/runtime/concurrency/mod.zig");
    _ = @import("services/runtime/concurrency/chase_lev.zig");
    _ = @import("services/runtime/concurrency/lockfree.zig");
    _ = @import("services/runtime/concurrency/mpmc_queue.zig");
    _ = @import("services/runtime/concurrency/priority_queue.zig");
    _ = @import("services/runtime/concurrency/epoch.zig");

    // Runtime scheduling
    _ = @import("services/runtime/scheduling/mod.zig");
    // async.zig spawns OS threads + Io.Threaded backend — hangs in test runner
    // _ = @import("services/runtime/scheduling/async.zig");
    _ = @import("services/runtime/scheduling/cancellation.zig");
    _ = @import("services/runtime/scheduling/future.zig");
    _ = @import("services/runtime/scheduling/task_group.zig");

    // Runtime engine + workload
    _ = @import("services/runtime/workload.zig");

    // Tasks subsystem
    _ = @import("services/tasks/types.zig");
    _ = @import("services/tasks/mod.zig");

    // Platform detection
    _ = @import("services/platform/detection.zig");

    // Core module tests (errors.zig transitively pulls in config/mod.zig)
    _ = @import("core/errors.zig");
    _ = @import("core/feature_catalog.zig");
    _ = @import("core/registry/mod.zig");
    _ = @import("core/registry/stub.zig");

    // vNext module inline tests
    _ = @import("vnext/capability.zig");
    _ = @import("vnext/config.zig");

    // Security sub-modules (16 modules, ~68 tests)
    _ = @import("services/shared/security/mod.zig");
    _ = @import("services/shared/security/csprng.zig");

    // Runtime engine: result_cache, steal_policy, numa + mod.zig for engine/types/benchmark tests
    _ = @import("services/runtime/engine/result_cache.zig");
    _ = @import("services/runtime/engine/steal_policy.zig");
    _ = @import("services/runtime/engine/numa.zig");
    _ = @import("services/runtime/engine/mod.zig");
    _ = @import("services/runtime/memory/mod.zig");

    // Additional shared utilities
    _ = @import("services/shared/errors.zig");
    _ = @import("services/shared/utils/binary.zig");
    _ = @import("services/shared/utils/encoding/mod.zig");
    _ = @import("services/shared/utils/config.zig");
    _ = @import("services/shared/utils/crypto/mod.zig");
    _ = @import("services/shared/utils/http/mod.zig");
    _ = @import("services/shared/utils/http/async_http.zig");
    _ = @import("services/shared/utils/fs/mod.zig");
    _ = @import("services/shared/utils/net/mod.zig");
    _ = @import("services/shared/utils/memory/thread_cache.zig");
    _ = @import("services/shared/os.zig");
    _ = @import("services/shared/simd/simd_test.zig");
    _ = @import("services/shared/resilience/circuit_breaker.zig");
    _ = @import("services/tasks/roadmap.zig");
}
