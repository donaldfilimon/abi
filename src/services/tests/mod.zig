//! Test entrypoint for ABI framework tests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Force-reference submodules to include their tests.
// NOTE: Must use `test {}` blocks (not `comptime {}`). In Zig 0.16, `comptime {}` forces
// compilation but does NOT include test blocks in the test runner. Only `test {}` blocks
// trigger true test discovery for referenced modules.
test {
    // LLM module tests (when enabled)
    if (build_options.enable_llm) {
        _ = abi.features.ai.llm.io;
        _ = abi.features.ai.llm.tensor;
        _ = abi.features.ai.llm.tokenizer;
        _ = abi.features.ai.llm.ops;
        _ = abi.features.ai.llm.cache;
        _ = abi.features.ai.llm.model;
        _ = abi.features.ai.llm.generation;
    }
    // Explore module tests (when enabled)
    if (build_options.enable_explore) {
        _ = abi.features.ai.explore;
    }
    // Persona integration tests
    if (build_options.enable_ai) {
        _ = abi.features.ai.personas;
    }
    // AI submodule tests (previously undiscovered)
    if (build_options.enable_ai) {
        _ = abi.features.ai.eval;
        _ = abi.features.ai.rag;
        _ = abi.features.ai.templates;
        _ = abi.features.ai.memory;
        _ = abi.features.ai.orchestration;
        _ = abi.features.ai.tools;
        _ = abi.features.ai.streaming;
        _ = abi.features.ai.documents;
        _ = abi.features.ai.abbey;
        _ = abi.features.ai.database;
    }
    if (@hasDecl(build_options, "enable_vision") and build_options.enable_vision) {
        _ = abi.features.ai.vision;
    }
    // Connector tests
    _ = @import("connectors_test.zig");
    // Connector integration tests (isAvailable consistency, boundary conditions)
    _ = @import("connector_integration_test.zig");
    // MCP/ACP service tests (force test discovery through abi module)
    _ = abi.services.mcp;
    _ = abi.services.acp;
    // Integration test matrix
    _ = @import("test_matrix.zig");
    // Include training demo test
    _ = @import("training_demo.zig");
    // LLM reference vectors for llama-cpp compatibility
    if (build_options.enable_llm) {
        _ = @import("llm_reference_vectors.zig");
    }
    // Cross-platform OS features tests
    _ = @import("os_test.zig");
    // Shared utilities tests (via abi module)
    _ = abi.services.shared.errors;
    // High Availability module tests
    _ = @import("ha_test.zig");
    // Stub parity verification tests (runtime checks)
    _ = @import("stub_parity.zig");
    // Comptime API parity tests (catches drift at compile time)
    _ = @import("parity/mod.zig");
    // End-to-end integration tests (issue #397)
    if (build_options.enable_ai) {
        _ = @import("e2e_llm_test.zig");
    }
    _ = @import("e2e_database_test.zig");
    _ = @import("e2e_personas_test.zig");
    _ = @import("error_handling_test.zig");
    // KernelRing fast‑path test
    _ = @import("kernel_ring_test.zig");

    // Network module comprehensive tests
    if (build_options.enable_network) {
        _ = @import("network_test.zig");
    }

    // Cloud adapter tests
    if (build_options.enable_web) {
        _ = @import("cloud_test.zig");
    }

    // Web module tests (handlers, routes, context)
    if (build_options.enable_web) {
        _ = @import("web_test.zig");
    }

    // Cross-module integration tests
    _ = @import("integration_test.zig");

    // Concurrency stress tests (lock-free primitives)
    _ = @import("concurrency_stress_test.zig");

    // Quantized kernel correctness tests (CPU reference implementations)
    _ = @import("quantized_kernels_test.zig");

    // Integration test infrastructure (fixtures, mocks, cross-module tests)
    _ = @import("integration/mod.zig");

    // Observability module comprehensive tests
    if (build_options.enable_profiling) {
        _ = @import("observability_test.zig");
        _ = @import("observability_alerting_test.zig");
        _ = @import("observability_metrics_test.zig");
        _ = @import("observability_edge_test.zig");
        _ = @import("observability_tracing_test.zig");
    }

    // Stress test infrastructure (HA, observability, database stress tests)
    _ = @import("stress/mod.zig");

    // Chaos testing framework (production-grade reliability testing)
    _ = @import("chaos/mod.zig");

    // Property-based testing infrastructure (comprehensive property tests)
    _ = @import("property/mod.zig");

    // E2E workflow tests (complete user workflow validation)
    _ = @import("e2e/mod.zig");

    // Common test helpers and utilities
    _ = @import("helpers.zig");

    // v2 module integration tests (SwissMap, ArenaPool, Channel, ThreadPool, DagPipeline)
    _ = @import("v2_integration_test.zig");

    // SIMD kernel validation (v2 kernels vs scalar reference)
    _ = @import("simd_validation_test.zig");

    // Analytics module tests
    if (@hasDecl(build_options, "enable_analytics") and build_options.enable_analytics) {
        _ = abi.features.analytics;
        _ = @import("analytics_test.zig");
    }

    // AI math correctness tests (Phase 5B)
    if (build_options.enable_ai) {
        _ = @import("ai_eval_test.zig");
    }
    if (build_options.enable_llm) {
        _ = @import("ai_sampler_test.zig");
        _ = @import("ai_quantization_test.zig");
        _ = @import("ai_attention_test.zig");
    }

    // AI state machine & resilience tests (Phase 5C)
    if (build_options.enable_ai) {
        _ = @import("ai_streaming_test.zig");
        _ = @import("ai_memory_test.zig");
        _ = @import("ai_rag_test.zig");
    }

    // AI secondary coverage tests (Phase 5D)
    if (build_options.enable_ai) {
        _ = @import("ai_orchestration_test.zig");
        _ = @import("ai_templates_test.zig");
        _ = @import("ai_tools_test.zig");
    }

    // Multi-agent integration tests (coordinator, circuit breaker, mailbox)
    if (build_options.enable_ai) {
        _ = @import("multi_agent_test.zig");
    }

    // Non-AI gap coverage tests (Phase 5E)
    if (build_options.enable_database) {
        _ = @import("database_batch_test.zig");
    }
    if (build_options.enable_gpu) {
        _ = @import("gpu_dispatcher_test.zig");
    }
    if (build_options.enable_network) {
        _ = @import("network_raft_test.zig");
    }
}

// Connector tests
pub const connectors_test = @import("connectors_test.zig");

// Connector integration tests (isAvailable consistency, boundary conditions)
pub const connector_integration_test = @import("connector_integration_test.zig");

// Integration test matrix
pub const test_matrix = @import("test_matrix.zig");
pub const TestMatrix = test_matrix.TestMatrix;

// Legacy property-based testing (prefer property/mod.zig for new tests)
pub const proptest = @import("proptest.zig");

// Common test helpers and utilities
pub const helpers = @import("helpers.zig");

// Cross-platform test utilities
pub const platform = @import("platform.zig");

// Cross-platform OS features tests
pub const os_test = @import("os_test.zig");

// LLM reference vectors for llama-cpp compatibility testing
pub const llm_reference_vectors = if (build_options.enable_llm) @import("llm_reference_vectors.zig") else struct {};

// End-to-end integration tests (issue #397)
pub const e2e_llm_test = if (build_options.enable_ai) @import("e2e_llm_test.zig") else struct {};
pub const e2e_database_test = @import("e2e_database_test.zig");
pub const e2e_personas_test = @import("e2e_personas_test.zig");
pub const error_handling_test = @import("error_handling_test.zig");

// Cross-module integration tests
pub const integration_test = @import("integration_test.zig");

// Concurrency stress tests (64+ thread high-contention scenarios)
pub const concurrency_stress_test = @import("concurrency_stress_test.zig");

// Quantized kernel correctness tests (CPU reference implementations)
pub const quantized_kernels_test = @import("quantized_kernels_test.zig");

// Integration test infrastructure
pub const integration = @import("integration/mod.zig");

// Observability module comprehensive tests
pub const observability_test = if (build_options.enable_profiling) @import("observability_test.zig") else struct {};
pub const observability_alerting_test = if (build_options.enable_profiling) @import("observability_alerting_test.zig") else struct {};
pub const observability_metrics_test = if (build_options.enable_profiling) @import("observability_metrics_test.zig") else struct {};
pub const observability_edge_test = if (build_options.enable_profiling) @import("observability_edge_test.zig") else struct {};
pub const observability_tracing_test = if (build_options.enable_profiling) @import("observability_tracing_test.zig") else struct {};

// Stress test infrastructure (production-grade stress tests)
pub const stress = @import("stress/mod.zig");

// Chaos testing framework (production-grade reliability testing)
pub const chaos = @import("chaos/mod.zig");

// Property-based testing infrastructure (comprehensive property tests)
pub const property = @import("property/mod.zig");

// E2E workflow tests (complete user workflow validation)
pub const e2e = @import("e2e/mod.zig");

// Comptime API parity verification (stub/real module consistency)
pub const parity = @import("parity/mod.zig");

// AI math correctness tests (Phase 5B)
pub const ai_eval_test = if (build_options.enable_ai) @import("ai_eval_test.zig") else struct {};
pub const ai_sampler_test = if (build_options.enable_llm) @import("ai_sampler_test.zig") else struct {};
pub const ai_quantization_test = if (build_options.enable_llm) @import("ai_quantization_test.zig") else struct {};
pub const ai_attention_test = if (build_options.enable_llm) @import("ai_attention_test.zig") else struct {};

// AI state machine & resilience tests (Phase 5C)
pub const ai_streaming_test = if (build_options.enable_ai) @import("ai_streaming_test.zig") else struct {};
pub const ai_memory_test = if (build_options.enable_ai) @import("ai_memory_test.zig") else struct {};
pub const ai_rag_test = if (build_options.enable_ai) @import("ai_rag_test.zig") else struct {};

// AI secondary coverage tests (Phase 5D)
pub const ai_orchestration_test = if (build_options.enable_ai) @import("ai_orchestration_test.zig") else struct {};
pub const ai_templates_test = if (build_options.enable_ai) @import("ai_templates_test.zig") else struct {};
pub const ai_tools_test = if (build_options.enable_ai) @import("ai_tools_test.zig") else struct {};

// Multi-agent integration tests (coordinator, circuit breaker, mailbox)
pub const multi_agent_test = if (build_options.enable_ai) @import("multi_agent_test.zig") else struct {};

// Non-AI gap coverage tests (Phase 5E)
pub const database_batch_test = if (build_options.enable_database) @import("database_batch_test.zig") else struct {};
pub const gpu_dispatcher_test = if (build_options.enable_gpu) @import("gpu_dispatcher_test.zig") else struct {};
pub const network_raft_test = if (build_options.enable_network) @import("network_raft_test.zig") else struct {};

test "abi version returns build package version" {
    try std.testing.expectEqualStrings("0.4.0", abi.version());
}

test "abi exports required symbols" {
    _ = abi;

    std.testing.refAllDecls(abi);

    if (@hasDecl(abi, "GpuDevice")) {
        std.log.info("GPU module symbols found", .{});
    } else {
        std.log.info("GPU module not compiled or disabled", .{});
    }

    if (@hasDecl(abi, "HnswIndex")) {
        std.log.info("HNSW module symbols found", .{});
    }
}

test "framework initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_instance = try abi.App.initDefault(gpa.allocator());
    defer framework_instance.deinit();

    try std.testing.expect(framework_instance.isRunning());
}

test "framework minimal initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_minimal = try abi.App.initDefault(gpa.allocator());
    defer framework_minimal.deinit();

    try std.testing.expect(framework_minimal.isRunning());
}

test "framework with gpu enabled" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.App.initDefault(gpa.allocator());
    defer framework.deinit();

    try std.testing.expect(framework.isEnabled(.gpu));
}

test "framework feature flags" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.App.initDefault(gpa.allocator());
    defer framework.deinit();

    try std.testing.expectEqual(build_options.enable_gpu, framework.isEnabled(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, framework.isEnabled(.ai));
    try std.testing.expectEqual(build_options.enable_web, framework.isEnabled(.web));
    try std.testing.expectEqual(build_options.enable_database, framework.isEnabled(.database));
    try std.testing.expectEqual(build_options.enable_network, framework.isEnabled(.network));
    try std.testing.expectEqual(build_options.enable_profiling, framework.isEnabled(.observability));
}
