//! Test entrypoint for ABI framework tests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Force-reference submodules to include their tests
comptime {
    // LLM module tests (when enabled)
    if (build_options.enable_llm) {
        _ = abi.ai.llm.io;
        _ = abi.ai.llm.tensor;
        _ = abi.ai.llm.tokenizer;
        _ = abi.ai.llm.ops;
        _ = abi.ai.llm.cache;
        _ = abi.ai.llm.model;
        _ = abi.ai.llm.generation;
    }
    // Explore module tests (when enabled)
    if (build_options.enable_explore) {
        _ = abi.ai.explore;
    }
    // Persona integration tests
    if (build_options.enable_ai) {
        _ = abi.ai.personas;
    }
    // Connector tests
    _ = @import("connectors_test.zig");
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
    _ = abi.shared.errors;
    // High Availability module tests
    _ = @import("ha_test.zig");
    // Stub parity verification tests (runtime checks)
    _ = @import("stub_parity.zig");
    // Comptime API parity tests (catches drift at compile time)
    _ = @import("parity/mod.zig");

    // End-to-end integration tests (issue #397)
    _ = @import("e2e_llm_test.zig");
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

    // Analytics module tests
    if (@hasDecl(build_options, "enable_analytics") and build_options.enable_analytics) {
        _ = abi.analytics;
        _ = @import("analytics_test.zig");
    }
}

// Connector tests
pub const connectors_test = @import("connectors_test.zig");

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
pub const e2e_llm_test = @import("e2e_llm_test.zig");
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

    var framework_instance = try abi.createDefaultFramework(gpa.allocator());
    defer framework_instance.deinit();

    try std.testing.expect(framework_instance.isRunning());
}

test "framework minimal initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_minimal = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = false,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = false,
    });
    defer framework_minimal.deinit();

    try std.testing.expect(framework_minimal.isRunning());
}

test "framework with gpu enabled" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = true,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = false,
    });
    defer framework.deinit();

    try std.testing.expect(framework.isEnabled(.gpu));
}

test "framework feature flags" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = build_options.enable_gpu,
        .enable_ai = build_options.enable_ai,
        .enable_web = build_options.enable_web,
        .enable_database = build_options.enable_database,
        .enable_network = build_options.enable_network,
        .enable_profiling = build_options.enable_profiling,
    });
    defer framework.deinit();

    try std.testing.expectEqual(build_options.enable_gpu, framework.isEnabled(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, framework.isEnabled(.ai));
    try std.testing.expectEqual(build_options.enable_web, framework.isEnabled(.web));
    try std.testing.expectEqual(build_options.enable_database, framework.isEnabled(.database));
    try std.testing.expectEqual(build_options.enable_network, framework.isEnabled(.network));
    try std.testing.expectEqual(build_options.enable_profiling, framework.isEnabled(.observability));
}
