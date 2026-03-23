//! Integration Tests: CLI
//!
//! Tests CLI data paths through the abi public API.
//! Validates that the data sources used by CLI commands
//! (version, doctor, info, chat) are accessible and functional.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// === Version Command Path ===

test "cli: package version is non-empty" {
    const version = abi.meta.package_version;
    try std.testing.expect(version.len > 0);
}

test "cli: version function returns package version" {
    const v1 = abi.version();
    const v2 = abi.meta.version();
    try std.testing.expectEqualStrings(v1, v2);
}

// === Doctor Command Path ===

test "cli: build_options feature flags are accessible" {
    // Verify all feature flags the doctor command prints are readable
    const flags = .{
        build_options.feat_ai,
        build_options.feat_gpu,
        build_options.feat_database,
        build_options.feat_network,
        build_options.feat_web,
        build_options.feat_search,
        build_options.feat_cache,
        build_options.feat_auth,
        build_options.feat_lsp,
        build_options.feat_mcp,
        build_options.feat_mobile,
        build_options.feat_desktop,
    };
    // Each flag is a bool — just verify they don't crash when accessed
    inline for (flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

test "cli: AI sub-feature flags are accessible" {
    const ai_flags = .{
        build_options.feat_llm,
        build_options.feat_training,
        build_options.feat_vision,
        build_options.feat_reasoning,
    };
    inline for (ai_flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

test "cli: GPU backend flags are accessible" {
    const gpu_flags = .{
        build_options.gpu_metal,
        build_options.gpu_cuda,
        build_options.gpu_vulkan,
        build_options.gpu_stdgpu,
    };
    inline for (gpu_flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

// === Info Command Path ===

test "cli: feature catalog has entries" {
    const catalog = abi.meta.features;
    try std.testing.expect(catalog.all.len > 0);
    try std.testing.expect(catalog.feature_count > 20);
}

test "cli: feature catalog contains expected features" {
    const catalog = abi.meta.features;
    // Verify key features exist in catalog via the description free function
    try std.testing.expectEqualStrings("GPU acceleration and compute", catalog.description(.gpu));
    try std.testing.expectEqualStrings("AI core functionality", catalog.description(.ai));
    try std.testing.expectEqualStrings("Vector database (WDBX)", catalog.description(.database));
}

test "cli: feature catalog metadata is consistent" {
    const catalog = abi.meta.features;
    for (catalog.all, 0..) |entry, idx| {
        // Each feature's enum ordinal matches its position in the array
        try std.testing.expectEqual(entry.feature, @as(catalog.Feature, @enumFromInt(idx)));
        // Description is non-empty
        try std.testing.expect(entry.description.len > 0);
        // Compile flag field is non-empty
        try std.testing.expect(entry.compile_flag_field.len > 0);
        // Module paths are non-empty
        try std.testing.expect(entry.real_module_path.len > 0);
        try std.testing.expect(entry.stub_module_path.len > 0);
    }
}

// === Chat Command Path ===

test "cli: persona router routes messages" {
    const persona = abi.ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("What is machine learning?");
    // Should produce a valid routing decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
    try std.testing.expect(decision.reason.len > 0);
}

test "cli: routing decision has valid weights" {
    const persona = abi.ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("Help me debug this code");
    // Each weight should be in valid range
    try std.testing.expect(decision.weights.abbey >= 0.0);
    try std.testing.expect(decision.weights.abbey <= 1.0);
    try std.testing.expect(decision.weights.aviva >= 0.0);
    try std.testing.expect(decision.weights.aviva <= 1.0);
    try std.testing.expect(decision.weights.abi >= 0.0);
    try std.testing.expect(decision.weights.abi <= 1.0);
}

// === App Builder Path ===

test "cli: app version returns non-empty string" {
    const version = abi.app.version();
    try std.testing.expect(version.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
