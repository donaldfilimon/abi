//! Integration tests for GPU policy contract validation.
//!
//! Verifies that the public GPU policy API is accessible through the
//! abi package and returns consistent results for the current platform.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "GPU policy module exports are accessible" {
    if (!build_options.feat_gpu) return error.SkipZigTest;

    const policy = abi.gpu.policy;
    // Verify core types exist
    _ = policy.PlatformClass;
    _ = policy.OptimizationHints;
    _ = policy.SelectionContext;
    _ = policy.BackendNameList;
}

test "GPU policy classifyBuiltin returns valid platform" {
    if (!build_options.feat_gpu) return error.SkipZigTest;

    const policy = abi.gpu.policy;
    const platform = policy.classifyBuiltin();

    // On any supported build host, classification should not be .unknown
    try std.testing.expect(platform != .unknown);
}

test "GPU policy resolveAutoBackendNames returns non-empty list" {
    if (!build_options.feat_gpu) return error.SkipZigTest;

    const policy = abi.gpu.policy;
    const platform = policy.classifyBuiltin();
    const names = policy.resolveAutoBackendNames(.{
        .platform = platform,
        .enable_gpu = true,
        .enable_web = false,
        .can_link_metal = (platform == .macos),
        .allow_simulated = true,
    });

    // Every supported platform should resolve at least one backend
    try std.testing.expect(names.slice().len > 0);
}

test "GPU policy optimization hints are valid" {
    if (!build_options.feat_gpu) return error.SkipZigTest;

    const policy = abi.gpu.policy;
    const platform = policy.classifyBuiltin();
    const h = policy.optimizationHintsForPlatform(platform);

    // Sanity checks on hint values
    try std.testing.expect(h.default_local_size > 0);
    try std.testing.expect(h.default_queue_depth > 0);
    try std.testing.expect(h.transfer_chunk_bytes > 0);
}
