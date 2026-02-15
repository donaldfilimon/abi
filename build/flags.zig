const std = @import("std");
const options_mod = @import("options.zig");
const gpu_mod = @import("gpu.zig");
const modules = @import("modules.zig");
const BuildOptions = options_mod.BuildOptions;

/// Compact flag combination for validation. Sub-feature flags (explore, llm,
/// vision, training, reasoning) inherit from enable_ai.
pub const FlagCombo = struct {
    name: []const u8,
    enable_ai: bool = false,
    enable_gpu: bool = false,
    enable_web: bool = false,
    enable_database: bool = false,
    enable_network: bool = false,
    enable_profiling: bool = false,
    enable_analytics: bool = false,
    enable_cloud: bool = false,
    enable_auth: bool = false,
    enable_messaging: bool = false,
    enable_cache: bool = false,
    enable_storage: bool = false,
    enable_search: bool = false,
    enable_gateway: bool = false,
    enable_pages: bool = false,
    enable_benchmarks: bool = false,
};

/// Critical flag combinations that must compile. Covers: all on, all off,
/// each feature solo, and each feature disabled with the rest enabled.
pub const validation_matrix = [_]FlagCombo{
    // All enabled / all disabled
    .{ .name = "all-enabled", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "all-disabled" },
    // Solo tests
    .{ .name = "ai-only", .enable_ai = true },
    .{ .name = "gpu-only", .enable_gpu = true },
    .{ .name = "web-only", .enable_web = true },
    .{ .name = "database-only", .enable_database = true },
    .{ .name = "network-only", .enable_network = true },
    .{ .name = "profiling-only", .enable_profiling = true },
    .{ .name = "analytics-only", .enable_analytics = true },
    .{ .name = "cloud-only", .enable_cloud = true },
    .{ .name = "auth-only", .enable_auth = true },
    .{ .name = "messaging-only", .enable_messaging = true },
    .{ .name = "cache-only", .enable_cache = true },
    .{ .name = "storage-only", .enable_storage = true },
    .{ .name = "search-only", .enable_search = true },
    .{ .name = "gateway-only", .enable_gateway = true },
    .{ .name = "benchmarks-only", .enable_benchmarks = true },
    .{ .name = "pages-only", .enable_pages = true },
    // No-X tests (everything except one)
    .{ .name = "no-ai", .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-gpu", .enable_ai = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-web", .enable_ai = true, .enable_gpu = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-database", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-network", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-profiling", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-analytics", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-cloud", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-auth", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-messaging", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-cache", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-storage", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_search = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-search", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_gateway = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-gateway", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_pages = true, .enable_benchmarks = true },
    .{ .name = "no-pages", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_benchmarks = true },
    .{ .name = "no-benchmarks", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true, .enable_cloud = true, .enable_auth = true, .enable_messaging = true, .enable_cache = true, .enable_storage = true, .enable_search = true, .enable_gateway = true, .enable_pages = true },
};

pub fn comboToBuildOptions(combo: FlagCombo) BuildOptions {
    return .{
        .enable_ai = combo.enable_ai,
        .enable_gpu = combo.enable_gpu,
        .enable_explore = combo.enable_ai,
        .enable_llm = combo.enable_ai,
        .enable_vision = combo.enable_ai,
        .enable_web = combo.enable_web,
        .enable_database = combo.enable_database,
        .enable_network = combo.enable_network,
        .enable_profiling = combo.enable_profiling,
        .enable_analytics = combo.enable_analytics,
        .enable_cloud = combo.enable_cloud,
        .enable_training = combo.enable_ai,
        .enable_reasoning = combo.enable_ai,
        .enable_auth = combo.enable_auth,
        .enable_messaging = combo.enable_messaging,
        .enable_cache = combo.enable_cache,
        .enable_storage = combo.enable_storage,
        .enable_search = combo.enable_search,
        .enable_gateway = combo.enable_gateway,
        .enable_pages = combo.enable_pages,
        .enable_benchmarks = combo.enable_benchmarks,
        .enable_mobile = false,
        .gpu_backends = if (combo.enable_gpu) &.{.vulkan} else &.{},
    };
}

pub fn addFlagValidation(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const validate_step = b.step(
        "validate-flags",
        "Compile with every critical feature-flag combination",
    );

    inline for (validation_matrix) |combo| {
        const opts = comboToBuildOptions(combo);
        const build_opts_mod = modules.createBuildOptionsModule(b, opts);
        const abi_mod = b.createModule(.{
            .root_source_file = b.path("src/abi.zig"),
            .target = target,
            .optimize = optimize,
        });
        abi_mod.addImport("build_options", build_opts_mod);

        const check = b.addLibrary(.{
            .name = "validate-" ++ combo.name,
            .root_module = abi_mod,
            .linkage = .static,
        });
        validate_step.dependOn(&check.step);
    }

    return validate_step;
}
