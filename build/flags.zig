const std = @import("std");
const options_mod = @import("options.zig");
const modules = @import("modules.zig");
const feature_catalog = @import("../src/core/feature_catalog.zig");
const BuildOptions = options_mod.BuildOptions;

/// Compact flag combination for validation.  Sub-feature flags (explore, llm,
/// vision, training, reasoning) inherit from feat_ai when converted via
/// `comboToBuildOptions`.
pub const FlagCombo = struct {
    name: []const u8,
    feat_ai: bool = false,
    feat_llm: bool = false,
    feat_training: bool = false,
    feat_reasoning: bool = false,
    feat_gpu: bool = false,
    feat_web: bool = false,
    feat_database: bool = false,
    feat_network: bool = false,
    feat_profiling: bool = false,
    feat_analytics: bool = false,
    feat_cloud: bool = false,
    feat_auth: bool = false,
    feat_messaging: bool = false,
    feat_cache: bool = false,
    feat_storage: bool = false,
    feat_search: bool = false,
    feat_mobile: bool = false,
    feat_gateway: bool = false,
    feat_pages: bool = false,
    feat_benchmarks: bool = false,
    feat_compute: bool = false,
    feat_documents: bool = false,
    feat_desktop: bool = false,
};

// Validate FlagCombo against the feature catalog using shared helpers.
comptime {
    for (feature_catalog.all) |entry| {
        if (!@hasField(FlagCombo, entry.compile_flag_field))
            @compileError("FlagCombo missing catalog flag: " ++ entry.compile_flag_field);
    }
    for (std.meta.fields(FlagCombo)) |field| {
        if (std.mem.eql(u8, field.name, "name")) continue;
        if (!std.mem.startsWith(u8, field.name, "feat_"))
            @compileError("FlagCombo non-flag field: " ++ field.name);
        if (!options_mod.isCatalogFlag(field.name) and !options_mod.isAllowedInternalFlag(field.name))
            @compileError("FlagCombo unknown flag: " ++ field.name);
    }
}

/// Critical flag combinations that must compile.  Covers: all on, all off,
/// each feature solo, and each feature disabled with the rest enabled.
pub const validation_matrix = [_]FlagCombo{
    // All enabled / all disabled
    .{ .name = "all-enabled", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "all-disabled" },
    // Solo tests — one feature at a time
    .{ .name = "ai-only", .feat_ai = true },
    .{ .name = "gpu-only", .feat_gpu = true },
    .{ .name = "web-only", .feat_web = true },
    .{ .name = "database-only", .feat_database = true },
    .{ .name = "network-only", .feat_network = true },
    .{ .name = "profiling-only", .feat_profiling = true },
    .{ .name = "analytics-only", .feat_analytics = true },
    .{ .name = "cloud-only", .feat_cloud = true },
    .{ .name = "auth-only", .feat_auth = true },
    .{ .name = "messaging-only", .feat_messaging = true },
    .{ .name = "cache-only", .feat_cache = true },
    .{ .name = "storage-only", .feat_storage = true },
    .{ .name = "search-only", .feat_search = true },
    .{ .name = "gateway-only", .feat_gateway = true },
    .{ .name = "benchmarks-only", .feat_benchmarks = true },
    .{ .name = "pages-only", .feat_pages = true },
    .{ .name = "compute-only", .feat_compute = true },
    .{ .name = "documents-only", .feat_documents = true },
    .{ .name = "desktop-only", .feat_desktop = true },
    // No-X tests — everything except one
    .{ .name = "no-ai", .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-gpu", .feat_ai = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-web", .feat_ai = true, .feat_gpu = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-database", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-network", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-profiling", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-analytics", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-cloud", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-auth", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-messaging", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-cache", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-storage", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-search", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-gateway", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-pages", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-benchmarks", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_compute = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-compute", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_documents = true, .feat_desktop = true },
    .{ .name = "no-documents", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_desktop = true },
    .{ .name = "no-desktop", .feat_ai = true, .feat_gpu = true, .feat_web = true, .feat_database = true, .feat_network = true, .feat_profiling = true, .feat_analytics = true, .feat_cloud = true, .feat_auth = true, .feat_messaging = true, .feat_cache = true, .feat_storage = true, .feat_search = true, .feat_gateway = true, .feat_pages = true, .feat_benchmarks = true, .feat_compute = true, .feat_documents = true },
};

/// Convert a compact `FlagCombo` into a full `BuildOptions`.  Sub-feature
/// flags inherit from `feat_ai` when not explicitly set.
pub fn comboToBuildOptions(combo: FlagCombo) BuildOptions {
    const canonical: options_mod.CanonicalFlags = .{
        .feat_ai = combo.feat_ai,
        .feat_gpu = combo.feat_gpu,
        .feat_explore = combo.feat_ai,
        .feat_llm = combo.feat_ai or combo.feat_llm,
        .feat_vision = combo.feat_ai,
        .feat_web = combo.feat_web,
        .feat_database = combo.feat_database,
        .feat_network = combo.feat_network,
        .feat_profiling = combo.feat_profiling,
        .feat_analytics = combo.feat_analytics,
        .feat_cloud = combo.feat_cloud,
        .feat_training = combo.feat_ai or combo.feat_training,
        .feat_reasoning = combo.feat_ai or combo.feat_reasoning,
        .feat_auth = combo.feat_auth,
        .feat_messaging = combo.feat_messaging,
        .feat_cache = combo.feat_cache,
        .feat_storage = combo.feat_storage,
        .feat_search = combo.feat_search,
        .feat_gateway = combo.feat_gateway,
        .feat_pages = combo.feat_pages,
        .feat_benchmarks = combo.feat_benchmarks,
        .feat_mobile = combo.feat_mobile,
        .feat_compute = combo.feat_compute,
        .feat_documents = combo.feat_documents,
        .feat_desktop = combo.feat_desktop,
    };

    return options_mod.canonicalToBuildOptions(
        canonical,
        if (combo.feat_gpu) &.{.vulkan} else &.{},
    );
}

/// Register the "validate-flags" build step.  For each entry in the
/// validation matrix, two static libraries are compiled: one for the `abi`
/// module itself and one for the `stub_surface_check` that dereferences
/// every public symbol.
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

        // Deep symbol access under each combo catches mod/stub surface drift.
        const surface_mod = b.createModule(.{
            .root_source_file = b.path("build/validate/stub_surface_check.zig"),
            .target = target,
            .optimize = optimize,
        });
        surface_mod.addImport("abi", abi_mod);
        surface_mod.addImport("build_options", build_opts_mod);
        const surface_check = b.addLibrary(.{
            .name = "validate-surface-" ++ combo.name,
            .root_module = surface_mod,
            .linkage = .static,
        });
        validate_step.dependOn(&surface_check.step);
    }

    return validate_step;
}
