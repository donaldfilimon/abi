const std = @import("std");
const gpu = @import("gpu.zig");
const feature_catalog = @import("../src/core/feature_catalog.zig");
const GpuBackend = gpu.GpuBackend;

/// Internal-only flags that live in BuildOptions but are not in the feature
/// catalog (they are derived from parent flags like enable_ai).
pub const internal_allowed_flags = [_][]const u8{ "enable_explore", "enable_vision" };

/// Returns true when `field_name` matches any `compile_flag_field` in the
/// canonical feature catalog.
pub fn isCatalogFlag(comptime field_name: []const u8) bool {
    @setEvalBranchQuota(4096);
    inline for (feature_catalog.all) |entry| {
        if (std.mem.eql(u8, field_name, entry.compile_flag_field))
            return true;
    }
    return false;
}

/// Returns true when `field_name` is an internal (non-catalog) flag that is
/// still allowed in BuildOptions / FlagCombo.
pub fn isAllowedInternalFlag(comptime field_name: []const u8) bool {
    inline for (internal_allowed_flags) |flag| {
        if (std.mem.eql(u8, field_name, flag))
            return true;
    }
    return false;
}

/// Canonical feature flag model used internally by the build system.
///
/// CLI compatibility is handled by accepting both `-Dfeat-*` and `-Denable-*`
/// arguments in `readBuildOptions`.
pub const CanonicalFlags = struct {
    feat_gpu: bool,
    feat_ai: bool,
    feat_explore: bool,
    feat_llm: bool,
    feat_vision: bool,
    feat_web: bool,
    feat_database: bool,
    feat_network: bool,
    feat_profiling: bool,
    feat_analytics: bool,
    feat_cloud: bool,
    feat_training: bool,
    feat_reasoning: bool,
    feat_auth: bool,
    feat_messaging: bool,
    feat_cache: bool,
    feat_storage: bool,
    feat_search: bool,
    feat_mobile: bool,
    feat_gateway: bool,
    feat_pages: bool,
    feat_benchmarks: bool,
};

/// All compile-time build options for the ABI project.
///
/// Boolean fields prefixed with `enable_` are feature gates that select
/// between real (`mod.zig`) and stub implementations at compile time.
/// `gpu_backends` lists the GPU backends to compile support for.
pub const BuildOptions = struct {
    enable_gpu: bool,
    enable_ai: bool,
    enable_explore: bool,
    enable_llm: bool,
    enable_vision: bool,
    enable_web: bool,
    enable_database: bool,
    enable_network: bool,
    enable_profiling: bool,
    enable_analytics: bool,
    enable_cloud: bool,
    enable_training: bool,
    enable_reasoning: bool,
    enable_auth: bool,
    enable_messaging: bool,
    enable_cache: bool,
    enable_storage: bool,
    enable_search: bool,
    enable_mobile: bool,
    enable_gateway: bool,
    enable_pages: bool,
    enable_benchmarks: bool,

    gpu_backends: []const GpuBackend,

    pub fn hasGpuBackend(self: BuildOptions, backend: GpuBackend) bool {
        for (self.gpu_backends) |b| if (b == backend) return true;
        return false;
    }

    pub fn hasAnyGpuBackend(self: BuildOptions, backends: []const GpuBackend) bool {
        for (backends) |check| if (self.hasGpuBackend(check)) return true;
        return false;
    }

    pub fn gpu_cuda(self: BuildOptions) bool {
        return self.hasGpuBackend(.cuda);
    }
    pub fn gpu_vulkan(self: BuildOptions) bool {
        return self.hasGpuBackend(.vulkan);
    }
    pub fn gpu_stdgpu(self: BuildOptions) bool {
        return self.hasGpuBackend(.stdgpu);
    }
    pub fn gpu_metal(self: BuildOptions) bool {
        return self.hasGpuBackend(.metal);
    }
    pub fn gpu_webgpu(self: BuildOptions) bool {
        return self.hasGpuBackend(.webgpu);
    }
    pub fn gpu_opengl(self: BuildOptions) bool {
        return self.hasGpuBackend(.opengl);
    }
    pub fn gpu_opengles(self: BuildOptions) bool {
        return self.hasGpuBackend(.opengles);
    }
    pub fn gpu_webgl2(self: BuildOptions) bool {
        return self.hasGpuBackend(.webgl2);
    }
    pub fn gpu_fpga(self: BuildOptions) bool {
        return self.hasGpuBackend(.fpga);
    }
    pub fn gpu_tpu(self: BuildOptions) bool {
        return self.hasGpuBackend(.tpu);
    }
    pub fn gpu_gl_any(self: BuildOptions) bool {
        return self.gpu_opengl() or self.gpu_opengles();
    }
    pub fn gpu_gl_desktop(self: BuildOptions) bool {
        return self.gpu_opengl();
    }
    pub fn gpu_gl_es(self: BuildOptions) bool {
        return self.gpu_opengles();
    }
};

pub fn canonicalToBuildOptions(canonical: CanonicalFlags, gpu_backends: []const GpuBackend) BuildOptions {
    return .{
        .enable_gpu = canonical.feat_gpu,
        .enable_ai = canonical.feat_ai,
        .enable_explore = canonical.feat_explore,
        .enable_llm = canonical.feat_llm,
        .enable_vision = canonical.feat_vision,
        .enable_web = canonical.feat_web,
        .enable_database = canonical.feat_database,
        .enable_network = canonical.feat_network,
        .enable_profiling = canonical.feat_profiling,
        .enable_analytics = canonical.feat_analytics,
        .enable_cloud = canonical.feat_cloud,
        .enable_training = canonical.feat_training,
        .enable_reasoning = canonical.feat_reasoning,
        .enable_auth = canonical.feat_auth,
        .enable_messaging = canonical.feat_messaging,
        .enable_cache = canonical.feat_cache,
        .enable_storage = canonical.feat_storage,
        .enable_search = canonical.feat_search,
        .enable_mobile = canonical.feat_mobile,
        .enable_gateway = canonical.feat_gateway,
        .enable_pages = canonical.feat_pages,
        .enable_benchmarks = canonical.feat_benchmarks,
        .gpu_backends = gpu_backends,
    };
}

pub fn buildOptionsToCanonical(options: BuildOptions) CanonicalFlags {
    return .{
        .feat_gpu = options.enable_gpu,
        .feat_ai = options.enable_ai,
        .feat_explore = options.enable_explore,
        .feat_llm = options.enable_llm,
        .feat_vision = options.enable_vision,
        .feat_web = options.enable_web,
        .feat_database = options.enable_database,
        .feat_network = options.enable_network,
        .feat_profiling = options.enable_profiling,
        .feat_analytics = options.enable_analytics,
        .feat_cloud = options.enable_cloud,
        .feat_training = options.enable_training,
        .feat_reasoning = options.enable_reasoning,
        .feat_auth = options.enable_auth,
        .feat_messaging = options.enable_messaging,
        .feat_cache = options.enable_cache,
        .feat_storage = options.enable_storage,
        .feat_search = options.enable_search,
        .feat_mobile = options.enable_mobile,
        .feat_gateway = options.enable_gateway,
        .feat_pages = options.enable_pages,
        .feat_benchmarks = options.enable_benchmarks,
    };
}

fn readFeatureGate(
    b: *std.Build,
    comptime canonical_name: []const u8,
    comptime legacy_name: []const u8,
    description: []const u8,
    default_value: bool,
) bool {
    if (b.option(bool, canonical_name, description)) |value| return value;
    if (b.option(bool, legacy_name, description)) |value| return value;
    return default_value;
}

// ── Comptime validation ─────────────────────────────────────────────────
// Every catalog flag must exist in BuildOptions, and every enable_* field
// in BuildOptions must be in the catalog or the internal-allowed list.
comptime {
    for (feature_catalog.all) |entry| {
        if (!@hasField(BuildOptions, entry.compile_flag_field))
            @compileError("BuildOptions missing catalog flag: " ++ entry.compile_flag_field);
    }
    for (std.meta.fields(BuildOptions)) |field| {
        if (std.mem.eql(u8, field.name, "gpu_backends")) continue;
        if (std.mem.startsWith(u8, field.name, "enable_")) {
            if (!isCatalogFlag(field.name) and !isAllowedInternalFlag(field.name))
                @compileError("BuildOptions unknown flag: " ++ field.name);
        }
    }
}

/// Read all feature-flag build options from the Zig build CLI.
pub fn readBuildOptions(
    b: *std.Build,
    target_os: std.Target.Os.Tag,
    target_abi: std.Target.Abi,
    can_link_metal: bool,
    backend_arg: ?[]const u8,
) BuildOptions {
    const feat_gpu = readFeatureGate(b, "feat-gpu", "enable-gpu", "Enable GPU support", true);
    const feat_ai = readFeatureGate(b, "feat-ai", "enable-ai", "Enable AI features", true);
    const feat_web = readFeatureGate(b, "feat-web", "enable-web", "Enable web features", true);

    const canonical = CanonicalFlags{
        .feat_gpu = feat_gpu,
        .feat_ai = feat_ai,
        .feat_explore = readFeatureGate(b, "feat-explore", "enable-explore", "Enable AI code exploration", feat_ai),
        .feat_llm = readFeatureGate(b, "feat-llm", "enable-llm", "Enable local LLM inference", feat_ai),
        .feat_vision = readFeatureGate(b, "feat-vision", "enable-vision", "Enable vision/image processing", feat_ai),
        .feat_web = feat_web,
        .feat_database = readFeatureGate(b, "feat-database", "enable-database", "Enable database features", true),
        .feat_network = readFeatureGate(b, "feat-network", "enable-network", "Enable network distributed compute", true),
        .feat_profiling = readFeatureGate(b, "feat-profiling", "enable-profiling", "Enable profiling and metrics", true),
        .feat_analytics = readFeatureGate(b, "feat-analytics", "enable-analytics", "Enable analytics event tracking", true),
        .feat_cloud = readFeatureGate(b, "feat-cloud", "enable-cloud", "Enable cloud provider integration", true),
        .feat_training = readFeatureGate(b, "feat-training", "enable-training", "Enable AI training pipelines", feat_ai),
        .feat_reasoning = readFeatureGate(b, "feat-reasoning", "enable-reasoning", "Enable AI reasoning (Abbey, eval, RAG)", feat_ai),
        .feat_auth = readFeatureGate(b, "feat-auth", "enable-auth", "Enable authentication and security", true),
        .feat_messaging = readFeatureGate(b, "feat-messaging", "enable-messaging", "Enable event bus and messaging", true),
        .feat_cache = readFeatureGate(b, "feat-cache", "enable-cache", "Enable in-memory caching", true),
        .feat_storage = readFeatureGate(b, "feat-storage", "enable-storage", "Enable unified file/object storage", true),
        .feat_search = readFeatureGate(b, "feat-search", "enable-search", "Enable full-text search", true),
        .feat_mobile = readFeatureGate(b, "feat-mobile", "enable-mobile", "Enable mobile target cross-compilation", false),
        .feat_gateway = readFeatureGate(b, "feat-gateway", "enable-gateway", "Enable API gateway (routing, rate limiting, circuit breaker)", true),
        .feat_pages = readFeatureGate(b, "feat-pages", "enable-pages", "Enable dashboard/UI pages with routing", true),
        .feat_benchmarks = readFeatureGate(b, "feat-benchmarks", "enable-benchmarks", "Enable performance benchmarking module", true),
    };

    const gpu_backends = gpu.parseGpuBackends(
        b,
        backend_arg,
        canonical.feat_gpu,
        canonical.feat_web,
        target_os,
        target_abi,
        can_link_metal,
    );

    return canonicalToBuildOptions(canonical, gpu_backends);
}

/// Warn about conflicting or nonsensical flag combinations.
pub fn validateOptions(options: BuildOptions) void {
    const has_native = options.hasAnyGpuBackend(&.{ .cuda, .vulkan, .stdgpu, .metal, .opengl, .opengles });
    const has_web = options.hasAnyGpuBackend(&.{ .webgpu, .webgl2 });

    if (has_native and !options.enable_gpu)
        std.log.err("GPU backends enabled but enable-gpu=false", .{});
    if (has_web and !options.enable_web)
        std.log.err("Web GPU backends enabled but enable-web=false", .{});
    if (options.hasGpuBackend(.cuda) and options.hasGpuBackend(.vulkan))
        std.log.warn("Both CUDA and Vulkan backends enabled; may cause conflicts", .{});
    if (options.hasGpuBackend(.opengl) and options.hasGpuBackend(.webgl2))
        std.log.warn("Both OpenGL and WebGL2 enabled; prefer one", .{});
}
