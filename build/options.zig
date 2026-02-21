const std = @import("std");
const gpu = @import("gpu.zig");
const feature_catalog = @import("../src/core/feature_catalog.zig");
const GpuBackend = gpu.GpuBackend;

const internal_allowed_flags = [_][]const u8{ "enable_explore", "enable_vision" };

fn isCatalogFlag(comptime field_name: []const u8) bool {
    @setEvalBranchQuota(4096);
    inline for (feature_catalog.all) |entry| {
        if (std.mem.eql(u8, field_name, entry.compile_flag_field)) {
            return true;
        }
    }
    return false;
}

fn isAllowedInternalFlag(comptime field_name: []const u8) bool {
    inline for (internal_allowed_flags) |flag| {
        if (std.mem.eql(u8, field_name, flag)) {
            return true;
        }
    }
    return false;
}

pub const BuildOptions = struct {
    // Existing feature flags
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

    // New feature flags (v2)
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

    // GPU backend accessors
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

comptime {
    for (feature_catalog.all) |entry| {
        if (!@hasField(BuildOptions, entry.compile_flag_field)) {
            @compileError("BuildOptions missing compile flag field from feature catalog: " ++ entry.compile_flag_field);
        }
    }

    for (std.meta.fields(BuildOptions)) |field| {
        if (std.mem.eql(u8, field.name, "gpu_backends")) continue;
        if (std.mem.startsWith(u8, field.name, "enable_")) {
            if (!isCatalogFlag(field.name) and !isAllowedInternalFlag(field.name)) {
                @compileError("BuildOptions defines unknown feature flag: " ++ field.name);
            }
        } else if (!std.mem.eql(u8, field.name, "gpu_backends")) {
            @compileError("BuildOptions contains unexpected non-flag field: " ++ field.name);
        }
    }
}

pub fn readBuildOptions(
    b: *std.Build,
    target_os: std.Target.Os.Tag,
    target_abi: std.Target.Abi,
    can_link_metal: bool,
    backend_arg: ?[]const u8,
) BuildOptions {
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse true;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true;
    const enable_web = b.option(bool, "enable-web", "Enable web features") orelse true;

    return .{
        .enable_gpu = enable_gpu,
        .enable_ai = enable_ai,
        .enable_explore = b.option(bool, "enable-explore", "Enable AI code exploration") orelse enable_ai,
        .enable_llm = b.option(bool, "enable-llm", "Enable local LLM inference") orelse enable_ai,
        .enable_vision = b.option(bool, "enable-vision", "Enable vision/image processing") orelse enable_ai,
        .enable_web = enable_web,
        .enable_database = b.option(bool, "enable-database", "Enable database features") orelse true,
        .enable_network = b.option(bool, "enable-network", "Enable network distributed compute") orelse true,
        .enable_profiling = b.option(bool, "enable-profiling", "Enable profiling and metrics") orelse true,
        .enable_analytics = b.option(bool, "enable-analytics", "Enable analytics event tracking") orelse true,

        // New flags: cloud now decoupled from web
        .enable_cloud = b.option(bool, "enable-cloud", "Enable cloud provider integration") orelse true,
        .enable_training = b.option(bool, "enable-training", "Enable AI training pipelines") orelse enable_ai,
        .enable_reasoning = b.option(bool, "enable-reasoning", "Enable AI reasoning (Abbey, eval, RAG)") orelse enable_ai,
        .enable_auth = b.option(bool, "enable-auth", "Enable authentication and security") orelse true,
        .enable_messaging = b.option(bool, "enable-messaging", "Enable event bus and messaging") orelse true,
        .enable_cache = b.option(bool, "enable-cache", "Enable in-memory caching") orelse true,
        .enable_storage = b.option(bool, "enable-storage", "Enable unified file/object storage") orelse true,
        .enable_search = b.option(bool, "enable-search", "Enable full-text search") orelse true,
        .enable_mobile = b.option(bool, "enable-mobile", "Enable mobile target cross-compilation") orelse false,
        .enable_gateway = b.option(bool, "enable-gateway", "Enable API gateway (routing, rate limiting, circuit breaker)") orelse true,
        .enable_pages = b.option(bool, "enable-pages", "Enable dashboard/UI pages with routing") orelse true,
        .enable_benchmarks = b.option(bool, "enable-benchmarks", "Enable performance benchmarking module") orelse true,

        .gpu_backends = gpu.parseGpuBackends(
            b,
            backend_arg,
            enable_gpu,
            enable_web,
            target_os,
            target_abi,
            can_link_metal,
        ),
    };
}

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
