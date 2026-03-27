const std = @import("std");

pub const FeatureFlags = struct {
    feat_gpu: bool,
    feat_ai: bool,
    feat_database: bool,
    feat_network: bool,
    feat_observability: bool,
    feat_web: bool,
    feat_pages: bool,
    feat_analytics: bool,
    feat_cloud: bool,
    feat_auth: bool,
    feat_messaging: bool,
    feat_cache: bool,
    feat_storage: bool,
    feat_search: bool,
    feat_mobile: bool,
    feat_gateway: bool,
    feat_benchmarks: bool,
    feat_compute: bool,
    feat_documents: bool,
    feat_desktop: bool,
    feat_tui: bool,
    feat_llm: bool,
    feat_training: bool,
    feat_vision: bool,
    feat_explore: bool,
    feat_reasoning: bool,
    feat_lsp: bool,
    feat_mcp: bool,
    feat_acp: bool,
    feat_ha: bool,
    feat_connectors: bool,
    feat_tasks: bool,
    feat_inference: bool,
    gpu_metal: bool,
    gpu_cuda: bool,
    gpu_vulkan: bool,
    gpu_webgpu: bool,
    gpu_opengl: bool,
    gpu_opengles: bool,
    gpu_webgl2: bool,
    gpu_stdgpu: bool,
    gpu_fpga: bool,
    gpu_tpu: bool,
};

pub fn hasBackend(backend_str: ?[]const u8, name: []const u8) bool {
    const str = backend_str orelse return false;
    var it = std.mem.splitScalar(u8, str, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (std.mem.eql(u8, trimmed, name)) return true;
    }
    return false;
}

pub fn addAllBuildOptions(opts: *std.Build.Step.Options, flags: FeatureFlags, package_version: []const u8) void {
    inline for (@typeInfo(FeatureFlags).@"struct".fields) |field| {
        opts.addOption(bool, field.name, @field(flags, field.name));
    }
    opts.addOption([]const u8, "package_version", package_version);
}
