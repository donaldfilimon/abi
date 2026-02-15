const std = @import("std");
const options_mod = @import("options.zig");
const BuildOptions = options_mod.BuildOptions;

pub fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var opts = b.addOptions();
    opts.addOption([]const u8, "package_version", "0.4.0");

    // Existing flags
    opts.addOption(bool, "enable_gpu", options.enable_gpu);
    opts.addOption(bool, "enable_ai", options.enable_ai);
    opts.addOption(bool, "enable_explore", options.enable_explore);
    opts.addOption(bool, "enable_llm", options.enable_llm);
    opts.addOption(bool, "enable_vision", options.enable_vision);
    opts.addOption(bool, "enable_web", options.enable_web);
    opts.addOption(bool, "enable_database", options.enable_database);
    opts.addOption(bool, "enable_network", options.enable_network);
    opts.addOption(bool, "enable_profiling", options.enable_profiling);
    opts.addOption(bool, "enable_analytics", options.enable_analytics);

    // New v2 flags
    opts.addOption(bool, "enable_cloud", options.enable_cloud);
    opts.addOption(bool, "enable_training", options.enable_training);
    opts.addOption(bool, "enable_reasoning", options.enable_reasoning);
    opts.addOption(bool, "enable_auth", options.enable_auth);
    opts.addOption(bool, "enable_messaging", options.enable_messaging);
    opts.addOption(bool, "enable_cache", options.enable_cache);
    opts.addOption(bool, "enable_storage", options.enable_storage);
    opts.addOption(bool, "enable_search", options.enable_search);
    opts.addOption(bool, "enable_mobile", options.enable_mobile);
    opts.addOption(bool, "enable_gateway", options.enable_gateway);
    opts.addOption(bool, "enable_pages", options.enable_pages);
    opts.addOption(bool, "enable_benchmarks", options.enable_benchmarks);

    // GPU backend flags
    opts.addOption(bool, "gpu_cuda", options.gpu_cuda());
    opts.addOption(bool, "gpu_vulkan", options.gpu_vulkan());
    opts.addOption(bool, "gpu_stdgpu", options.gpu_stdgpu());
    opts.addOption(bool, "gpu_metal", options.gpu_metal());
    opts.addOption(bool, "gpu_webgpu", options.gpu_webgpu());
    opts.addOption(bool, "gpu_opengl", options.gpu_opengl());
    opts.addOption(bool, "gpu_opengles", options.gpu_opengles());
    opts.addOption(bool, "gpu_webgl2", options.gpu_webgl2());
    opts.addOption(bool, "gpu_fpga", options.gpu_fpga());

    return opts.createModule();
}

pub fn createCliModule(
    b: *std.Build,
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const cli = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli.addImport("abi", abi_module);
    return cli;
}

pub fn createAbiModule(
    b: *std.Build,
    options: BuildOptions,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const build_opts = createBuildOptionsModule(b, options);
    const abi = b.createModule(.{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi.addImport("build_options", build_opts);
    return abi;
}
