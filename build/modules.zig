const std = @import("std");
const options_mod = @import("options.zig");
const BuildOptions = options_mod.BuildOptions;

/// Build the `build_options` module that source code imports as
/// `@import("build_options")`.  Feature flags are forwarded from the
/// `BuildOptions` struct, and GPU backend booleans are derived from the
/// selected backend list.
pub fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var opts = b.addOptions();
    opts.addOption([]const u8, "package_version", "0.4.0");

    // Forward every feat_* bool field from BuildOptions automatically.
    inline for (std.meta.fields(BuildOptions)) |field| {
        if (field.type == bool) {
            opts.addOption(bool, field.name, @field(options, field.name));
        }
    }

    // GPU backend convenience flags (derived from the backend list).
    opts.addOption(bool, "gpu_cuda", options.gpu_cuda());
    opts.addOption(bool, "gpu_vulkan", options.gpu_vulkan());
    opts.addOption(bool, "gpu_stdgpu", options.gpu_stdgpu());
    opts.addOption(bool, "gpu_metal", options.gpu_metal());
    opts.addOption(bool, "gpu_webgpu", options.gpu_webgpu());
    opts.addOption(bool, "gpu_opengl", options.gpu_opengl());
    opts.addOption(bool, "gpu_opengles", options.gpu_opengles());
    opts.addOption(bool, "gpu_gl_any", options.gpu_gl_any());
    opts.addOption(bool, "gpu_gl_desktop", options.gpu_gl_desktop());
    opts.addOption(bool, "gpu_gl_es", options.gpu_gl_es());
    opts.addOption(bool, "gpu_webgl2", options.gpu_webgl2());
    opts.addOption(bool, "gpu_fpga", options.gpu_fpga());
    opts.addOption(bool, "gpu_tpu", options.gpu_tpu());

    return opts.createModule();
}

pub fn createSharedServicesModule(
    b: *std.Build,
    build_opts: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const shared_services = b.createModule(.{
        .root_source_file = b.path("src/services/shared/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    shared_services.addImport("build_options", build_opts);
    return shared_services;
}

pub fn createCoreModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_opts: *std.Build.Module,
) *std.Build.Module {
    const core = b.createModule(.{
        .root_source_file = b.path("src/core/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    core.addImport("build_options", build_opts);
    return core;
}

pub fn wireAbiImports(
    abi_module: *std.Build.Module,
    build_opts: *std.Build.Module,
    shared_services: *std.Build.Module,
    core_module: *std.Build.Module,
) void {
    abi_module.addImport("build_options", build_opts);
    abi_module.addImport("shared_services", shared_services);
    abi_module.addImport("core", core_module);
}

/// Create the `cli` module that the CLI executable imports.
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

/// Create a standalone `abi` module with its own build-options (used by
/// profile builds and other targets that need different options from the
/// default).
pub fn createAbiModule(
    b: *std.Build,
    options: BuildOptions,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const build_opts = createBuildOptionsModule(b, options);
    const shared_services = createSharedServicesModule(b, build_opts, target, optimize);
    const core = createCoreModule(b, target, optimize, build_opts);
    const abi = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    wireAbiImports(abi, build_opts, shared_services, core);
    return abi;
}
