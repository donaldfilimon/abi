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

    // Forward every enable_* bool field from BuildOptions automatically.
    inline for (std.meta.fields(BuildOptions)) |field| {
        if (field.type == bool) {
            opts.addOption(bool, field.name, @field(options, field.name));
        }
    }

    // Export canonical feat_* aliases for migration to the v2 flag model.
    const canonical = options_mod.buildOptionsToCanonical(options);
    inline for (std.meta.fields(options_mod.CanonicalFlags)) |field| {
        opts.addOption(bool, field.name, @field(canonical, field.name));
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
    const abi = b.createModule(.{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi.addImport("build_options", build_opts);
    return abi;
}
