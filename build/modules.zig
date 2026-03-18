const std = @import("std");
const options_mod = @import("options.zig");
const BuildOptions = options_mod.BuildOptions;

/// Build the `build_options` module that source code imports as
/// `@import("build_options")`.  Feature flags are forwarded from the
/// `BuildOptions` struct, and GPU backend booleans are derived from the
/// selected backend list.  The package version is parsed from
/// `build.zig.zon` at comptime — no hardcoded fallback.
pub fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    return createBuildOptionsModuleWithVersion(b, options, parsePackageVersion());
}

/// Parse the package version from build.zig.zon at comptime.
/// Falls back to "0.0.0" if the version field is not found.
pub fn parsePackageVersion() []const u8 {
    const zon_bytes = @embedFile("../build.zig.zon");
    const marker = ".version = \"";
    const start_idx = std.mem.indexOf(u8, zon_bytes, marker) orelse return "0.0.0";
    const version_start = start_idx + marker.len;
    const end_idx = std.mem.indexOfScalarPos(u8, zon_bytes, version_start, '"') orelse return "0.0.0";
    return zon_bytes[version_start..end_idx];
}

/// Build the `build_options` module with an explicit package version.
/// Prefer this over `createBuildOptionsModule` when the caller has parsed
/// the authoritative version from `build.zig.zon`.
pub fn createBuildOptionsModuleWithVersion(
    b: *std.Build,
    options: BuildOptions,
    package_version: []const u8,
) *std.Build.Module {
    var opts = b.addOptions();
    opts.addOption([]const u8, "package_version", package_version);

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

/// Wire the `abi` module with its build-time dependencies.
/// In dev.2905+, all src/ files belong to the single `abi` module.
pub fn wireAbiImports(
    abi_module: *std.Build.Module,
    build_opts: *std.Build.Module,
) void {
    abi_module.addImport("build_options", build_opts);
}

/// Create the `cli` module that the CLI executable imports.
pub fn createCliModule(
    b: *std.Build,
    abi_module: *std.Build.Module,
    toolchain_support_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const cli = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cli.addImport("abi", abi_module);
    cli.addImport("toolchain_support", toolchain_support_module);
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
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    wireAbiImports(abi, build_opts);
    return abi;
}
