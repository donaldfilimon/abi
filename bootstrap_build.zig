const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.4.0");
    build_options.addOption(bool, "feat_ai", true);
    build_options.addOption(bool, "feat_database", true);
    build_options.addOption(bool, "feat_gpu", false);
    build_options.addOption(bool, "feat_web", false);
    build_options.addOption(bool, "feat_cloud", false);
    build_options.addOption(bool, "feat_storage", false);
    build_options.addOption(bool, "feat_network", true);
    build_options.addOption(bool, "feat_profiling", false);
    build_options.addOption(bool, "feat_benchmarks", false);
    build_options.addOption(bool, "feat_mobile", false);
    build_options.addOption(bool, "gpu_cuda", false);
    build_options.addOption(bool, "gpu_vulkan", false);
    build_options.addOption(bool, "gpu_stdgpu", false);
    build_options.addOption(bool, "gpu_metal", false);
    build_options.addOption(bool, "gpu_webgpu", false);
    build_options.addOption(bool, "gpu_opengl", false);
    build_options.addOption(bool, "gpu_opengles", false);
    build_options.addOption(bool, "gpu_gl_any", false);
    build_options.addOption(bool, "gpu_gl_desktop", false);
    build_options.addOption(bool, "gpu_gl_es", false);
    build_options.addOption(bool, "gpu_webgl2", false);
    build_options.addOption(bool, "gpu_fpga", false);
    build_options.addOption(bool, "gpu_tpu", false);

    const build_options_module = build_options.createModule();

    const abi = b.addModule("abi", .{
        .root_source_file = b.path("src/abi.zig"),
    });
    abi.addImport("build_options", build_options_module);

    const cli_module = b.addModule("cli", .{
        .root_source_file = b.path("tools/cli/mod.zig"),
    });
    cli_module.addImport("abi", abi);

    const exe = b.addExecutable(.{
        .name = "abi-bootstrap-cli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addImport("abi", abi);
    exe.root_module.addImport("cli", cli_module);
    exe.root_module.addImport("build_options", build_options_module);

    if (target.result.os.tag == .macos) {
        b.sysroot = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";
    }

    b.installArtifact(exe);
}
