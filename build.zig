const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_ansi = b.option(bool, "enable-ansi", "Enable ANSI formatting in CLI output") orelse true;
    const strict_io = b.option(bool, "strict-io", "Exit on first I/O error encountered") orelse false;
    const experimental = b.option(bool, "experimental", "Enable experimental feature set") orelse false;

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.1.0");
    build_options.addOption(bool, "enable_ansi", enable_ansi);
    build_options.addOption(bool, "strict_io", strict_io);
    build_options.addOption(bool, "experimental", experimental);

    const abi_module = b.createModule(.{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addOptions("build_options", build_options);

    const lib = b.addLibrary(.{
        .name = "abi",
        .root_module = abi_module,
        .linkage = .static,
    });
    b.installArtifact(lib);

    const cli_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_module.addImport("abi", abi_module);
    cli_module.addOptions("build_options", build_options);

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_module,
    });
    exe.strip = optimize != .Debug;
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cmd.step);

    const test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_create.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("abi", abi_module);

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    const docs_install = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate ABI documentation");
    docs_step.dependOn(&docs_install.step);

    const fmt = b.addFmt(.{ .paths = &.{ "src", "tests", "build.zig" } });
    const fmt_step = b.step("fmt", "Format ABI sources");
    fmt_step.dependOn(&fmt.step);

    const summary = b.step("summary", "Run docs, fmt, and tests");
    summary.dependOn(&docs_step.step);
    summary.dependOn(&fmt_step.step);
    summary.dependOn(&test_step.step);
}
