
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options for the library
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.2.0");
    build_options.addOption(bool, "enable_gpu", b.option(bool, "enable-gpu", "Enable GPU support") orelse true);
    build_options.addOption(bool, "enable_ai", b.option(bool, "enable-ai", "Enable AI features") orelse true);
    build_options.addOption(bool, "enable_web", b.option(bool, "enable-web", "Enable web features") orelse true);
    build_options.addOption(bool, "enable_database", b.option(bool, "enable-database", "Enable database features") orelse true);

    // Core library module
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options.createModule());

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("abi", abi_module);
    b.installArtifact(exe);

    // Run step for CLI
    const run_cli = b.addRunArtifact(exe);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cli.addArgs(args);
    }

    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cli.step);

    // Test suite
    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    main_tests.root_module.addImport("abi", abi_module);

    const run_main_tests = b.addRunArtifact(main_tests);
    run_main_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_main_tests.step);
}
