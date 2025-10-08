const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.1.0a");

    // ABI library module
    const abi_lib = b.addModule("abi", .{
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_lib.addOptions("build_options", build_options);

    // CLI executable
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("bin/abi-cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_exe.root_module.addImport("abi", abi_lib);
    b.installArtifact(cli_exe);

    // Run step
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&b.addRunArtifact(cli_exe).step);

    // Test suite
    const tests = b.addTest(.{
        .name = "abi_tests",
        .root_source_file = b.path("tests/unit/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("abi", abi_lib);
    tests.root_module.addOptions("build_options", build_options);

    const run_tests = b.addRunArtifact(tests);
    run_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run the ABI test suite");
    test_step.dependOn(&run_tests.step);

    // Integration tests
    const integration_tests = b.addTest(.{
        .name = "abi_integration_tests",
        .root_source_file = b.path("tests/integration/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("abi", abi_lib);
    integration_tests.root_module.addOptions("build_options", build_options);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    run_integration_tests.skip_foreign_checks = true;

    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Benchmarks
    const benchmarks = b.addExecutable(.{
        .name = "abi_benchmarks",
        .root_source_file = b.path("tests/benchmarks/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmarks.root_module.addImport("abi", abi_lib);
    benchmarks.root_module.addOptions("build_options", build_options);

    const benchmark_step = b.step("bench", "Run benchmarks");
    benchmark_step.dependOn(&b.addRunArtifact(benchmarks).step);

    // Documentation generator
    const docs_gen = b.addExecutable(.{
        .name = "docs_generator",
        .root_source_file = b.path("tools/build/docs_generator.zig"),
        .target = target,
        .optimize = optimize,
    });
    docs_gen.root_module.addImport("abi", abi_lib);

    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&b.addRunArtifact(docs_gen).step);

    // Format step
    const fmt_step = b.step("fmt", "Format source code");
    const fmt_cmd = b.addFmt(.{
        .paths = &.{ "lib", "bin", "tests", "tools" },
    });
    fmt_step.dependOn(&fmt_cmd.step);

    // Lint step
    const lint_step = b.step("lint", "Run linter");
    const lint_cmd = b.addExecutable(.{
        .name = "zig_lint",
        .root_source_file = b.path("tools/dev/linter.zig"),
        .target = target,
        .optimize = optimize,
    });
    lint_cmd.root_module.addImport("abi", abi_lib);
    lint_step.dependOn(&b.addRunArtifact(lint_cmd).step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{"rm", "-rf", "zig-out", "zig-cache"});
    clean_step.dependOn(&clean_cmd.step);

    // All tests step
    const test_all_step = b.step("test-all", "Run all tests");
    test_all_step.dependOn(test_step);
    test_all_step.dependOn(integration_test_step);

    // Development step (format + lint + test)
    const dev_step = b.step("dev", "Run development checks (format + lint + test)");
    dev_step.dependOn(fmt_step);
    dev_step.dependOn(lint_step);
    dev_step.dependOn(test_all_step);

    // Install step for library
    const install_lib_step = b.step("install-lib", "Install ABI library");
    const lib_artifact = b.addStaticLibrary(.{
        .name = "abi",
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_artifact.root_module.addImport("abi", abi_lib);
    lib_artifact.root_module.addOptions("build_options", build_options);
    b.installArtifact(lib_artifact);
    install_lib_step.dependOn(&lib_artifact.step);
}