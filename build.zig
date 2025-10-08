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

    // Tools CLI (aggregates utilities under src/tools)
    const tools_exe = b.addExecutable(.{
        .name = "abi-tools",
        .root_source_file = b.path("src/tools/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tools_exe.root_module.addImport("abi", abi_mod);
    b.installArtifact(tools_exe);

    const tools_step = b.step("tools", "Build the ABI tools CLI");
    tools_step.dependOn(&tools_exe.step);

    const run_tools = b.addRunArtifact(tools_exe);
    const tools_run_step = b.step("tools-run", "Run the ABI tools CLI");
    tools_run_step.dependOn(&run_tools.step);
}
