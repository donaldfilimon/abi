const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.1.0a");

    // ABI module
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addOptions("build_options", build_options);

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("src/comprehensive_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("abi", abi_mod);
    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&b.addRunArtifact(exe).step);

    // Test suite
    const tests = b.addTest(.{
        .name = "abi_tests",
        .root_source_file = b.path("src/tests/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("abi", abi_mod);
    tests.root_module.addOptions("build_options", build_options);

    const run_tests = b.addRunArtifact(tests);
    run_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run the ABI test suite");
    test_step.dependOn(&run_tests.step);

    // Documentation generator
    const docs_gen = b.addExecutable(.{
        .name = "docs_generator",
        .root_source_file = b.path("src/tools/docs_generator.zig"),
        .target = target,
        .optimize = optimize,
    });
    docs_gen.root_module.addImport("abi", abi_mod);

    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&b.addRunArtifact(docs_gen).step);

    // Benchmarks executable
    const bench = b.addExecutable(.{
        .name = "abi-bench",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench.root_module.addImport("abi", abi_mod);

    const bench_step = b.step("bench", "Run the benchmark suite");
    bench_step.dependOn(&b.addRunArtifact(bench).step);

    // Developer tools executable
    const tools_exe = b.addExecutable(.{
        .name = "abi-tools",
        .root_source_file = b.path("src/tools/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tools_exe.root_module.addImport("abi", abi_mod);

    const tools_step = b.step("tools", "Run developer tools entrypoint");
    tools_step.dependOn(&b.addRunArtifact(tools_exe).step);

    // Formatting step
    const fmt_step = b.step("fmt", "Format Zig sources");
    const fmt = b.addFmt(&[_][]const u8{"."});
    fmt_step.dependOn(&fmt.step);

    // Aggregate check step: format + tests
    const check = b.step("check", "Run formatting and tests");
    check.dependOn(&fmt.step);
    check.dependOn(test_step);
}
