const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Feature flags (currently unused)
    _ = b.option(bool, "enable-gpu", "Enable GPU features") orelse false;
    _ = b.option(bool, "enable-web", "Enable Web features") orelse false;
    _ = b.option(bool, "enable-monitoring", "Enable monitoring features") orelse false;
    _ = b.option(bool, "enable-tracy", "Enable Tracy instrumentation hooks") orelse false;

    // ABI module
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("src/comprehensive_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("abi", abi_mod);

    const cli_module = b.createModule(.{
        .root_source_file = b.path("src/cli_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_module.addImport("abi", abi_mod);
    cli_module.addOptions("build_options", build_options);

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_module,
    });
    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&b.addRunArtifact(exe).step);

<<<<<<< HEAD
    // Test suite
    const tests = b.addTest(.{
        .name = "abi_tests",
=======
    const tests_module = b.createModule(.{
>>>>>>> a2b63365b817a190f4e5938b2b24240c5cbea742
        .root_source_file = b.path("src/tests/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
<<<<<<< HEAD
    tests.root_module.addImport("abi", abi_mod);
=======
    tests_module.addImport("abi", abi_mod);
    tests_module.addOptions("build_options", build_options);

    const tests = b.addTest(.{
        .root_module = tests_module,
    });

    const run_tests = b.addRunArtifact(tests);
    run_tests.skip_foreign_checks = true;
>>>>>>> a2b63365b817a190f4e5938b2b24240c5cbea742

    const test_step = b.step("test", "Run the ABI test suite");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
