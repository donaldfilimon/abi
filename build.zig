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

    const test_step = b.step("test", "Run the ABI test suite");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
