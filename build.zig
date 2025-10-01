const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU features") orelse false;
    const enable_web = b.option(bool, "enable-web", "Enable Web features") orelse false;
    const enable_monitoring = b.option(bool, "enable-monitoring", "Enable monitoring features") orelse false;
    const enable_tracy = b.option(bool, "enable-tracy", "Enable Tracy instrumentation hooks") orelse false;

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.1.0a");
    build_options.addOption(bool, "enable_gpu", enable_gpu);
    build_options.addOption(bool, "enable_web", enable_web);
    build_options.addOption(bool, "enable_monitoring", enable_monitoring);
    build_options.addOption(bool, "enable_tracy", enable_tracy);

    const abi_mod = b.addModule("abi", .{
        .root_source_file = .{ .path = "src/mod.zig" },
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addOptions("build_options", build_options);

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = .{ .path = "src/comprehensive_cli.zig" },
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("abi", abi_mod);
    exe.root_module.addOptions("build_options", build_options);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_source_file = .{ .path = "src/tests/mod.zig" },
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("abi", abi_mod);
    tests.root_module.addOptions("build_options", build_options);

    const run_tests = b.addRunArtifact(tests);
    run_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run the ABI test suite");
    test_step.dependOn(&run_tests.step);
}
