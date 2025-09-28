const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Simplified version handling
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", "0.1.0");

    // Core ABI module
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addOptions("build_options", build_options);

    // Create main module first
    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_module.addImport("abi", abi_mod);
    main_module.addOptions("build_options", build_options);

    // Simple executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = main_module,
    });

    b.installArtifact(exe);

    // Simple test
    const unit_tests = b.addTest(.{
        .root_module = main_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
