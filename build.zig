const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_module.addImport("abi", abi_mod);

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = main_module,
    });
    b.installArtifact(exe);

    const unit_tests = b.addTest(.{
        .root_module = main_module,
    });

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&unit_tests.step);
}
