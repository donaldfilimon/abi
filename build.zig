const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = main_mod,
    });

    b.installArtifact(exe);

    const test_step = b.step("test", "Run all tests");
    const unit_tests = b.addTest(.{
        .root_module = main_mod,
    });
    test_step.dependOn(&unit_tests.step);
}
