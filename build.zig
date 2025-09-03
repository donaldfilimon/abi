const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    // Main executable for the Abi/WDBX framework
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    // Aggregate top-level tests (most modules are imported via root.zig)
    const root_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(root_tests);
    b.default_step.dependOn(&run_tests.step);
}