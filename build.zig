const std = @import("std");

pub fn build(b: *std.Build) void {

    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{

        .name = "multi_persona_framework", // Retained the name from 'enhance-abbey-aviva-abi-framework-with-visual-aids'

        .root_source_file = .{ .path = "src/main.zig" },

        .target = target,

        .optimize = optimize,

    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    if (b.args) |args| {

        run_cmd.addArgs(args);

    }

    b.step("run", "Run the program").dependOn(&run_cmd.step); // Combined the dependOn from both branches

}


    text: []const u8,