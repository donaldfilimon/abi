const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zigly",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    
    exe.root_module.link_libc = true;

    const download_mod = b.createModule(.{
        .root_source_file = b.path("src/download.zig"),
    });
    exe.root_module.addImport("download", download_mod);

    const archive_mod = b.createModule(.{
        .root_source_file = b.path("src/archive.zig"),
    });
    exe.root_module.addImport("archive", archive_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
