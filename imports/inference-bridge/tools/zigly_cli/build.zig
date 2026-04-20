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

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const test_cli = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_cli.root_module.link_libc = true;
    const run_cli_tests = b.addRunArtifact(test_cli);
    const test_cli_step = b.step("test-cli", "Run cli unit tests");
    test_cli_step.dependOn(&run_cli_tests.step);

    const test_core = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/core.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_core.root_module.link_libc = true;
    const run_core_tests = b.addRunArtifact(test_core);
    const test_core_step = b.step("test-core", "Run core unit tests");
    test_core_step.dependOn(&run_core_tests.step);

    const test_download = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/download.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_download.root_module.link_libc = true;
    const run_download_tests = b.addRunArtifact(test_download);
    const test_download_step = b.step("test-download", "Run download unit tests");
    test_download_step.dependOn(&run_download_tests.step);

    const test_archive = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/archive.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_archive.root_module.link_libc = true;
    const run_archive_tests = b.addRunArtifact(test_archive);
    const test_archive_step = b.step("test-archive", "Run archive unit tests");
    test_archive_step.dependOn(&run_archive_tests.step);

    const test_all = b.step("test", "Run all unit tests");
    test_all.dependOn(&run_cli_tests.step);
    test_all.dependOn(&run_core_tests.step);
    test_all.dependOn(&run_download_tests.step);
    test_all.dependOn(&run_archive_tests.step);
}
