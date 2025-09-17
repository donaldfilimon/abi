const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create build options
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_gpu", false); // Disable GPU for tests
    build_options.addOption(bool, "enable_simd", true);
    build_options.addOption(bool, "enable_tracy", false);
    build_options.addOption(usize, "max_memory", 512 * 1024 * 1024); // 512MB for tests

    // Test executable
    const exe = b.addExecutable(.{
        .name = "test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add build options to the executable
    exe.root_module.addOptions("build_options", build_options);

    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the test application");
    run_step.dependOn(&run_cmd.step);
}
