//! Build configuration for ABI C bindings.
//! Builds shared and static libraries plus installs headers.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shared library
    const shared_lib = b.addLibrary(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
        .linkage = .dynamic,
    });
    b.installArtifact(shared_lib);

    // Static library
    const static_lib = b.addLibrary(.{
        .name = "abi_static",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    b.installArtifact(static_lib);

    // Install headers
    b.installDirectory(.{
        .source_dir = b.path("include"),
        .install_dir = .header,
        .install_subdir = "",
    });

    // Tests
    const test_step = b.step("test", "Run C bindings tests");

    const test_modules = [_][]const u8{
        "src/errors.zig",
        "src/simd_impl.zig",
        "src/simd.zig",
        "src/framework.zig",
        "src/database.zig",
        "src/gpu.zig",
        "src/agent.zig",
        "src/main.zig",
    };

    for (test_modules) |module| {
        const unit_test = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(module),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run_test = b.addRunArtifact(unit_test);
        test_step.dependOn(&run_test.step);
    }
}
