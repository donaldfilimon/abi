//! Build script for ABI C bindings
//!
//! This produces a static and shared library that can be linked from C/C++.
//!
//! Usage:
//!   zig build                    # Build library
//!   zig build test               # Run tests
//!   zig build -Doptimize=ReleaseFast  # Optimized build

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get the main ABI module from parent project
    const abi_dep = b.dependency("abi", .{
        .target = target,
        .optimize = optimize,
    });
    const abi_module = abi_dep.module("abi");

    // Build options module (for feature flags)
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_ai", true);
    build_options.addOption(bool, "enable_gpu", true);
    build_options.addOption(bool, "enable_database", true);
    build_options.addOption(bool, "enable_network", true);
    build_options.addOption(bool, "enable_web", true);
    build_options.addOption(bool, "enable_profiling", true);
    build_options.addOption([]const u8, "package_version", "0.4.0");

    // Create static library module (with libc)
    const static_module = b.createModule(.{
        .root_source_file = b.path("src/abi_c.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    static_module.addImport("abi", abi_module);
    static_module.addImport("build_options", build_options.createModule());

    // Static library
    const static_lib = b.addLibrary(.{
        .name = "abi",
        .root_module = static_module,
        .linkage = .static,
    });

    // Install static library
    b.installArtifact(static_lib);

    // Create shared library module (with libc)
    const shared_module = b.createModule(.{
        .root_source_file = b.path("src/abi_c.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    shared_module.addImport("abi", abi_module);
    shared_module.addImport("build_options", build_options.createModule());

    // Shared library
    const shared_lib = b.addLibrary(.{
        .name = "abi",
        .root_module = shared_module,
        .linkage = .dynamic,
    });

    // Install shared library
    b.installArtifact(shared_lib);

    // Install headers (skip if directory doesn't exist)
    // Users should generate headers separately

    // Create test module (with libc)
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/abi_c.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_module.addImport("abi", abi_module);
    test_module.addImport("build_options", build_options.createModule());

    // Tests
    const tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run C bindings tests");
    test_step.dependOn(&run_tests.step);
}
