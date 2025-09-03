//! Plugin Build Script
//!
//! This script builds all example plugins as shared libraries

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build example plugin
    const example_plugin = b.addSharedLibrary(.{
        .name = "example_plugin",
        .root_source_file = b.path("examples/plugins/example_plugin.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Set the ABI version
    example_plugin.addCSourceFile(.{
        .file = b.path("src/plugins/interface.zig"),
        .flags = &.{},
    });
    
    b.installArtifact(example_plugin);

    // Build advanced plugin example
    const advanced_plugin = b.addSharedLibrary(.{
        .name = "advanced_plugin",
        .root_source_file = b.path("examples/plugins/advanced_plugin_example.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    b.installArtifact(advanced_plugin);

    // Plugin validation tool
    const plugin_validator = b.addExecutable(.{
        .name = "plugin_validator",
        .root_source_file = b.path("tools/plugin_validator.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    b.installArtifact(plugin_validator);

    // Add build steps
    const build_plugins_step = b.step("plugins", "Build all example plugins");
    build_plugins_step.dependOn(&example_plugin.step);
    build_plugins_step.dependOn(&advanced_plugin.step);

    const validate_step = b.step("validate-plugin", "Validate a plugin's ABI compatibility");
    const run_validator = b.addRunArtifact(plugin_validator);
    
    if (b.args) |args| {
        run_validator.addArgs(args);
    }
    
    validate_step.dependOn(&run_validator.step);

    // Plugin test harness
    const test_harness = b.addExecutable(.{
        .name = "plugin_test_harness",
        .root_source_file = b.path("tools/plugin_test_harness.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    b.installArtifact(test_harness);

    const test_plugins_step = b.step("test-plugin-harness", "Run plugin test harness");
    const run_harness = b.addRunArtifact(test_harness);
    test_plugins_step.dependOn(&run_harness.step);
}