const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // Build options for feature flags
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "simd_level", "auto");
    build_options.addOption(bool, "gpu", false);
    build_options.addOption(bool, "simd", true);
    build_options.addOption(bool, "neural_accel", false);
    build_options.addOption(bool, "webgpu", false);
    build_options.addOption(bool, "hot_reload", false);
    build_options.addOption(bool, "enable_tracy", false);
    build_options.addOption(bool, "is_wasm", false);

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const core_mod = b.createModule(.{
        .root_source_file = b.path("src/core/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("abi", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });

    mod.addImport("core", core_mod);

    // Main executable
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = mod },
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or -p`.
    b.installArtifact(cli_exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the CLI app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by
    // the build runner when they are depended on by another step.
    const cli_run = b.addRunArtifact(cli_exe);

    run_step.dependOn(&cli_run.step);

    // Creates a step for unit testing. This will expose a `test` step that
    // can be invoked like this: `zig build test`
    const unit_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step that
    // can be invoked like this: `zig build test`
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "database_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/database_benchmark.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "database", .module = mod },
            },
        }),
    });

    // Benchmark step
    const benchmark_step = b.step("benchmark", "Run database performance benchmarks");
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    benchmark_step.dependOn(&run_benchmark.step);

    // Server integration tests removed during cleanup (outdated, flaky on Windows)

    // Static analysis tool
    const static_analysis = b.addExecutable(.{
        .name = "static_analysis",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/static_analysis.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(static_analysis);

    const run_static_analysis = b.addRunArtifact(static_analysis);
    const analyze_step = b.step("analyze", "Run static analysis");
    analyze_step.dependOn(&run_static_analysis.step);

    // Windows network diagnostic tool
    const network_test = b.addExecutable(.{
        .name = "windows_network_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("windows_network_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = mod },
            },
        }),
    });
    b.installArtifact(network_test);

    const run_network_test = b.addRunArtifact(network_test);
    const network_test_step = b.step("test-network", "Run Windows network diagnostic");
    network_test_step.dependOn(&run_network_test.step);

    // Plugin system tests
    const plugin_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/plugins/mod.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = mod },
            },
        }),
    });

    const run_plugin_tests = b.addRunArtifact(plugin_tests);
    const plugin_test_step = b.step("test-plugins", "Run plugin system tests");
    plugin_test_step.dependOn(&run_plugin_tests.step);
}
