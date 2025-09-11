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
    // Add the abi module for imports
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI module
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });

    // Main executable
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_mod,
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
    const unit_tests_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = unit_tests_mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step that
    // can be invoked like this: `zig build test`
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Database is now consumed via abi.wdbx.database; no separate named module is needed.

    // Additional modules used by tests
    // Named modules for tests
    // ai module is imported via 'abi' in tests
    const weather_mod = b.createModule(.{
        .root_source_file = b.path("src/weather.zig"),
        .target = target,
        .optimize = optimize,
    });
    const web_server_mod = b.createModule(.{
        .root_source_file = b.path("src/server/web_server.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Root module is already available via 'abi'

    // Collect more tests from tests/*.zig (excluding integration mains)
    // Attach them to the existing `test` step so `zig build test` includes them.
    test_step.dependOn(&run_unit_tests.step);

    // Unit-style tests in tests/ (standalone or package imports)
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_ai.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "abi", .module = abi_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cli_integration.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "abi", .module = abi_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_database.zig"),
            .target = target,
            .optimize = optimize,
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Heavy tests flag
    const enable_heavy = b.option(bool, "heavy-tests", "Enable heavy DB/HNSW tests") orelse false;

    // HNSW tests
    const hnsw_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_database_hnsw.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const hnsw_tests = b.addTest(.{ .root_module = hnsw_mod });
    const run_hnsw = b.addRunArtifact(hnsw_tests);
    const heavy_step = b.step("test-heavy", "Run heavy HNSW/integration tests");
    heavy_step.dependOn(&run_hnsw.step);

    // DB integration tests
    const db_int_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_database_integration.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const db_int_tests = b.addTest(.{ .root_module = db_int_mod });
    const run_db_int = b.addRunArtifact(db_int_tests);
    heavy_step.dependOn(&run_db_int.step);

    if (enable_heavy) {
        test_step.dependOn(heavy_step);
    }

    // (database_integration test temporarily disabled pending semantics alignment)

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_memory_management.zig"),
            .target = target,
            .optimize = optimize,
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // (Performance tests disabled for 0.15.1 setup)

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_simd_vector.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Configuration validation tests
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_config_validation.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "abi", .module = abi_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Weather and web server tests
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_weather.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "weather", .module = weather_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_web_server.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "web_server", .module = web_server_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Socket-level web server test (heavy)
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_web_server_socket.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "web_server", .module = web_server_mod }},
        });
        const t = b.addTest(.{ .root_module = mod });
        const run_t = b.addRunArtifact(t);
        heavy_step.dependOn(&run_t.step);
    }

    // Integration executables are already defined later in this file as part of
    // the existing 'test-integration' and 'test-all' steps.

    // Benchmark executable
    const benchmark_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/database_benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });

    const benchmark_exe = b.addExecutable(.{
        .name = "database_benchmark",
        .root_module = benchmark_mod,
    });

    // Unified benchmark system
    const benchmark_main_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });

    const benchmark_main = b.addExecutable(.{
        .name = "benchmark_main",
        .root_module = benchmark_main_mod,
    });

    const benchmark_step = b.step("benchmark", "Run unified benchmark suite");
    const run_benchmark_main = b.addRunArtifact(benchmark_main);
    run_benchmark_main.addArg("all");
    benchmark_step.dependOn(&run_benchmark_main.step);

    // Separate benchmark executables
    // Support modules for benchmarks
    const neural_benchmark_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/benchmark_suite.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });

    const neural_benchmark = b.addExecutable(.{
        .name = "neural_benchmark",
        .root_module = neural_benchmark_mod,
    });

    const simple_benchmark_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/simple_benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });

    const simple_benchmark = b.addExecutable(.{
        .name = "simple_benchmark",
        .root_module = simple_benchmark_mod,
    });

    // Benchmark steps
    const neural_benchmark_step = b.step("benchmark-neural", "Run neural network benchmarks");
    const run_neural_benchmark = b.addRunArtifact(neural_benchmark);
    neural_benchmark_step.dependOn(&run_neural_benchmark.step);

    const simple_benchmark_step = b.step("benchmark-simple", "Run simple VDBench-style benchmarks");
    const run_simple_benchmark = b.addRunArtifact(simple_benchmark);
    simple_benchmark_step.dependOn(&run_simple_benchmark.step);

    // Legacy database benchmark support
    const legacy_benchmark_step = b.step("benchmark-db", "Run database performance benchmarks");
    const run_legacy_benchmark = b.addRunArtifact(benchmark_exe);
    legacy_benchmark_step.dependOn(&run_legacy_benchmark.step);

    // Aggregate: run all benchmark suites in one command
    const bench_all_step = b.step("bench-all", "Run all benchmark suites (unified + neural + simple + db)");
    bench_all_step.dependOn(benchmark_step);
    bench_all_step.dependOn(neural_benchmark_step);
    bench_all_step.dependOn(simple_benchmark_step);
    bench_all_step.dependOn(legacy_benchmark_step);

    // Server integration tests removed during cleanup (outdated, flaky on Windows)

    // Static analysis tool
    const static_analysis_mod = b.createModule(.{
        .root_source_file = b.path("tools/static_analysis.zig"),
        .target = target,
        .optimize = optimize,
    });

    const static_analysis = b.addExecutable(.{
        .name = "static_analysis",
        .root_module = static_analysis_mod,
    });
    b.installArtifact(static_analysis);

    const run_static_analysis = b.addRunArtifact(static_analysis);
    const analyze_step = b.step("analyze", "Run static analysis");
    analyze_step.dependOn(&run_static_analysis.step);

    // Windows network diagnostic tool (Windows-only)
    const network_test_mod = b.createModule(.{
        .root_source_file = b.path("tools/windows_network_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    const network_test = b.addExecutable(.{
        .name = "windows_network_test",
        .root_module = network_test_mod,
    });
    network_test.root_module.link_libc = true;
    // Only install on Windows or when target is unspecified (native)
    if (target.result.os.tag == .windows) {
        b.installArtifact(network_test);
    }

    const run_network_test = b.addRunArtifact(network_test);
    const network_test_step = b.step("test-network", "Run Windows network diagnostic");
    network_test_step.dependOn(&run_network_test.step);

    // Plugin system tests
    const plugin_mod = b.createModule(.{
        .root_source_file = b.path("src/plugins/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const plugin_tests = b.addTest(.{
        .root_module = plugin_mod,
    });

    const run_plugin_tests = b.addRunArtifact(plugin_tests);
    const plugin_test_step = b.step("test-plugins", "Run plugin system tests");
    plugin_test_step.dependOn(&run_plugin_tests.step);

    // Code coverage integration (requires kcov to be installed)
    const coverage_step = b.step("coverage", "Generate code coverage report");
    const coverage_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .Debug, // Debug build for better coverage
    });

    const coverage_tests = b.addTest(.{
        .root_module = coverage_mod,
    });

    // Coverage with kcov (if available)
    const kcov_exe = b.addSystemCommand(&[_][]const u8{
        "kcov",
        "--clean",
        "--include-pattern=src/",
        "--exclude-pattern=tests/",
        b.pathJoin(&.{ b.install_path, "coverage" }),
    });
    kcov_exe.addArtifactArg(coverage_tests);
    coverage_step.dependOn(&kcov_exe.step);

    // Documentation generation (Markdown + Zig HTML docs)
    const docs_step = b.step("docs", "Generate API documentation");
    const docs_mod = b.createModule(.{
        .root_source_file = b.path("tools/docs_generator.zig"),
        .target = target,
        .optimize = optimize,
    });

    const docs_exe = b.addExecutable(.{
        .name = "docs_generator",
        .root_module = docs_mod,
    });
    const run_docs = b.addRunArtifact(docs_exe);
    docs_step.dependOn(&run_docs.step);

    // Performance profiling
    const profile_step = b.step("profile", "Run performance profiling");
    const profile_mod = b.createModule(.{
        .root_source_file = b.path("tools/performance_profiler.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Optimized for profiling
    });

    const profile_exe = b.addExecutable(.{
        .name = "performance_profiler",
        .root_module = profile_mod,
    });
    const run_profile = b.addRunArtifact(profile_exe);
    profile_step.dependOn(&run_profile.step);

    // Perf guard (CI gate)
    const perf_threshold_opt = b.option(u64, "perf-threshold-ns", "Average search time threshold (ns)");
    const perf_guard_mod = b.createModule(.{
        .root_source_file = b.path("tools/perf_guard.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const perf_guard_exe = b.addExecutable(.{ .name = "perf_guard", .root_module = perf_guard_mod });
    const run_perf_guard = b.addRunArtifact(perf_guard_exe);
    run_perf_guard.addArg(if (perf_threshold_opt) |t| b.fmt("{d}", .{t}) else "20000000");
    const perf_guard_step = b.step("perf-guard", "Run performance regression guard");
    perf_guard_step.dependOn(&run_perf_guard.step);

    // Performance CI/CD tool
    const perf_ci_mod = b.createModule(.{
        .root_source_file = b.path("tools/performance_ci.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const perf_ci_exe = b.addExecutable(.{ .name = "performance_ci", .root_module = perf_ci_mod });
    b.installArtifact(perf_ci_exe);

    const run_perf_ci = b.addRunArtifact(perf_ci_exe);
    const perf_ci_step = b.step("perf-ci", "Run comprehensive performance CI/CD testing");
    perf_ci_step.dependOn(&run_perf_ci.step);

    // Integration tests
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration_test_suite.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });

    const integration_tests = b.addExecutable(.{
        .name = "integration_tests",
        .root_module = integration_mod,
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // All tests (unit + integration)
    const all_tests_step = b.step("test-all", "Run all tests (unit + integration + heavy)");
    all_tests_step.dependOn(test_step);
    all_tests_step.dependOn(integration_test_step);
    if (enable_heavy) all_tests_step.dependOn(heavy_step);

    // Test matrix (cross-target unit tests with foreign checks skipped)
    const test_matrix_step = b.step("test-matrix", "Run unit tests across multiple targets (skip foreign checks)");
    const test_targets = [_]std.Target.Query{
        .{}, // native
        .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
        .{ .cpu_arch = .aarch64, .os_tag = .macos },
    };
    for (test_targets) |tq| {
        const resolved = b.resolveTargetQuery(tq);
        const unit_mod_matrix = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = resolved,
            .optimize = optimize,
        });
        const unit_tests_matrix = b.addTest(.{ .root_module = unit_mod_matrix });
        const run_unit_tests_matrix = b.addRunArtifact(unit_tests_matrix);
        run_unit_tests_matrix.skip_foreign_checks = true;
        test_matrix_step.dependOn(&run_unit_tests_matrix.step);
    }

    // Cross-platform verification step
    const cross_platform_step = b.step("cross-platform", "Verify cross-platform compatibility");

    // Define supported cross-compilation targets
    const cross_targets = [_][]const u8{
        "x86_64-linux-gnu",
        "aarch64-linux-gnu",
        "x86_64-macos",
        "aarch64-macos",
        "wasm32-wasi",
    };

    for (cross_targets) |cross_target| {
        const cross_target_query = std.Target.Query.parse(.{ .arch_os_abi = cross_target }) catch unreachable;
        const cross_target_resolved = b.resolveTargetQuery(cross_target_query);

        const cross_cli_mod = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = cross_target_resolved,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        });

        const cross_cli_exe = b.addExecutable(.{
            .name = b.fmt("abi-{s}", .{cross_target}),
            .root_module = cross_cli_mod,
        });
        // Link libc explicitly for targets that require it (e.g., Linux, macOS)
        const os_tag = cross_target_resolved.result.os.tag;
        if (os_tag == .linux or os_tag == .macos) {
            cross_cli_exe.linkLibC();
        }

        const install_cross = b.addInstallArtifact(cross_cli_exe, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("cross/{s}", .{cross_target}) } },
        });
        cross_platform_step.dependOn(&install_cross.step);
    }
    // SIMD micro-benchmark
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/simd_micro.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "abi", .module = abi_mod }},
        });
        const exe = b.addExecutable(.{ .name = "simd-micro", .root_module = mod });
        const run_bench = b.addRunArtifact(exe);
        const bench_step = b.step("bench-simd", "Run SIMD micro-benchmark");
        bench_step.dependOn(&run_bench.step);
    }
}
