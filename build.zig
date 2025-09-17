const std = @import("std");

/// Production-ready build configuration for a high-performance vector database system.
/// This build script configures a comprehensive multi-platform project with SIMD optimizations,
/// HTTP server capabilities, plugin system, and extensive testing infrastructure.
///
/// Key Features:
/// - Cross-platform support (Windows, Linux, macOS, WebAssembly)
/// - SIMD acceleration with configurable optimization levels
/// - Modular architecture with clean dependency management
/// - Comprehensive test suites and benchmarking tools
/// - C API for language interoperability
/// - Production monitoring and profiling capabilities
///
/// Usage Examples:
///   zig build                           # Build CLI executable
///   zig build -Doptimize=ReleaseFast    # Optimized release build
///   zig build -Dsimd=true -Dsimd_level=avx2  # Enable AVX2 SIMD
///   zig build run-server                # Start HTTP server
///   zig build test-all                  # Run comprehensive test suite
///   zig build benchmark                 # Run performance benchmarks
pub fn build(b: *std.Build) void {
    // ========================================================================
    // STANDARD BUILD CONFIGURATION
    // ========================================================================

    // Allow user to specify target and optimization via CLI (e.g. -Dtarget, -Doptimize)
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ========================================================================
    // CROSS-PLATFORM DETECTION
    // ========================================================================

    const target_info = target.result;
    const is_windows = target_info.os.tag == .windows;
    const is_linux = target_info.os.tag == .linux;
    const is_macos = target_info.os.tag == .macos;
    const is_wasm = target_info.cpu.arch == .wasm32 or target_info.cpu.arch == .wasm64;

    // ========================================================================
    // BUILD OPTIONS & FEATURE FLAGS
    // ========================================================================

    // These options can be set via CLI, e.g. -Dsimd=true, -Dgpu=true
    const build_options = b.addOptions();

    // Performance and acceleration options
    build_options.addOption([]const u8, "simd_level", b.option([]const u8, "simd_level", "SIMD optimization level (auto, sse, avx, avx2, avx512)") orelse "auto");
    build_options.addOption(bool, "simd", b.option(bool, "simd", "Enable SIMD optimizations for vector operations") orelse !is_wasm);
    build_options.addOption(bool, "gpu", b.option(bool, "gpu", "Enable GPU acceleration support") orelse false);
    build_options.addOption(bool, "neural_accel", b.option(bool, "neural_accel", "Enable neural network acceleration") orelse false);
    build_options.addOption(bool, "webgpu", b.option(bool, "webgpu", "Enable WebGPU support for WASM targets") orelse is_wasm);

    // Development and debugging options
    build_options.addOption(bool, "hot_reload", b.option(bool, "hot_reload", "Enable hot reload functionality for development") orelse false);
    build_options.addOption(bool, "enable_tracy", b.option(bool, "enable_tracy", "Enable Tracy profiler integration") orelse false);
    build_options.addOption(bool, "enable_logging", b.option(bool, "enable_logging", "Enable detailed logging output") orelse true);
    build_options.addOption(bool, "enable_metrics", b.option(bool, "enable_metrics", "Enable performance metrics collection") orelse false);

    // Memory management options
    build_options.addOption(bool, "enable_memory_tracking", b.option(bool, "enable_memory_tracking", "Enable memory leak detection and tracking") orelse (optimize == .Debug));
    build_options.addOption(bool, "enable_performance_profiling", b.option(bool, "enable_performance_profiling", "Enable performance profiling") orelse (optimize == .Debug));

    // Platform detection flags
    build_options.addOption(bool, "is_wasm", is_wasm);
    build_options.addOption(bool, "is_windows", is_windows);
    build_options.addOption(bool, "is_linux", is_linux);
    build_options.addOption(bool, "is_macos", is_macos);

    // Optimization flags
    build_options.addOption(bool, "enable_lto", b.option(bool, "enable_lto", "Enable Link Time Optimization") orelse (optimize == .ReleaseFast));
    build_options.addOption(bool, "enable_strip", b.option(bool, "enable_strip", "Strip debug symbols") orelse (optimize == .ReleaseFast));
    build_options.addOption(bool, "enable_single_threaded", b.option(bool, "enable_single_threaded", "Disable threading for single-threaded builds") orelse false);

    // Create reusable build options module
    const build_options_mod = build_options.createModule();

    // ========================================================================
    // CORE MODULE SYSTEM
    // ========================================================================

    // Core foundation module - provides basic types, utilities, and cross-platform abstractions
    const core_mod = b.createModule(.{
        .root_source_file = b.path("src/core/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // High-performance SIMD vector operations module
    const simd_mod = b.createModule(.{
        .root_source_file = b.path("src/simd/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // AI and machine learning module
    const ai_mod = b.createModule(.{
        .root_source_file = b.path("src/ai/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // Core database functionality
    const database_mod = b.createModule(.{
        .root_source_file = b.path("src/database.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // HTTP server implementation for REST API endpoints
    const http_mod = b.createModule(.{
        .root_source_file = b.path("src/wdbx/http.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "database", .module = database_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // Dynamic plugin loading and management system
    const plugins_mod = b.createModule(.{
        .root_source_file = b.path("src/plugins/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // Enhanced HTTP client with retry/backoff and proxy support
    const http_client_mod = b.createModule(.{
        .root_source_file = b.path("src/http_client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // WDBX utilities and CLI components
    const wdbx_mod = b.createModule(.{
        .root_source_file = b.path("src/wdbx/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "database", .module = database_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // Public API module for external consumers
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "ai", .module = ai_mod },
            .{ .name = "database", .module = database_mod },
            .{ .name = "http", .module = http_mod },
            .{ .name = "plugins", .module = plugins_mod },
            .{ .name = "wdbx", .module = wdbx_mod },
            .{ .name = "http_client", .module = http_client_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // ========================================================================
    // EXECUTABLE TARGETS
    // ========================================================================

    // Primary CLI application
    const cli_exe = b.addExecutable(.{
        .name = if (is_windows) "abi.exe" else "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });

    // Set stack size for deep recursion (neural networks)
    cli_exe.stack_size = 8 * 1024 * 1024; // 8MB

    // Platform-specific system library linking
    if (is_windows) {
        cli_exe.linkSystemLibrary("ws2_32"); // Windows Sockets 2
        cli_exe.linkSystemLibrary("kernel32"); // Windows Kernel
        cli_exe.linkSystemLibrary("user32"); // Windows User Interface
    } else if (is_linux) {
        cli_exe.linkSystemLibrary("c"); // C runtime
        cli_exe.linkSystemLibrary("pthread"); // POSIX threads
        cli_exe.linkSystemLibrary("m"); // Math library
    } else if (is_macos) {
        cli_exe.linkSystemLibrary("c");
        cli_exe.linkSystemLibrary("pthread");
        cli_exe.linkSystemLibrary("m");
        cli_exe.linkFramework("Foundation"); // macOS Foundation framework
    }

    b.installArtifact(cli_exe);

    // Standalone HTTP server for production deployment
    if (!is_wasm) {
        const wdbx_server_exe = b.addExecutable(.{
            .name = if (is_windows) "wdbx_server.exe" else "wdbx_server",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/wdbx_unified.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "http", .module = http_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        // Platform-specific networking libraries
        if (is_windows) {
            wdbx_server_exe.linkSystemLibrary("ws2_32");
            wdbx_server_exe.linkSystemLibrary("kernel32");
            wdbx_server_exe.linkSystemLibrary("user32");
        } else if (is_linux) {
            wdbx_server_exe.linkSystemLibrary("c");
            wdbx_server_exe.linkSystemLibrary("pthread");
            wdbx_server_exe.linkSystemLibrary("m");
        } else if (is_macos) {
            wdbx_server_exe.linkSystemLibrary("c");
            wdbx_server_exe.linkSystemLibrary("pthread");
            wdbx_server_exe.linkSystemLibrary("m");
            wdbx_server_exe.linkFramework("Foundation");
        }

        b.installArtifact(wdbx_server_exe);

        // Server run step for development
        const run_server_step = b.step("run-server", "Run the WDBX HTTP server");
        const server_run = b.addRunArtifact(wdbx_server_exe);
        if (b.args) |args| {
            server_run.addArgs(args);
        }
        run_server_step.dependOn(&server_run.step);
    }

    // ========================================================================
    // BENCHMARKING & PERFORMANCE TOOLS
    // ========================================================================

    // Database performance benchmarks
    if (!is_wasm) {
        const benchmark_exe = b.addExecutable(.{
            .name = if (is_windows) "database_benchmark.exe" else "database_benchmark",
            .root_module = b.createModule(.{
                .root_source_file = b.path("benchmarks/database_benchmark.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            benchmark_exe.linkSystemLibrary("kernel32");
        } else {
            benchmark_exe.linkSystemLibrary("c");
            benchmark_exe.linkSystemLibrary("m");
        }

        b.installArtifact(benchmark_exe);

        const benchmark_step = b.step("benchmark", "Run database performance benchmarks");
        const run_benchmark = b.addRunArtifact(benchmark_exe);
        benchmark_step.dependOn(&run_benchmark.step);
    }

    // Comprehensive performance testing suite
    if (!is_wasm) {
        const perf_suite_exe = b.addExecutable(.{
            .name = if (is_windows) "performance_suite.exe" else "performance_suite",
            .root_module = b.createModule(.{
                .root_source_file = b.path("benchmarks/performance_suite.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "simd", .module = simd_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            perf_suite_exe.linkSystemLibrary("kernel32");
        } else {
            perf_suite_exe.linkSystemLibrary("c");
            perf_suite_exe.linkSystemLibrary("m");
        }

        b.installArtifact(perf_suite_exe);

        const perf_suite_step = b.step("benchmark-perf", "Run comprehensive performance suite");
        const run_perf_suite = b.addRunArtifact(perf_suite_exe);
        perf_suite_step.dependOn(&run_perf_suite.step);
    }

    // ========================================================================
    // DEVELOPMENT & ANALYSIS TOOLS
    // ========================================================================

    // Static code analysis tool
    if (!is_wasm) {
        const static_analysis = b.addExecutable(.{
            .name = if (is_windows) "static_analysis.exe" else "static_analysis",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/static_analysis.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            static_analysis.linkSystemLibrary("kernel32");
        } else {
            static_analysis.linkSystemLibrary("c");
        }

        b.installArtifact(static_analysis);

        const analysis_step = b.step("analyze", "Run static code analysis");
        const run_analysis = b.addRunArtifact(static_analysis);
        analysis_step.dependOn(&run_analysis.step);
    }

    // Continuous performance monitoring tool
    if (!is_wasm) {
        const continuous_monitor = b.addExecutable(.{
            .name = if (is_windows) "continuous_monitor.exe" else "continuous_monitor",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/continuous_monitor.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            continuous_monitor.linkSystemLibrary("kernel32");
        } else {
            continuous_monitor.linkSystemLibrary("c");
        }

        b.installArtifact(continuous_monitor);

        const monitor_step = b.step("monitor", "Run continuous performance monitoring");
        const run_monitor = b.addRunArtifact(continuous_monitor);
        monitor_step.dependOn(&run_monitor.step);
    }

    // Enhanced HTTP client test
    if (!is_wasm) {
        const http_test_exe = b.addExecutable(.{
            .name = if (is_windows) "test_enhanced_http.exe" else "test_enhanced_http",
            .root_module = b.createModule(.{
                .root_source_file = b.path("test_enhanced_http.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "http_client", .module = http_client_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            http_test_exe.linkSystemLibrary("ws2_32");
            http_test_exe.linkSystemLibrary("kernel32");
            http_test_exe.linkSystemLibrary("user32");
        } else {
            http_test_exe.linkSystemLibrary("c");
        }

        const http_test_step = b.step("test-http-enhanced", "Test enhanced HTTP client with retry/backoff");
        const run_http_test = b.addRunArtifact(http_test_exe);
        http_test_step.dependOn(&run_http_test.step);
    }

    // HTTP client demo
    if (!is_wasm) {
        const demo_exe = b.addExecutable(.{
            .name = if (is_windows) "demo_http_client.exe" else "demo_http_client",
            .root_module = b.createModule(.{
                .root_source_file = b.path("demo_http_client.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            demo_exe.linkSystemLibrary("ws2_32");
            demo_exe.linkSystemLibrary("kernel32");
            demo_exe.linkSystemLibrary("user32");
        } else {
            demo_exe.linkSystemLibrary("c");
        }

        b.installArtifact(demo_exe);
        const demo_step = b.step("demo-http", "Run enhanced HTTP client demonstration");
        const run_demo = b.addRunArtifact(demo_exe);
        demo_step.dependOn(&run_demo.step);
    }

    // Windows-specific network diagnostics
    if (is_windows) {
        const network_test = b.addExecutable(.{
            .name = "windows_network_test.exe",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tests/test_windows_networking.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "http", .module = http_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });
        network_test.linkSystemLibrary("ws2_32");
        network_test.linkSystemLibrary("kernel32");
        b.installArtifact(network_test);

        const run_network_test = b.addRunArtifact(network_test);
        const network_test_step = b.step("test-network", "Run Windows network diagnostics");
        network_test_step.dependOn(&run_network_test.step);
    }

    // Integration testing suite
    if (!is_wasm) {
        const integration_tests = b.addExecutable(.{
            .name = if (is_windows) "integration_tests.exe" else "integration_tests",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tests/integration_tests.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "http", .module = http_mod },
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        if (is_windows) {
            integration_tests.linkSystemLibrary("ws2_32");
            integration_tests.linkSystemLibrary("kernel32");
        } else {
            integration_tests.linkSystemLibrary("c");
            integration_tests.linkSystemLibrary("pthread");
        }

        const run_integration_tests = b.addRunArtifact(integration_tests);
        const integration_test_step = b.step("test-integration", "Run integration tests");
        integration_test_step.dependOn(&run_integration_tests.step);
    }

    // ========================================================================
    // LIBRARY TARGETS FOR INTEROPERABILITY
    // ========================================================================

    // C API shared library for language interoperability
    if (!is_wasm) {
        const c_api = b.addLibrary(.{
            .name = "wdbx_c_api",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/c_api.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
            .linkage = .dynamic,
        });

        if (is_windows) {
            c_api.linkSystemLibrary("kernel32");
        } else {
            c_api.linkSystemLibrary("c");
        }

        b.installArtifact(c_api);

        // Static version of C API for embedded use cases
        const c_api_static = b.addLibrary(.{
            .name = "wdbx_c_api_static",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/c_api.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
            .linkage = .static,
        });
        b.installArtifact(c_api_static);
    }

    // ========================================================================
    // BUILD STEPS & COMMANDS
    // ========================================================================

    // Primary run step for CLI application
    const run_step = b.step("run", "Run the CLI application");
    const cli_run = b.addRunArtifact(cli_exe);
    if (b.args) |args| {
        cli_run.addArgs(args);
    }
    run_step.dependOn(&cli_run.step);

    // ========================================================================
    // COMPREHENSIVE TEST SUITE
    // ========================================================================

    // Main unit tests for the entire project
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.addImport("core", core_mod);
    unit_tests.root_module.addImport("simd", simd_mod);
    unit_tests.root_module.addImport("ai", ai_mod);
    unit_tests.root_module.addImport("database", database_mod);
    unit_tests.root_module.addImport("http", http_mod);
    unit_tests.root_module.addImport("plugins", plugins_mod);
    unit_tests.root_module.addImport("wdbx", wdbx_mod);
    unit_tests.root_module.addImport("build_options", build_options_mod);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run comprehensive unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // SIMD-specific test suite
    const simd_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/simd/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    simd_tests.root_module.addImport("core", core_mod);
    simd_tests.root_module.addImport("build_options", build_options_mod);
    const run_simd_tests = b.addRunArtifact(simd_tests);
    const simd_test_step = b.step("test-simd", "Run SIMD optimization tests");
    simd_test_step.dependOn(&run_simd_tests.step);

    // Database functionality tests
    const database_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/database.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    database_tests.root_module.addImport("core", core_mod);
    database_tests.root_module.addImport("simd", simd_mod);
    database_tests.root_module.addImport("build_options", build_options_mod);
    const run_database_tests = b.addRunArtifact(database_tests);
    const database_test_step = b.step("test-database", "Run database functionality tests");
    database_test_step.dependOn(&run_database_tests.step);

    // HTTP server functionality tests
    const http_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx/http.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    http_tests.root_module.addImport("core", core_mod);
    http_tests.root_module.addImport("database", database_mod);
    http_tests.root_module.addImport("build_options", build_options_mod);
    const run_http_tests = b.addRunArtifact(http_tests);
    const http_test_step = b.step("test-http", "Run HTTP server tests");
    http_test_step.dependOn(&run_http_tests.step);

    // Plugin system tests
    const plugin_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/plugins/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    plugin_tests.root_module.addImport("core", core_mod);
    plugin_tests.root_module.addImport("build_options", build_options_mod);
    const run_plugin_tests = b.addRunArtifact(plugin_tests);
    const plugin_test_step = b.step("test-plugins", "Run plugin system tests");
    plugin_test_step.dependOn(&run_plugin_tests.step);

    // WDBX module tests
    const wdbx_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    wdbx_tests.root_module.addImport("core", core_mod);
    wdbx_tests.root_module.addImport("database", database_mod);
    wdbx_tests.root_module.addImport("build_options", build_options_mod);
    const run_wdbx_tests = b.addRunArtifact(wdbx_tests);
    const wdbx_test_step = b.step("test-wdbx", "Run WDBX module tests");
    wdbx_test_step.dependOn(&run_wdbx_tests.step);

    // TCP echo unit test
    const tcp_echo_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_tcp_echo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tcp_echo_tests = b.addRunArtifact(tcp_echo_tests);
    const tcp_echo_step = b.step("test-tcp-echo", "Run TCP echo client/server test");
    tcp_echo_step.dependOn(&run_tcp_echo_tests.step);

    // WebSocket upgrade test (requires server running)
    const ws_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_websocket_echo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ws_tests = b.addRunArtifact(ws_tests);
    const ws_step = b.step("test-ws", "Run WebSocket /ws handshake test (server must be running)");
    ws_step.dependOn(&run_ws_tests.step);

    // Rate limiting tests
    const rate_limit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_rate_limiting.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_rate_limit_tests = b.addRunArtifact(rate_limit_tests);
    const rate_limit_step = b.step("test-rate-limit", "Run rate limiting tests");
    rate_limit_step.dependOn(&run_rate_limit_tests.step);

    // HTTP smoke tool
    const http_smoke = b.addExecutable(.{
        .name = if (is_windows) "http_smoke.exe" else "http_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/http_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    if (is_windows) {
        http_smoke.linkSystemLibrary("ws2_32");
        http_smoke.linkSystemLibrary("kernel32");
        http_smoke.linkSystemLibrary("user32");
    } else {
        http_smoke.linkSystemLibrary("c");
    }
    b.installArtifact(http_smoke);
    const http_smoke_step = b.step("smoke-http", "Run HTTP smoke test against local server");
    const run_http_smoke = b.addRunArtifact(http_smoke);
    http_smoke_step.dependOn(&run_http_smoke.step);

    // ========================================================================
    // DOCUMENTATION & MAINTENANCE
    // ========================================================================

    // Generate comprehensive API documentation
    const docs_step = b.step("docs", "Generate API documentation");
    const docs = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    docs.root_module.addImport("core", core_mod);
    docs.root_module.addImport("simd", simd_mod);
    docs.root_module.addImport("ai", ai_mod);
    docs.root_module.addImport("database", database_mod);
    docs.root_module.addImport("http", http_mod);
    docs.root_module.addImport("plugins", plugins_mod);
    docs.root_module.addImport("wdbx", wdbx_mod);
    docs.root_module.addImport("build_options", build_options_mod);
    const docs_install = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);

    // Add API documentation extraction tool
    const api_docs = b.addExecutable(.{
        .name = if (is_windows) "generate_api_docs.exe" else "generate_api_docs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_api_docs.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    b.installArtifact(api_docs);

    const api_docs_step = b.step("api-docs", "Generate markdown API documentation");
    const run_api_docs = b.addRunArtifact(api_docs);
    run_api_docs.addArg("--output");
    run_api_docs.addArg("docs/api/");
    api_docs_step.dependOn(&run_api_docs.step);

    // Cross-platform build artifact cleanup
    const clean_step = b.step("clean", "Remove all build artifacts");
    if (is_windows) {
        const clean_cmd = b.addSystemCommand(&[_][]const u8{
            "cmd", "/c", "rmdir", "/s", "/q", "zig-cache", "zig-out", "2>nul", "||", "echo", "Clean completed",
        });
        clean_step.dependOn(&clean_cmd.step);
    } else {
        const clean_cmd = b.addSystemCommand(&[_][]const u8{
            "rm", "-rf", "zig-cache", "zig-out",
        });
        clean_step.dependOn(&clean_cmd.step);
    }

    // Install C API headers for external integration
    if (!is_wasm) {
        const install_headers_step = b.step("install-headers", "Install C API headers");
        const install_header = b.addInstallFile(
            b.path("include/wdbx_c_api.h"),
            "include/wdbx_c_api.h",
        );
        install_headers_step.dependOn(&install_header.step);
    }

    // ========================================================================
    // CONVENIENCE AGGREGATION STEPS
    // ========================================================================

    // Run all test suites in sequence
    const test_all_step = b.step("test-all", "Run all test suites comprehensively");
    test_all_step.dependOn(&run_unit_tests.step);
    test_all_step.dependOn(&run_simd_tests.step);
    test_all_step.dependOn(&run_database_tests.step);
    test_all_step.dependOn(&run_http_tests.step);
    test_all_step.dependOn(&run_plugin_tests.step);
    test_all_step.dependOn(&run_wdbx_tests.step);
    test_all_step.dependOn(&run_tcp_echo_tests.step);
    test_all_step.dependOn(&run_rate_limit_tests.step);

    // Run all benchmark suites (non-WASM targets only)
    if (!is_wasm) {
        _ = b.step("bench-all", "Run all benchmark suites");
        // Individual benchmark steps are added conditionally above
    }

    // ========================================================================
    // DEFAULT BUILD TARGET
    // ========================================================================

    // Set the CLI executable as the default build target
    b.default_step.dependOn(&cli_exe.step);
}
