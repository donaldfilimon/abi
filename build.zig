const std = @import("std");

/// Production-ready build configuration for a high-performance vector database system.
///
/// This build script configures a comprehensive multi-platform project with SIMD optimizations,
/// HTTP server capabilities, plugin system, and extensive testing infrastructure.
///
/// ## Key Features
/// - Cross-platform support (Windows, Linux, macOS, WebAssembly)
/// - SIMD acceleration with configurable optimization levels
/// - Modular architecture with clean dependency management
/// - Comprehensive test suites and benchmarking tools
/// - C API for language interoperability
/// - Production monitoring and profiling capabilities
///
/// ## Usage Examples
/// ```bash
/// # Basic builds
/// zig build                           # Build CLI executable
/// zig build -Doptimize=ReleaseFast    # Optimized release build
/// zig build -Dtarget=x86_64-windows   # Cross-compile for Windows
///
/// # Feature-specific builds
/// zig build -Dsimd=true -Dsimd_level=avx2  # Enable AVX2 SIMD
/// zig build -Dgpu=true -Dneural_accel=true # Enable GPU acceleration
/// zig build -Denable_tracy=true            # Enable Tracy profiler
///
/// # Development and testing
/// zig build run-server                # Start HTTP server
/// zig build test-all                  # Run comprehensive test suite
/// zig build benchmark                 # Run performance benchmarks
/// zig build docs                      # Generate API documentation
///
/// # Auto-cleanup commands (run operation then clean artifacts)
/// zig build test-clean                # Run all tests then auto-clean
/// zig build test-simd-clean           # Run SIMD tests then auto-clean
/// zig build test-database-clean       # Run database tests then auto-clean
/// zig build test-http-clean           # Run HTTP tests then auto-clean
/// zig build dev-clean                 # Build for development then auto-clean
/// zig build ci                        # Full CI pipeline (build + test + clean)
/// zig build clean                     # Manual cleanup of build artifacts
/// ```
///
/// ## Build Options
/// - `simd_level`: SIMD optimization level (auto, sse, avx, avx2, avx512, neon)
/// - `simd`: Enable SIMD optimizations (default: true, false for WASM/RISC-V)
/// - `gpu`: Enable GPU acceleration support
/// - `neural_accel`: Enable neural network acceleration
/// - `enable_tracy`: Enable Tracy profiler integration
/// - `enable_logging`: Enable detailed logging output
/// - `enable_metrics`: Enable performance metrics collection
///
/// ## Cross-Platform Support
/// - **Windows**: Full support with Windows Sockets, Kernel32, User32, AdvAPI32
/// - **Linux**: Full support with GLIBC, dynamic linking, real-time extensions
/// - **macOS**: Full support with Foundation framework, CoreFoundation
/// - **BSD Variants**: Support for FreeBSD, OpenBSD, NetBSD with execinfo
/// - **Architecture**: x86, x86_64, ARM, AArch64, RISC-V detection and optimization
/// - **WASM**: WebAssembly support with appropriate feature disabling
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
    const is_x86 = target_info.cpu.arch == .x86 or target_info.cpu.arch == .x86_64;
    const is_arm = target_info.cpu.arch == .arm or target_info.cpu.arch == .aarch64;
    const is_riscv = target_info.cpu.arch == .riscv32 or target_info.cpu.arch == .riscv64;

    // ========================================================================
    // BUILD OPTIONS & FEATURE FLAGS
    // ========================================================================

    // Create build options module for compile-time configuration
    const build_options = b.addOptions();

    // ========================================================================
    // PERFORMANCE & ACCELERATION OPTIONS
    // ========================================================================

    // SIMD configuration with cross-platform detection
    const default_simd_level = if (is_x86) "avx2" // Most modern x86 CPUs support AVX2
        else if (is_arm) "neon" else "auto";

    const simd_level = b.option([]const u8, "simd_level", "SIMD optimization level (auto, sse, avx, avx2, avx512, neon)") orelse default_simd_level;
    const enable_simd = b.option(bool, "simd", "Enable SIMD optimizations for vector operations") orelse (!is_wasm and !is_riscv);

    build_options.addOption([]const u8, "simd_level", simd_level);
    build_options.addOption(bool, "simd", enable_simd);

    // GPU and neural acceleration
    const enable_gpu = b.option(bool, "gpu", "Enable GPU acceleration support") orelse false;
    const enable_neural_accel = b.option(bool, "neural_accel", "Enable neural network acceleration") orelse false;
    const enable_webgpu = b.option(bool, "webgpu", "Enable WebGPU support for WASM targets") orelse is_wasm;

    build_options.addOption(bool, "gpu", enable_gpu);
    build_options.addOption(bool, "neural_accel", enable_neural_accel);
    build_options.addOption(bool, "webgpu", enable_webgpu);

    // ========================================================================
    // DEVELOPMENT & DEBUGGING OPTIONS
    // ========================================================================

    const enable_hot_reload = b.option(bool, "hot_reload", "Enable hot reload functionality for development") orelse false;
    const enable_tracy = b.option(bool, "enable_tracy", "Enable Tracy profiler integration") orelse false;
    const enable_logging = b.option(bool, "enable_logging", "Enable detailed logging output") orelse true;
    const enable_metrics = b.option(bool, "enable_metrics", "Enable performance metrics collection") orelse false;

    build_options.addOption(bool, "hot_reload", enable_hot_reload);
    build_options.addOption(bool, "enable_tracy", enable_tracy);
    build_options.addOption(bool, "enable_logging", enable_logging);
    build_options.addOption(bool, "enable_metrics", enable_metrics);

    // ========================================================================
    // MEMORY MANAGEMENT OPTIONS
    // ========================================================================

    const enable_memory_tracking = b.option(bool, "enable_memory_tracking", "Enable memory leak detection and tracking") orelse (optimize == .Debug);
    const enable_performance_profiling = b.option(bool, "enable_performance_profiling", "Enable performance profiling") orelse (optimize == .Debug);

    build_options.addOption(bool, "enable_memory_tracking", enable_memory_tracking);
    build_options.addOption(bool, "enable_performance_profiling", enable_performance_profiling);

    // ========================================================================
    // PLATFORM DETECTION FLAGS
    // ========================================================================

    build_options.addOption(bool, "is_wasm", is_wasm);
    build_options.addOption(bool, "is_windows", is_windows);
    build_options.addOption(bool, "is_linux", is_linux);
    build_options.addOption(bool, "is_macos", is_macos);

    // ========================================================================
    // OPTIMIZATION FLAGS
    // ========================================================================

    const enable_lto = b.option(bool, "enable_lto", "Enable Link Time Optimization") orelse (optimize == .ReleaseFast);
    const enable_strip = b.option(bool, "enable_strip", "Strip debug symbols") orelse (optimize == .ReleaseFast);
    const enable_single_threaded = b.option(bool, "enable_single_threaded", "Disable threading for single-threaded builds") orelse false;

    build_options.addOption(bool, "enable_lto", enable_lto);
    build_options.addOption(bool, "enable_strip", enable_strip);
    build_options.addOption(bool, "enable_single_threaded", enable_single_threaded);

    // ========================================================================
    // MAIN APPLICATION MODULES
    // ========================================================================

    // ========================================================================
    // MODULE REGISTRATION
    // ========================================================================

    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core/mod.zig"),
        .target = target,
    });

    const simd_mod = b.addModule("simd", .{
        .root_source_file = b.path("src/simd.zig"),
        .target = target,
    });

    // AI module
    const ai_mod = b.addModule("ai", .{
        .root_source_file = b.path("src/ai/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    // GPU module
    const gpu_mod = b.addModule("gpu", .{
        .root_source_file = b.path("src/gpu/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/abi/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "ai", .module = ai_mod },
            .{ .name = "gpu", .module = gpu_mod },
        },
    });
    abi_mod.addOptions("build_options", build_options);

    // Services module
    const services_mod = b.addModule("services", .{
        .root_source_file = b.path("src/services/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
        },
    });

    // Plugins module
    const plugins_mod = b.addModule("plugins", .{
        .root_source_file = b.path("src/plugins/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
        },
    });

    // Connectors module
    const connectors_mod = b.addModule("connectors", .{
        .root_source_file = b.path("src/connectors/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "plugins", .module = plugins_mod },
        },
    });

    // WDBX database module
    const wdbx_mod = b.addModule("wdbx", .{
        .root_source_file = b.path("src/wdbx/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
        },
    });

    const web_server_mod = b.addModule("web_server", .{
        .root_source_file = b.path("src/server/web_server.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });

    // CLI module - provides command-line interface
    const cli_mod = b.addModule("cli", .{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "ai", .module = ai_mod },
            .{ .name = "gpu", .module = gpu_mod },
            .{ .name = "wdbx", .module = wdbx_mod },
            .{ .name = "services", .module = services_mod },
            .{ .name = "connectors", .module = connectors_mod },
            .{ .name = "plugins", .module = plugins_mod },
        },
    });

    // ========================================================================
    // EXECUTABLE TARGETS
    // ========================================================================

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_mod,
    });

    // Add build options to executable
    exe.root_module.addOptions("build_options", build_options);

    // ========================================================================
    // DEPENDENCY MANAGEMENT
    // ========================================================================

    // Apply platform-specific dependencies
    if (is_windows) {
        exe.linkSystemLibrary("kernel32");
        exe.linkSystemLibrary("user32");
        exe.linkSystemLibrary("ws2_32");
    } else if (is_linux) {
        exe.linkLibC();
    } else if (is_macos) {
        exe.linkLibC();
    }

    // ========================================================================
    // BUILD STEPS
    // ========================================================================

    // Install executable
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{ .root_module = abi_mod, .use_lld = true });
    unit_tests.root_module.addOptions("build_options", build_options);

    const integration_mod = b.addModule("integration_tests", .{
        .root_source_file = b.path("tests/main.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const integration_tests = b.addTest(.{
        .root_module = integration_mod,
    });
    integration_tests.root_module.addOptions("build_options", build_options);
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);

    const bench_mod = b.addModule("benchmarks", .{
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "benchmarks",
        .root_module = bench_mod,
    });
    bench_exe.root_module.addOptions("build_options", build_options);
    const run_benchmarks = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run benchmark suite");
    bench_step.dependOn(&run_benchmarks.step);

    const static_mod = b.addModule("static_analysis", .{
        .root_source_file = b.path("tools/static_analysis.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const static_analyzer = b.addExecutable(.{
        .name = "static_analysis",
        .root_module = static_mod,
    });
    static_analyzer.root_module.addOptions("build_options", build_options);
    const run_static = b.addRunArtifact(static_analyzer);
    const static_step = b.step("static-analysis", "Run static analysis checks");
    static_step.dependOn(&run_static.step);

    const perf_mod = b.addModule("perf_guard", .{
        .root_source_file = b.path("tools/perf_guard.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const perf_guard = b.addExecutable(.{
        .name = "perf_guard",
        .root_module = perf_mod,
    });
    perf_guard.root_module.addOptions("build_options", build_options);
    const run_perf_guard = b.addRunArtifact(perf_guard);
    const perf_step = b.step("perf-guard", "Run performance guard tool");
    perf_step.dependOn(&run_perf_guard.step);

    const docs_mod = b.addModule("docs_generator", .{
        .root_source_file = b.path("tools/generate_api_docs.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const docs_exe = b.addExecutable(.{
        .name = "generate_docs",
        .root_module = docs_mod,
    });
    docs_exe.root_module.addOptions("build_options", build_options);
    const run_docs = b.addRunArtifact(docs_exe);
    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&run_docs.step);

    const security_mod = b.addModule("security_scan", .{
        .root_source_file = b.path("tools/advanced_code_analyzer.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const security_exe = b.addExecutable(.{
        .name = "security_scan",
        .root_module = security_mod,
    });
    security_exe.root_module.addOptions("build_options", build_options);
    const run_security = b.addRunArtifact(security_exe);
    const security_step = b.step("security-scan", "Run security analysis suite");
    security_step.dependOn(&run_security.step);

    const e2e_mod = b.addModule("e2e_tests", .{
        .root_source_file = b.path("tests/test_web_server_e2e.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
            .{ .name = "web_server", .module = web_server_mod },
        },
    });
    const e2e_tests = b.addTest(.{
        .root_module = e2e_mod,
    });
    e2e_tests.root_module.addOptions("build_options", build_options);
    const run_e2e = b.addRunArtifact(e2e_tests);
    const e2e_step = b.step("e2e", "Run end-to-end test suite");
    e2e_step.dependOn(&run_e2e.step);

    const lint_cmd = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "fmt",
        "--check",
        "build.zig",
        "src/root.zig",
        "src/abi/mod.zig",
        "src/core/errors.zig",
        "src/core/logging.zig",
        "src/core/lifecycle.zig",
        "src/core/mod.zig",
    });
    const lint_step = b.step("lint", "Run formatting lint checks");
    lint_step.dependOn(&lint_cmd.step);

    const ci_step = b.step("ci", "Run full CI pipeline");
    ci_step.dependOn(test_step);
    ci_step.dependOn(bench_step);
    ci_step.dependOn(static_step);
    ci_step.dependOn(perf_step);
    ci_step.dependOn(security_step);
    ci_step.dependOn(lint_step);
    ci_step.dependOn(e2e_step);
    ci_step.dependOn(docs_step);
}
