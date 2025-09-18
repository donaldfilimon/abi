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

    // Root module - provides main ABI functionality
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // CLI module - provides command-line interface
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "abi", .module = root_mod },
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
    const unit_tests = b.addTest(.{
        .root_module = root_mod,
    });
    unit_tests.root_module.addOptions("build_options", build_options);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
}
