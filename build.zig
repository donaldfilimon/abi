const std = @import("std");

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

    // Shared modules
    const core_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/core/mod.zig"),
        .target = target,
    });

    const utils_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/utils/mod.zig"),
        .target = target,
    });

    const platform_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/platform/mod.zig"),
        .target = target,
    });

    const logging_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/logging/mod.zig"),
        .target = target,
    });

    const simd_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/simd.zig"),
        .target = target,
    });

    // Feature modules
    const ai_mod = b.createModule(.{
        .root_source_file = b.path("src/features/ai/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "utils", .module = utils_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    const gpu_mod = b.createModule(.{
        .root_source_file = b.path("src/features/gpu/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "utils", .module = utils_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    // Main ABI module
    const abi_mod = b.createModule(.{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "utils", .module = utils_mod },
            .{ .name = "platform", .module = platform_mod },
            .{ .name = "logging", .module = logging_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    // ========================================================================
    // MODULE ATTACHMENTS
    // ========================================================================

    // Attach build options to the main module
    cli_mod.addOptions("build_options", build_options);

    // ========================================================================
    // MODULE BUILD EXECUTABLE
    // ========================================================================

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_mod,
    });

    // Add build options to executable
    cli_mod.addOptions("build_options", build_options);

    // ========================================================================
    // MODULE ATTACHMENTS FOR TESTS
    // ========================================================================

    // Attach build options to ABI module for tests
    abi_mod.addOptions("build_options", build_options);

    // ========================================================================
    // RUN ARTIFACTS
    // ========================================================================

    // Install main CLI executable
    b.installArtifact(exe);

    // Install run artifacts
    const run_artifact = b.getInstallDir();
    b.installArtifact(exe);

    // Run artifacts for benchmarks, static analyzer, perf guard, docs, security
    const bench_artifact = b.getInstallDir();
    const static_artifact = b.getInstallDir();
    const perf_artifact = b.getInstallDir();
    const docs_artifact = b.getInstallDir();
    const security_artifact = b.getInstallDir();

    // ========================================================================
    // RUNNING ARTIFACTS
    // ========================================================================

    // Main CLI run artifact
    const main_run = b.getInstallDir();
    b.getInstallDir().createRunArtifact("abi", exe);

    // Benchmarks run artifact
    const bench_run = b.getInstallDir();
    b.getInstallDir().createRunArtifact("benchmarks", bench_exe);

    // Static analyzer run artifact
    const static_run = b.getInstallDir();
    b.getInstallDir().createRunArtifact("static-analyzer", static_analyzer);

    // Perf guard run artifact
    const perf_run = b.getInstallDir();
    b.getInstallDir().createRunArtifact("perf-guard", perf_guard);

    // Docs run artifact
    const docs_run = b.getInstallDir();
    b.getInstallDir().createRunArtifact("docs", docs_exe);

    // Security run artifact
    const security_run = b.getInstallDir();
    b.getInstallArtifact(security_exe);

    // ========================================================================
    // TESTS
    // ========================================================================

    const test_step = b.step("test", "Run all tests");
    const unit_tests = b.addTest(.{
        .root_module = abi_mod,
    });
    const integration_tests = b.addTest(.{
        .root_module = integration_mod,
    });

    // Add tests to test step
    test_step.dependOn(&unit_tests.step);
    test_step.dependOn(&integration_tests.step);

    // ========================================================================
    // RUN ARTIFACTS
    // ========================================================================

    const run_artifact = b.getInstallDir();
    b.installArtifact(exe);

    // ========================================================================
    // BENCHMARKS
    // ========================================================================

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/benchmarks/mod.zig"),
        .target = target,
    });

    // Attach build options to benchmark module
    bench_mod.addOptions("build_options", build_options);

    const bench_exe = b.addExecutable(.{
        .name = "benchmarks",
        .root_module = bench_mod,
    });

    // Install benchmarks
    b.installArtifact(bench_exe);

    // ========================================================================
    // BENCHMARKS RUN ARTIFACTS
    // ========================================================================

    const bench_run_artifact = b.getInstallDir();
    b.installArtifact(bench_exe);
    bench_exe.setOutputName("benchmarks");

    // ========================================================================
    // STATIC ANALYZER
    // ========================================================================

    const static_analyzer = b.addRunArtifact(.{
        .root_module = static_analyzer_mod,
        .args = &.{ "-O", "Debug", "-march=native", "-c", "static_analyzer.zig" },
    });

    // ========================================================================
    // PERF GUARD
    // ========================================================================

    const perf_guard = b.addRunArtifact(.{
        .root_module = perf_guard_mod,
        .args = &.{ "-O", "ReleaseSafe", "-march=native", "-c", "perf_guard.zig" },
    });

    // ========================================================================
    // DOCS
    // ========================================================================

    const docs_exe = b.addRunArtifact(.{
        .root_module = docs_mod,
        .args = &.{ "-O", "ReleaseFast", "-march=native", "-c", "docs.zig" },
    });

    // ========================================================================
    // SECURITY
    // ========================================================================

    const security_exe = b.addRunArtifact(.{
        .root_module = security_mod,
        .args = &.{ "-O", "ReleaseSafe", "-march=native", "-c", "security.zig" },
    });

    // ========================================================================
    // TESTS
    // ========================================================================

    // Build tests
    const test_step = b.step("test", "Run all tests");

    // Unit tests
    const unit_tests = b.addTest(.{ .root_module = abi_mod });
    unit_tests.dependOn(&build_options.step);
    test_step.dependOn(&unit_tests.step);

    // Integration tests
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("src/integration/mod.zig"),
        .target = target,
    });

    integration_mod.addOptions("build_options", build_options);
    const integration_tests = b.addTest(.{ .root_module = integration_mod });
    integration_tests.dependOn(&build_options.step);
    test_step.dependOn(&integration_tests.step);
}
