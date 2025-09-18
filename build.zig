const std = @import("std");

/// @Definitions
/// Production-ready Zig build configuration for a high-performance vector database with neural network acceleration.
///
/// **Core Architecture:**
/// - Cross-platform compatibility: Windows, macOS, Linux, WASM, RISC-V
/// - Modular design with clean separation between CLI, database, GPU, and AI components
/// - Multiple optimization levels with comprehensive feature flags
/// - Production-grade error handling and resource management
///
/// **GPU Support Matrix:**
/// - CUDA support for NVIDIA GPUs (configurable via --enable_cuda flag)
/// - Vulkan/SPIRV support for cross-platform GPU computing (enabled by default)
/// - Platform-specific GPU frameworks:
///   * Metal (macOS) - Native Apple GPU acceleration
///   * DirectX 12 (Windows) - Microsoft GPU stack
///   * OpenGL/Vulkan (Linux) - Cross-vendor GPU support
/// - Automatic library path detection and graceful fallback handling
/// - Runtime GPU capability detection and optimal backend selection
///
/// **Module Structure:**
/// - abi: Core API module with comprehensive vector database operations
/// - cli: Command-line interface with advanced argument parsing
/// - gpu: Multi-backend GPU management with automatic fallback
/// - plugins: Dynamic plugin system with hot-reload capability
/// - services: External service integrations (weather, web server)
/// - Database integration with HNSW vector search and persistence
///
/// **Test Infrastructure:**
/// - Unit tests: Comprehensive coverage of all core modules
/// - Integration tests: End-to-end functionality validation
/// - Heavy tests: Database and HNSW operations (optional via --heavy-tests)
/// - Cross-platform test matrix: Compatibility verification across targets
/// - GPU test suites: Backend-specific validation and performance testing
/// - Memory leak detection and resource cleanup verification
///
/// **Benchmark Suite:**
/// - Unified benchmark system with categorized performance tests
/// - Neural network inference and training benchmarks
/// - Vector database operation benchmarks (insert, search, update, delete)
/// - SIMD micro-benchmarks for optimization validation
/// - Performance regression guards for CI/CD integration
/// - Memory usage and allocation pattern analysis
///
/// **Development Tools:**
/// - Static analysis with comprehensive linting
/// - Code coverage reporting with kcov integration
/// - API documentation generation (HTML + Markdown)
/// - Performance profiling with detailed metrics
/// - Cross-compilation verification for production targets
/// - Continuous integration performance gates
///
/// **Production Features:**
/// - Memory-mapped file I/O for large datasets
/// - Configurable logging levels and output formats
/// - Graceful error handling with detailed diagnostics
/// - Resource cleanup and leak prevention
/// - Production-ready configuration management
/// - Performance monitoring and telemetry hooks
///
/// **Build Steps Available:**
/// - `zig build` - Build optimized CLI executable
/// - `zig build test-all` - Run comprehensive test suite
/// - `zig build bench-all` - Execute all benchmark categories
/// - `zig build cross-platform` - Verify multi-target compatibility
/// - `zig build gpu-verify` - Validate GPU backend functionality
/// - `zig build docs` - Generate complete API documentation
/// - `zig build coverage` - Create detailed code coverage reports
/// - `zig build perf-ci` - Run performance regression testing
/// - `zig build analyze` - Execute static code analysis
pub fn build(b: *std.Build) void {
    // Standard target and optimization configuration
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Production build options with comprehensive feature flags
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "simd_level", "auto");
    build_options.addOption(bool, "gpu", true);
    build_options.addOption(bool, "simd", true);
    build_options.addOption(bool, "neural_accel", false);
    build_options.addOption(bool, "webgpu", true);
    build_options.addOption(bool, "hot_reload", false);
    build_options.addOption(bool, "enable_tracy", false);
    build_options.addOption(bool, "is_wasm", false);

    // GPU configuration with production defaults
    const enable_cuda = b.option(bool, "enable_cuda", "Enable CUDA support") orelse true;
    const enable_spirv = b.option(bool, "enable_spirv", "Enable SPIRV compilation support") orelse true;
    const enable_wasm = b.option(bool, "enable_wasm", "Enable WebAssembly support") orelse true;
    const enable_cross_compilation = b.option(bool, "enable_cross_compilation", "Enable cross-compilation support") orelse true;
    const cuda_path_default = if (target.result.os.tag == .windows) "C:\\Users\\donald\\scoop\\apps\\cuda\\current" else "";
    const cuda_path = b.option([]const u8, "cuda_path", "Path to CUDA installation") orelse cuda_path_default;
    const vulkan_sdk_path_default = if (target.result.os.tag == .windows) "C:\\VulkanSDK\\1.4.321.1" else "";
    const vulkan_sdk_path = b.option([]const u8, "vulkan_sdk_path", "Path to Vulkan SDK") orelse vulkan_sdk_path_default;

    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption(bool, "enable_spirv", enable_spirv);
    build_options.addOption(bool, "enable_wasm", enable_wasm);
    build_options.addOption(bool, "enable_cross_compilation", enable_cross_compilation);
    build_options.addOption([]const u8, "cuda_path", cuda_path);
    build_options.addOption([]const u8, "vulkan_sdk_path", vulkan_sdk_path);

    // Production-grade GPU dependency management
    const applyGPUDeps = struct {
        fn apply(builder: *std.Build, exe: *std.Build.Step.Compile, tgt: std.Build.ResolvedTarget, cuda_enabled: bool, spirv_enabled: bool, cuda_lib_path: []const u8, vulkan_lib_path: []const u8) void {
            // CUDA support with error handling
            if (cuda_enabled) {
                if (cuda_lib_path.len > 0) {
                    if (std.fs.path.isAbsolute(cuda_lib_path)) {
                        exe.addLibraryPath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "lib", "x64" }) catch "" });
                        exe.addLibraryPath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "lib64" }) catch "" });
                        exe.addLibraryPath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "lib" }) catch "" });
                    } else {
                        exe.addLibraryPath(builder.path(std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "lib64" }) catch ""));
                        exe.addLibraryPath(builder.path(std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "lib" }) catch ""));
                    }
                }
                exe.linkSystemLibrary("cuda");
                exe.linkSystemLibrary("cudart");
                exe.linkSystemLibrary("cublas");
                exe.linkSystemLibrary("cusolver");
                exe.linkSystemLibrary("cusparse");
                if (cuda_lib_path.len > 0) {
                    if (std.fs.path.isAbsolute(cuda_lib_path)) {
                        exe.addIncludePath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "include" }) catch "" });
                    } else {
                        exe.addIncludePath(builder.path(std.fs.path.join(builder.allocator, &.{ cuda_lib_path, "include" }) catch ""));
                    }
                }
            }

            // Vulkan/SPIRV support with platform optimization
            if (spirv_enabled) {
                if (vulkan_lib_path.len > 0) {
                    if (std.fs.path.isAbsolute(vulkan_lib_path)) {
                        exe.addLibraryPath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ vulkan_lib_path, "Lib" }) catch "" });
                        exe.addIncludePath(.{ .cwd_relative = std.fs.path.join(builder.allocator, &.{ vulkan_lib_path, "Include" }) catch "" });
                    } else {
                        exe.addLibraryPath(builder.path(std.fs.path.join(builder.allocator, &.{ vulkan_lib_path, "lib" }) catch ""));
                        exe.addIncludePath(builder.path(std.fs.path.join(builder.allocator, &.{ vulkan_lib_path, "include" }) catch ""));
                    }
                }

                // Platform-specific Vulkan configuration
                if (tgt.result.os.tag == .windows) {
                    exe.linkSystemLibrary("vulkan-1");
                } else if (tgt.result.os.tag == .linux) {
                    exe.linkSystemLibrary("vulkan");
                    exe.linkLibC();
                } else if (tgt.result.os.tag == .macos) {
                    exe.linkSystemLibrary("MoltenVK");
                    exe.linkLibC();
                }

                // SPIRV compilation tools with graceful fallback
                if (tgt.result.os.tag == .windows) {
                    exe.linkSystemLibrary("SPIRV-Tools");
                    exe.linkSystemLibrary("SPIRV-Tools-opt");
                    exe.linkSystemLibrary("glslang");
                    exe.linkSystemLibrary("glslang-default-resource-limits");
                } else {
                    exe.linkSystemLibrary("SPIRV-Tools");
                    exe.linkSystemLibrary("SPIRV-Tools-opt");
                    exe.linkSystemLibrary("glslang");
                    exe.linkSystemLibrary("glslang-default-resource-limits");
                }
            }

            // Platform-optimized GPU frameworks
            if (tgt.result.os.tag == .macos) {
                // Apple Metal framework stack
                exe.linkFramework("Metal");
                exe.linkFramework("MetalKit");
                exe.linkFramework("MetalPerformanceShaders");
                exe.linkFramework("Foundation");
                exe.linkFramework("CoreGraphics");
                exe.linkFramework("QuartzCore");
                exe.linkLibC();
            } else if (tgt.result.os.tag == .windows) {
                // Microsoft DirectX 12 stack
                exe.addLibraryPath(.{ .cwd_relative = "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.26100.0\\um\\x64" });
                exe.linkSystemLibrary("d3d12");
                exe.linkSystemLibrary("dxgi");
                exe.linkSystemLibrary("d3dcompiler");
                exe.linkSystemLibrary("dxguid");
                exe.linkSystemLibrary("user32");
                exe.linkSystemLibrary("kernel32");
                exe.linkSystemLibrary("gdi32");
            } else if (tgt.result.os.tag == .linux) {
                // Linux graphics and system libraries
                exe.linkSystemLibrary("X11");
                exe.linkSystemLibrary("Xrandr");
                exe.linkSystemLibrary("Xinerama");
                exe.linkSystemLibrary("Xcursor");
                exe.linkSystemLibrary("Xi");
                exe.linkSystemLibrary("Xext");
                exe.linkSystemLibrary("pthread");
                exe.linkSystemLibrary("dl");
                exe.linkSystemLibrary("m");
                exe.linkLibC();
            }
        }
    }.apply;

    // Core API module - foundation of the system
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI module with advanced argument processing
    const cli_mod = b.addModule("cli", .{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });

    // Main CLI executable with production optimizations
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = cli_mod,
    });
    cli_exe.root_module.addOptions("options", build_options);

    // Apply GPU dependencies with error handling
    applyGPUDeps(b, cli_exe, target, enable_cuda, enable_spirv, cuda_path, vulkan_sdk_path);

    b.installArtifact(cli_exe);

    // CLI execution step
    const run_step = b.step("run", "Run the CLI application");
    const cli_run = b.addRunArtifact(cli_exe);
    run_step.dependOn(&cli_run.step);

    // Core unit testing infrastructure
    const unit_tests_mod = b.addModule("unit_tests", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = unit_tests_mod,
    });
    unit_tests.root_module.addOptions("options", build_options);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Support modules for comprehensive testing
    const weather_mod = b.addModule("weather", .{
        .root_source_file = b.path("src/services/weather.zig"),
        .target = target,
        .optimize = optimize,
    });

    const gpu_mod = b.addModule("gpu", .{
        .root_source_file = b.path("src/gpu/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const web_server_mod = b.addModule("web_server", .{
        .root_source_file = b.path("src/server/web_server.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });

    // Comprehensive test suite
    const test_files = [_]struct { path: []const u8, imports: []const std.Build.Module.Import }{
        .{ .path = "tests/test_ai.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .path = "tests/test_cli_integration.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .path = "tests/test_database.zig", .imports = &.{} },
        .{ .path = "tests/test_memory_management.zig", .imports = &.{} },
        .{ .path = "tests/test_simd_vector.zig", .imports = &.{} },
        .{ .path = "tests/test_config_validation.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .path = "tests/test_weather.zig", .imports = &.{.{ .name = "weather", .module = weather_mod }} },
        .{ .path = "tests/test_gpu.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .path = "tests/test_gpu_renderer.zig", .imports = &.{.{ .name = "gpu", .module = gpu_mod }} },
        .{ .path = "tests/test_gpu_advanced.zig", .imports = &.{.{ .name = "gpu", .module = gpu_mod }} },
        .{ .path = "tests/test_gpu_backend_manager.zig", .imports = &.{.{ .name = "gpu", .module = gpu_mod }} },
        .{ .path = "tests/test_gpu_ai_acceleration.zig", .imports = &.{.{ .name = "gpu", .module = gpu_mod }} },
        .{ .path = "tests/test_web_server.zig", .imports = &.{.{ .name = "web_server", .module = web_server_mod }} },
    };

    for (test_files, 0..) |test_file, i| {
        const module_name = b.fmt("test_{d}", .{i});
        const mod = b.addModule(module_name, .{
            .root_source_file = b.path(test_file.path),
            .target = target,
            .optimize = optimize,
            .imports = test_file.imports,
        });
        const t = b.addTest(.{ .root_module = mod });
        t.root_module.addOptions("options", build_options);
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Heavy testing infrastructure
    const enable_heavy = b.option(bool, "heavy-tests", "Enable heavy DB/HNSW tests") orelse false;
    const heavy_step = b.step("test-heavy", "Run heavy HNSW/integration tests");

    // HNSW performance tests
    const hnsw_mod = b.addModule("hnsw_tests", .{
        .root_source_file = b.path("tests/test_database_hnsw.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const hnsw_tests = b.addTest(.{ .root_module = hnsw_mod });
    hnsw_tests.root_module.addOptions("options", build_options);
    const run_hnsw = b.addRunArtifact(hnsw_tests);
    heavy_step.dependOn(&run_hnsw.step);

    // Database integration tests
    const db_int_mod = b.addModule("db_int_tests", .{
        .root_source_file = b.path("tests/test_database_integration.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const db_int_tests = b.addTest(.{ .root_module = db_int_mod });
    db_int_tests.root_module.addOptions("options", build_options);
    const run_db_int = b.addRunArtifact(db_int_tests);
    heavy_step.dependOn(&run_db_int.step);

    // Socket-level web server stress tests
    const web_socket_mod = b.addModule("web_socket_tests", .{
        .root_source_file = b.path("tests/test_web_server_socket.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "web_server", .module = web_server_mod }},
    });
    const web_socket_tests = b.addTest(.{ .root_module = web_socket_mod });
    web_socket_tests.root_module.addOptions("options", build_options);
    const run_web_socket = b.addRunArtifact(web_socket_tests);
    heavy_step.dependOn(&run_web_socket.step);

    if (enable_heavy) {
        test_step.dependOn(heavy_step);
    }

    // Production benchmark suite
    const benchmark_configs = [_]struct { name: []const u8, path: []const u8, imports: []const std.Build.Module.Import }{
        .{ .name = "database_benchmark", .path = "benchmarks/database_benchmark.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .name = "benchmark_main", .path = "benchmarks/main.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .name = "neural_benchmark", .path = "benchmarks/benchmark_suite.zig", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .name = "simple_benchmark", .path = "benchmarks/simple_benchmark.zig", .imports = &.{} },
    };

    const benchmark_steps = [_]struct { name: []const u8, desc: []const u8, exe_idx: usize }{
        .{ .name = "benchmark-db", .desc = "Run database performance benchmarks", .exe_idx = 0 },
        .{ .name = "benchmark", .desc = "Run unified benchmark suite", .exe_idx = 1 },
        .{ .name = "benchmark-neural", .desc = "Run neural network benchmarks", .exe_idx = 2 },
        .{ .name = "benchmark-simple", .desc = "Run simple VDBench-style benchmarks", .exe_idx = 3 },
    };

    var benchmark_exes: [benchmark_configs.len]*std.Build.Step.Compile = undefined;

    for (benchmark_configs, 0..) |config, i| {
        const mod = b.addModule(b.fmt("{s}_bench", .{config.name}), .{
            .root_source_file = b.path(config.path),
            .target = target,
            .optimize = optimize,
            .imports = config.imports,
        });
        benchmark_exes[i] = b.addExecutable(.{
            .name = config.name,
            .root_module = mod,
        });
    }

    for (benchmark_steps) |step_config| {
        const step = b.step(step_config.name, step_config.desc);
        const run_benchmark = b.addRunArtifact(benchmark_exes[step_config.exe_idx]);
        if (step_config.exe_idx == 1) run_benchmark.addArg("all"); // Unified benchmark
        step.dependOn(&run_benchmark.step);
    }

    // Aggregate benchmark step
    const bench_all_step = b.step("bench-all", "Run all benchmark suites");
    for (benchmark_steps) |step_config| {
        const run_benchmark = b.addRunArtifact(benchmark_exes[step_config.exe_idx]);
        if (step_config.exe_idx == 1) run_benchmark.addArg("all"); // Unified benchmark
        bench_all_step.dependOn(&run_benchmark.step);
    }

    // SIMD micro-benchmark
    const simd_micro_mod = b.addModule("simd_micro_bench", .{
        .root_source_file = b.path("benchmarks/simd_micro.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const simd_micro_exe = b.addExecutable(.{ .name = "simd-micro", .root_module = simd_micro_mod });
    simd_micro_exe.root_module.addOptions("options", build_options);
    const run_simd_bench = b.addRunArtifact(simd_micro_exe);
    const simd_bench_step = b.step("bench-simd", "Run SIMD micro-benchmark");
    simd_bench_step.dependOn(&run_simd_bench.step);

    // Development and analysis tools
    const tool_configs = [_]struct { name: []const u8, path: []const u8, desc: []const u8, imports: []const std.Build.Module.Import }{
        .{ .name = "static_analysis", .path = "tools/static_analysis.zig", .desc = "Run static analysis", .imports = &.{} },
        .{ .name = "docs_generator", .path = "tools/docs_generator.zig", .desc = "Generate API documentation", .imports = &.{} },
        .{ .name = "performance_profiler", .path = "tools/performance_profiler.zig", .desc = "Run performance profiling", .imports = &.{} },
        .{ .name = "perf_guard", .path = "tools/perf_guard.zig", .desc = "Run performance regression guard", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
        .{ .name = "performance_ci", .path = "tools/performance_ci.zig", .desc = "Run performance CI/CD testing", .imports = &.{.{ .name = "abi", .module = abi_mod }} },
    };

    for (tool_configs, 0..) |config, i| {
        const module_name = b.fmt("tool_{d}_{s}", .{ i, config.name });
        const mod = b.addModule(module_name, .{
            .root_source_file = b.path(config.path),
            .target = target,
            .optimize = if (std.mem.eql(u8, config.name, "performance_profiler")) .ReleaseFast else optimize,
            .imports = config.imports,
        });
        const exe = b.addExecutable(.{ .name = config.name, .root_module = mod });
        exe.root_module.addOptions("options", build_options);
        b.installArtifact(exe);

        const run_tool = b.addRunArtifact(exe);
        if (std.mem.eql(u8, config.name, "perf_guard")) {
            const perf_threshold_opt = b.option(u64, "perf-threshold-ns", "Average search time threshold (ns)");
            run_tool.addArg(if (perf_threshold_opt) |t| b.fmt("{d}", .{t}) else "50000000");
        }

        const step_name = if (std.mem.eql(u8, config.name, "static_analysis")) "analyze" else if (std.mem.eql(u8, config.name, "docs_generator")) "docs" else if (std.mem.eql(u8, config.name, "performance_profiler")) "profile" else if (std.mem.eql(u8, config.name, "perf_guard")) "perf-guard" else "perf-ci";

        const step = b.step(step_name, config.desc);
        step.dependOn(&run_tool.step);
    }

    // Windows network diagnostic tool
    const network_test_mod = b.addModule("windows_network_test", .{
        .root_source_file = b.path("tools/windows_network_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const network_test = b.addExecutable(.{
        .name = "windows_network_test",
        .root_module = network_test_mod,
    });
    network_test.root_module.addOptions("options", build_options);
    network_test.root_module.link_libc = true;
    if (target.result.os.tag == .windows) {
        b.installArtifact(network_test);
    }
    const run_network_test = b.addRunArtifact(network_test);
    const network_test_step = b.step("test-network", "Run Windows network diagnostic");
    network_test_step.dependOn(&run_network_test.step);

    // Plugin system testing
    const plugin_mod = b.addModule("plugin_tests", .{
        .root_source_file = b.path("src/plugins/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    const plugin_tests = b.addTest(.{ .root_module = plugin_mod });
    plugin_tests.root_module.addOptions("options", build_options);
    const run_plugin_tests = b.addRunArtifact(plugin_tests);
    const plugin_test_step = b.step("test-plugins", "Run plugin system tests");
    plugin_test_step.dependOn(&run_plugin_tests.step);

    // Code coverage with kcov integration
    const coverage_step = b.step("coverage", "Generate code coverage report");
    const coverage_mod = b.addModule("coverage", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const coverage_tests = b.addTest(.{ .root_module = coverage_mod });
    coverage_tests.root_module.addOptions("options", build_options);
    const kcov_exe = b.addSystemCommand(&[_][]const u8{
        "kcov",
        "--clean",
        "--include-pattern=src/",
        "--exclude-pattern=tests/",
        b.pathJoin(&.{ b.install_path, "coverage" }),
    });
    kcov_exe.addArtifactArg(coverage_tests);
    coverage_step.dependOn(&kcov_exe.step);

    // GPU verification and demo
    const gpu_demo_mod = b.addModule("gpu_demo", .{
        .root_source_file = b.path("src/gpu/demo/gpu_demo.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gpu", .module = gpu_mod }},
    });
    const gpu_demo_exe = b.addExecutable(.{
        .name = "gpu_demo",
        .root_module = gpu_demo_mod,
    });
    gpu_demo_exe.root_module.addOptions("options", build_options);

    // Apply GPU dependencies for hardware acceleration (temporarily disabled due to MSVC linking issues)
    // applyGPUDeps(b, gpu_demo_exe, target, enable_cuda, enable_spirv, cuda_path, vulkan_sdk_path);

    b.installArtifact(gpu_demo_exe);
    const run_gpu_demo = b.addRunArtifact(gpu_demo_exe);

    // Enhanced GPU demo with advanced library integration
    const enhanced_gpu_demo_mod = b.addModule("enhanced_gpu_demo", .{
        .root_source_file = b.path("src/gpu/demo/enhanced_gpu_demo.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gpu", .module = gpu_mod }},
    });
    const enhanced_gpu_demo_exe = b.addExecutable(.{
        .name = "enhanced_gpu_demo",
        .root_module = enhanced_gpu_demo_mod,
    });
    enhanced_gpu_demo_exe.root_module.addOptions("options", build_options);

    b.installArtifact(enhanced_gpu_demo_exe);
    const run_enhanced_gpu_demo = b.addRunArtifact(enhanced_gpu_demo_exe);
    const gpu_demo_step = b.step("gpu-demo", "Run GPU backend manager demo");
    gpu_demo_step.dependOn(&run_gpu_demo.step);

    const enhanced_gpu_demo_step = b.step("enhanced-gpu-demo", "Run enhanced GPU demo with advanced library integration");
    enhanced_gpu_demo_step.dependOn(&run_enhanced_gpu_demo.step);

    // Advanced GPU demo with next-level features
    const advanced_gpu_demo_mod = b.addModule("advanced_gpu_demo", .{
        .root_source_file = b.path("src/gpu/demo/advanced_gpu_demo.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gpu", .module = gpu_mod }},
    });
    const advanced_gpu_demo_exe = b.addExecutable(.{
        .name = "advanced_gpu_demo",
        .root_module = advanced_gpu_demo_mod,
    });
    advanced_gpu_demo_exe.root_module.addOptions("options", build_options);

    b.installArtifact(advanced_gpu_demo_exe);
    const run_advanced_gpu_demo = b.addRunArtifact(advanced_gpu_demo_exe);

    const advanced_gpu_demo_step = b.step("advanced-gpu-demo", "Run advanced GPU demo with next-level features");
    advanced_gpu_demo_step.dependOn(&run_advanced_gpu_demo.step);

    // Standalone AI/ML Acceleration Demo (no GPU dependencies)
    const gpu_ai_demo_mod = b.addModule("gpu_ai_demo", .{
        .root_source_file = b.path("examples/gpu_ai_acceleration_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    const gpu_ai_demo_exe = b.addExecutable(.{
        .name = "gpu_ai_acceleration_demo",
        .root_module = gpu_ai_demo_mod,
    });
    gpu_ai_demo_exe.root_module.addOptions("options", build_options);
    b.installArtifact(gpu_ai_demo_exe);
    const run_gpu_ai_demo = b.addRunArtifact(gpu_ai_demo_exe);
    const gpu_ai_demo_step = b.step("gpu-ai-demo", "Run GPU AI/ML acceleration demo");
    gpu_ai_demo_step.dependOn(&run_gpu_ai_demo.step);

    // Neural Network Integration Demo
    const gpu_nn_integration_mod = b.addModule("gpu_nn_integration", .{
        .root_source_file = b.path("examples/gpu_neural_network_integration.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const gpu_nn_integration_exe = b.addExecutable(.{
        .name = "gpu_neural_network_integration",
        .root_module = gpu_nn_integration_mod,
    });
    gpu_nn_integration_exe.root_module.addOptions("options", build_options);
    b.installArtifact(gpu_nn_integration_exe);
    const run_gpu_nn_integration = b.addRunArtifact(gpu_nn_integration_exe);
    const gpu_nn_integration_step = b.step("gpu-nn-integration", "Run GPU neural network integration demo");
    gpu_nn_integration_step.dependOn(&run_gpu_nn_integration.step);

    // Transformer Architecture Example
    const transformer_example_mod = b.addModule("transformer_example", .{
        .root_source_file = b.path("examples/transformer_example.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
        },
    });
    const transformer_example_exe = b.addExecutable(.{
        .name = "transformer_example",
        .root_module = transformer_example_mod,
    });
    transformer_example_exe.root_module.addOptions("options", build_options);
    b.installArtifact(transformer_example_exe);
    const run_transformer_example = b.addRunArtifact(transformer_example_exe);
    const transformer_example_step = b.step("transformer-example", "Run transformer architecture example");
    transformer_example_step.dependOn(&run_transformer_example.step);

    const gpu_verify_step = b.step("gpu-verify", "Verify GPU functionality and backends");
    gpu_verify_step.dependOn(&run_gpu_demo.step);

    // WebAssembly compilation step
    if (enable_wasm) {
        const wasm_step = b.step("wasm", "Compile to WebAssembly");

        // High-performance WASM build
        const wasm_high_perf_mod = b.addModule("wasm_high_perf", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .wasm32,
                .os_tag = .freestanding,
            }),
            .optimize = .ReleaseSmall,
        });
        const wasm_high_perf = b.addExecutable(.{
            .name = "abi_high_perf",
            .root_module = wasm_high_perf_mod,
        });
        wasm_high_perf.root_module.addOptions("abi", build_options);
        wasm_step.dependOn(&wasm_high_perf.step);

        // Size-optimized WASM build
        const wasm_size_opt_mod = b.addModule("wasm_size_opt", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .wasm32,
                .os_tag = .freestanding,
            }),
            .optimize = .ReleaseSmall,
        });
        const wasm_size_opt = b.addExecutable(.{
            .name = "abi_size_opt",
            .root_module = wasm_size_opt_mod,
        });
        wasm_size_opt.root_module.addOptions("abi", build_options);
        wasm_step.dependOn(&wasm_size_opt.step);
    }

    // Cross-compilation step
    if (enable_cross_compilation) {
        const cross_compile_step = b.step("cross-compile", "Cross-compile for multiple architectures");

        // ARM64 Linux build
        const arm64_linux_mod = b.addModule("arm64_linux", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .aarch64,
                .os_tag = .linux,
            }),
            .optimize = optimize,
        });
        const arm64_linux = b.addExecutable(.{
            .name = "abi_arm64_linux",
            .root_module = arm64_linux_mod,
        });
        arm64_linux.root_module.addOptions("abi", build_options);
        cross_compile_step.dependOn(&arm64_linux.step);

        // RISC-V Linux build
        const riscv64_linux_mod = b.addModule("riscv64_linux", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .riscv64,
                .os_tag = .linux,
            }),
            .optimize = optimize,
        });
        const riscv64_linux = b.addExecutable(.{
            .name = "abi_riscv64_linux",
            .root_module = riscv64_linux_mod,
        });
        riscv64_linux.root_module.addOptions("abi", build_options);
        cross_compile_step.dependOn(&riscv64_linux.step);

        // ARM64 macOS build
        const arm64_macos_mod = b.addModule("arm64_macos", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .aarch64,
                .os_tag = .macos,
            }),
            .optimize = optimize,
        });
        const arm64_macos = b.addExecutable(.{
            .name = "abi_arm64_macos",
            .root_module = arm64_macos_mod,
        });
        arm64_macos.root_module.addOptions("abi", build_options);
        cross_compile_step.dependOn(&arm64_macos.step);

        // x86_64 Windows build
        const x86_64_windows_mod = b.addModule("x86_64_windows", .{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .x86_64,
                .os_tag = .windows,
            }),
            .optimize = optimize,
        });
        const x86_64_windows = b.addExecutable(.{
            .name = "abi_x86_64_windows",
            .root_module = x86_64_windows_mod,
        });
        x86_64_windows.root_module.addOptions("abi", build_options);
        cross_compile_step.dependOn(&x86_64_windows.step);
    }

    // Integration testing
    const integration_mod = b.addModule("integration_tests", .{
        .root_source_file = b.path("tests/integration_test_suite.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "abi", .module = abi_mod }},
    });
    const integration_tests = b.addExecutable(.{
        .name = "integration_tests",
        .root_module = integration_mod,
    });
    integration_tests.root_module.addOptions("options", build_options);
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Comprehensive test execution
    const all_tests_step = b.step("test-all", "Run all tests (unit + integration + heavy)");
    all_tests_step.dependOn(test_step);
    all_tests_step.dependOn(integration_test_step);
    if (enable_heavy) all_tests_step.dependOn(heavy_step);

    // Cross-platform test matrix
    const test_matrix_step = b.step("test-matrix", "Run unit tests across multiple targets");
    const test_targets = [_]std.Target.Query{
        .{}, // native
        .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
        .{ .cpu_arch = .aarch64, .os_tag = .macos },
    };
    for (test_targets, 0..) |tq, i| {
        const resolved = b.resolveTargetQuery(tq);
        const module_name = b.fmt("test_matrix_{d}", .{i});
        const unit_mod_matrix = b.addModule(module_name, .{
            .root_source_file = b.path("src/root.zig"),
            .target = resolved,
            .optimize = optimize,
        });
        const unit_tests_matrix = b.addTest(.{ .root_module = unit_mod_matrix });
        unit_tests_matrix.root_module.addOptions("options", build_options);
        const run_unit_tests_matrix = b.addRunArtifact(unit_tests_matrix);
        run_unit_tests_matrix.skip_foreign_checks = true;
        test_matrix_step.dependOn(&run_unit_tests_matrix.step);
    }

    // Production cross-platform verification
    const cross_platform_step = b.step("cross-platform", "Verify cross-platform compatibility");
    const cross_targets = [_][]const u8{
        "x86_64-linux-gnu",
        "aarch64-linux-gnu",
        "x86_64-macos",
        "aarch64-macos",
        "riscv64-linux-gnu",
        "x86_64-windows-gnu",
        "aarch64-windows-gnu",
        "x86_64-windows-msvc",
        "aarch64-windows-msvc",
        "wasm32-wasi",
        "wasm32-freestanding",
    };

    for (cross_targets) |cross_target| {
        const cross_target_query = std.Target.Query.parse(.{ .arch_os_abi = cross_target }) catch unreachable;
        const cross_target_resolved = b.resolveTargetQuery(cross_target_query);
        const os_tag = cross_target_resolved.result.os.tag;
        const abi_tag = cross_target_resolved.result.abi;
        const arch_tag = cross_target_resolved.result.cpu.arch;

        const root_src_path: []const u8 = blk: {
            if (arch_tag == .wasm32 and os_tag == .wasi) break :blk "src/cli/wasm_wasi_stub.zig";
            if (arch_tag == .wasm32 and os_tag == .freestanding) break :blk "src/cli/wasm_freestanding_stub.zig";
            break :blk "src/cli/main.zig";
        };

        const module_name = b.fmt("cross_cli_{s}", .{cross_target});
        const cross_cli_mod = b.addModule(module_name, .{
            .root_source_file = b.path(root_src_path),
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

        if (os_tag == .linux or os_tag == .macos or (os_tag == .windows and abi_tag == .gnu) or os_tag == .wasi) {
            cross_cli_exe.linkLibC();
        }

        const install_cross = b.addInstallArtifact(cross_cli_exe, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("cross/{s}", .{cross_target}) } },
        });
        cross_platform_step.dependOn(&install_cross.step);
    }
}
