const std = @import("std");

<<<<<<< HEAD
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
    const is_freebsd = target_info.os.tag == .freebsd;
    const is_openbsd = target_info.os.tag == .openbsd;
    const is_netbsd = target_info.os.tag == .netbsd;
    const is_unix = is_linux or is_macos or is_freebsd or is_openbsd or is_netbsd;
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

    // Create reusable build options module
    const build_options_mod = build_options.createModule();

    // ========================================================================
    // CORE MODULE SYSTEM
    // ========================================================================

    // ========================================================================
    // FOUNDATION MODULES
    // ========================================================================

    // Core foundation module - provides basic types, utilities, and cross-platform abstractions
    const core_mod = b.createModule(.{
        .root_source_file = b.path("src/core/mod.zig"),
=======
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
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

<<<<<<< HEAD
    // ========================================================================
    // PERFORMANCE MODULES
    // ========================================================================

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

    // ========================================================================
    // AI & MACHINE LEARNING MODULES
    // ========================================================================

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

    // ========================================================================
    // DATA STORAGE MODULES
    // ========================================================================

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

    // ========================================================================
    // NETWORKING MODULES
    // ========================================================================

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

    // Enhanced HTTP client with retry/backoff and proxy support
    const http_client_mod = b.createModule(.{
        .root_source_file = b.path("src/net/http_client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // ========================================================================
    // EXTENSIBILITY MODULES
    // ========================================================================

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

    // ========================================================================
    // APPLICATION MODULES
    // ========================================================================

    // WDBX utilities and CLI components
    const wdbx_mod = b.createModule(.{
        .root_source_file = b.path("src/wdbx/mod.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "simd", .module = simd_mod },
            .{ .name = "database", .module = database_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // ========================================================================
    // PUBLIC API MODULE
    // ========================================================================

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

    // ========================================================================
    // PRIMARY APPLICATIONS
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
=======
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
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
    });
    cli_exe.root_module.addOptions("options", build_options);

    // Apply GPU dependencies with error handling
    applyGPUDeps(b, cli_exe, target, enable_cuda, enable_spirv, cuda_path, vulkan_sdk_path);

<<<<<<< HEAD
    // Set stack size for deep recursion (neural networks)
    cli_exe.stack_size = 8 * 1024 * 1024; // 8MB

    // Platform-specific system library linking
    if (is_windows) {
        cli_exe.linkSystemLibrary("ws2_32"); // Windows Sockets 2
        cli_exe.linkSystemLibrary("kernel32"); // Windows Kernel
        cli_exe.linkSystemLibrary("user32"); // Windows User Interface
        cli_exe.linkSystemLibrary("advapi32"); // Windows Advanced API
    } else if (is_unix) {
        cli_exe.linkSystemLibrary("c"); // C runtime
        cli_exe.linkSystemLibrary("pthread"); // POSIX threads
        cli_exe.linkSystemLibrary("m"); // Math library

        if (is_linux) {
            cli_exe.linkSystemLibrary("dl"); // Dynamic linking
            cli_exe.linkSystemLibrary("rt"); // Real-time extensions
        } else if (is_macos) {
            cli_exe.linkFramework("Foundation"); // macOS Foundation framework
            cli_exe.linkFramework("CoreFoundation"); // Core Foundation
        } else if (is_freebsd or is_openbsd or is_netbsd) {
            cli_exe.linkSystemLibrary("execinfo"); // Backtrace support
        }
    }

    b.installArtifact(cli_exe);

    // Standalone HTTP server for production deployment
    if (!is_wasm) {
        const wdbx_server_exe = b.addExecutable(.{
            .name = if (is_windows) "wdbx_server.exe" else "wdbx_server",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/wdbx/unified.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "database", .module = database_mod },
                    .{ .name = "http", .module = http_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        // Platform-specific system library linking
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

        // Apply minimal platform-specific configurations for benchmarks
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

        // Apply minimal platform-specific configurations for benchmarks
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

        // Apply minimal platform-specific configurations for dev tools
        if (is_windows) {
            static_analysis.linkSystemLibrary("kernel32");
        } else {
            static_analysis.linkSystemLibrary("c");
        }

        b.installArtifact(static_analysis);

        const analysis_step = b.step("analyze", "Run static code analysis");
        const run_analysis = b.addRunArtifact(static_analysis);
        analysis_step.dependOn(&run_analysis.step);

        // Analysis with auto-cleanup
        const analyze_clean_step = b.step("analyze-clean", "Run static analysis then auto-clean");
        analyze_clean_step.dependOn(&run_analysis.step);
        // Note: clean_step is defined later, so we can't reference it here
        // analyze_clean_step.dependOn(clean_step);
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

        // Apply minimal platform-specific configurations for dev tools
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

    // ========================================================================
    // HTTP TESTING & DEMONSTRATION TOOLS
    // ========================================================================

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

        // Platform-specific system library linking
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
                .root_source_file = b.path("examples/demo_http_client.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "abi", .module = abi_mod },
                    .{ .name = "http_client", .module = http_client_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });

        // Platform-specific system library linking
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

        // Demo with auto-cleanup
        const demo_clean_step = b.step("demo-clean", "Run HTTP client demo then auto-clean");
        demo_clean_step.dependOn(&run_demo.step);
        // Note: clean_step is defined later, so we can't reference it here
        // demo_clean_step.dependOn(clean_step);
    }

    // ========================================================================
    // PLATFORM-SPECIFIC TESTING TOOLS
    // ========================================================================

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

    // ========================================================================
    // INTEGRATION TESTING SUITE
    // ========================================================================

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

        // Platform-specific system library linking
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
                .root_source_file = b.path("src/api/c_api.zig"),
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

        // Apply platform-specific configurations
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
                .root_source_file = b.path("src/api/c_api.zig"),
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

    // Add all dependencies
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
=======
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
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
    const run_plugin_tests = b.addRunArtifact(plugin_tests);
    const plugin_test_step = b.step("test-plugins", "Run plugin system tests");
    plugin_test_step.dependOn(&run_plugin_tests.step);

<<<<<<< HEAD
    // WDBX module tests
    const wdbx_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    wdbx_tests.root_module.addImport("core", core_mod);
    wdbx_tests.root_module.addImport("simd", simd_mod);
    wdbx_tests.root_module.addImport("database", database_mod);
    wdbx_tests.root_module.addImport("build_options", build_options_mod);
    const run_wdbx_tests = b.addRunArtifact(wdbx_tests);
    const wdbx_test_step = b.step("test-wdbx", "Run WDBX module tests");
    wdbx_test_step.dependOn(&run_wdbx_tests.step);

    // ========================================================================
    // SPECIALIZED TEST SUITES
    // ========================================================================

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

    // Performance optimization tests
    const performance_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_performance_optimizations.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_performance_tests = b.addRunArtifact(performance_tests);
    const performance_test_step = b.step("test-performance", "Run performance optimization tests");
    performance_test_step.dependOn(&run_performance_tests.step);

    // ========================================================================
    // UTILITY TOOLS
    // ========================================================================

    // HTTP smoke tool
    const http_smoke = b.addExecutable(.{
        .name = if (is_windows) "http_smoke.exe" else "http_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/http_smoke.zig"),
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

    // ========================================================================
    // MAINTENANCE & CLEANUP
    // ========================================================================

    // Cross-platform build artifact cleanup
    const clean_step = b.step("clean", "Remove all build artifacts and test files");
    if (is_windows) {
        const clean_cmd = b.addSystemCommand(&[_][]const u8{
            "cmd",                                      "/c",
            "if exist zig-cache rmdir /s /q zig-cache", "&& if exist zig-out rmdir /s /q zig-out",
            "&& del /q *.wdbx 2>nul",                   "&& del /q *.wal 2>nul",
            "&& echo Clean completed",
        });
        clean_step.dependOn(&clean_cmd.step);
    } else {
        const clean_cmd = b.addSystemCommand(&[_][]const u8{
            "rm",                                                   "-rf",                                                 "zig-cache",               "zig-out",
            "&& find . -name '*.wdbx' -delete 2>/dev/null || true", "&& find . -name '*.wal' -delete 2>/dev/null || true", "&& echo Clean completed",
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
    test_all_step.dependOn(&run_performance_tests.step);

    // Test with auto-cleanup
    const test_clean_step = b.step("test-clean", "Run all tests then auto-clean artifacts");
    test_clean_step.dependOn(&run_unit_tests.step);
    test_clean_step.dependOn(&run_simd_tests.step);
    test_clean_step.dependOn(&run_database_tests.step);
    test_clean_step.dependOn(&run_http_tests.step);
    test_clean_step.dependOn(&run_plugin_tests.step);
    test_clean_step.dependOn(&run_wdbx_tests.step);
    test_clean_step.dependOn(&run_tcp_echo_tests.step);
    test_clean_step.dependOn(&run_rate_limit_tests.step);
    test_clean_step.dependOn(&run_performance_tests.step);
    test_clean_step.dependOn(clean_step);

    // Run all benchmark suites (non-WASM targets only)
    if (!is_wasm) {
        const bench_all_step = b.step("bench-all", "Run all benchmark suites");
        // Individual benchmark steps are added conditionally above
        // This step can be extended to depend on specific benchmark steps
        // Add benchmark steps as dependencies here when available

        // Benchmark with auto-cleanup
        const bench_clean_step = b.step("bench-clean", "Run benchmarks then auto-clean artifacts");
        bench_clean_step.dependOn(bench_all_step);
        bench_clean_step.dependOn(clean_step);
    }

    // Note: Demo and analysis cleanup steps are defined within their respective conditional blocks
    // to avoid scope issues with run_demo and run_analysis variables

    // Development workflow steps with cleanup
    const dev_clean_step = b.step("dev-clean", "Development build with auto-cleanup");
    dev_clean_step.dependOn(&cli_exe.step);
    dev_clean_step.dependOn(clean_step);

    // Full CI/CD pipeline simulation with cleanup
    const ci_step = b.step("ci", "Full CI pipeline: build, test, then cleanup");
    ci_step.dependOn(&cli_exe.step);
    ci_step.dependOn(&run_unit_tests.step);
    ci_step.dependOn(clean_step);

    // Individual test steps with auto-cleanup for convenience
    const test_simd_clean_step = b.step("test-simd-clean", "Run SIMD tests then auto-clean");
    test_simd_clean_step.dependOn(&run_simd_tests.step);
    test_simd_clean_step.dependOn(clean_step);

    const test_database_clean_step = b.step("test-database-clean", "Run database tests then auto-clean");
    test_database_clean_step.dependOn(&run_database_tests.step);
    test_database_clean_step.dependOn(clean_step);

    const test_http_clean_step = b.step("test-http-clean", "Run HTTP tests then auto-clean");
    test_http_clean_step.dependOn(&run_http_tests.step);
    test_http_clean_step.dependOn(clean_step);

    // ========================================================================
    // DEFAULT BUILD TARGET
    // ========================================================================

    // Set the CLI executable as the default build target
    b.default_step.dependOn(&cli_exe.step);
=======
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
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b
}
