//! Advanced Cross-Compilation Support for GPU Systems
//!
//! This module provides comprehensive cross-compilation capabilities:
//! - ARM64, RISC-V, and other architecture support
//! - WebAssembly compilation for web deployment
//! - Mobile platform support (iOS/Android) with Metal/Vulkan backends
//! - Cross-platform GPU backend selection
//! - Architecture-specific optimizations
//!
//! Leverages Zig's -target flag for maximum compatibility

const std = @import("std");
const builtin = @import("builtin");

/// Cross-compilation target configuration
pub const CrossCompilationTarget = struct {
    arch: std.Target.Cpu.Arch,
    os: std.Target.Os.Tag,
    abi: std.Target.Abi,
    gpu_backend: GPUBackend,
    optimization_level: OptimizationLevel,
    features: TargetFeatures,
    memory_model: MemoryModel,
    threading_model: ThreadingModel,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CrossCompilationTarget) void {
        self.features.deinit();
    }
};

/// GPU backend selection for cross-compilation
pub const GPUBackend = enum {
    auto,
    webgpu,
    vulkan,
    metal,
    directx12,
    opengl,
    cuda,
    opencl,
    spirv,
    wasm_webgpu,
    mobile_metal,
    mobile_vulkan,
    embedded_opengl,
};

/// Optimization level for cross-compilation
pub const OptimizationLevel = enum {
    debug,
    release_safe,
    release_fast,
    release_small,
    size_optimized,
    performance_optimized,
};

/// Target-specific features
pub const TargetFeatures = struct {
    supports_simd: bool,
    supports_neon: bool,
    supports_avx: bool,
    supports_sse: bool,
    supports_wasm_simd: bool,
    supports_riscv_vector: bool,
    supports_arm_sve: bool,
    supports_gpu_compute: bool,
    supports_unified_memory: bool,
    supports_shared_memory: bool,
    supports_atomic_operations: bool,
    supports_threading: bool,
    supports_async_operations: bool,
    max_memory_size: u64,
    max_thread_count: u32,
    cache_line_size: u32,
    page_size: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TargetFeatures) void {
        _ = self;
    }
};

/// Memory model for target architecture
pub const MemoryModel = enum {
    relaxed,
    acquire_release,
    sequential_consistent,
    wasm_linear,
    gpu_shared,
    gpu_private,
    gpu_constant,
};

/// Threading model for target architecture
pub const ThreadingModel = enum {
    single_threaded,
    multi_threaded,
    async_threaded,
    gpu_threaded,
    wasm_threaded,
    embedded_threaded,
};

/// Cross-compilation manager
pub const CrossCompilationManager = struct {
    allocator: std.mem.Allocator,
    target_configs: std.HashMap(TargetKey, CrossCompilationTarget, TargetKey.hash, TargetKey.eql),
    build_configs: std.HashMap(TargetKey, BuildConfig, TargetKey.hash, TargetKey.eql),

    const Self = @This();

    /// Target key for hash map
    const TargetKey = struct {
        arch: std.Target.Cpu.Arch,
        os: std.Target.Os.Tag,
        abi: std.Target.Abi,

        pub fn hash(self: TargetKey) u64 {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(std.mem.asBytes(&self.arch));
            hasher.update(std.mem.asBytes(&self.os));
            hasher.update(std.mem.asBytes(&self.abi));
            return hasher.final();
        }

        pub fn eql(self: TargetKey, other: TargetKey) bool {
            return self.arch == other.arch and self.os == other.os and self.abi == other.abi;
        }
    };

    /// Build configuration for target
    const BuildConfig = struct {
        target_string: []const u8,
        optimization_flags: []const []const u8,
        gpu_backend_flags: []const []const u8,
        architecture_flags: []const []const u8,
        memory_flags: []const []const u8,
        threading_flags: []const []const u8,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *BuildConfig) void {
            self.allocator.free(self.target_string);
            for (self.optimization_flags) |flag| {
                self.allocator.free(flag);
            }
            self.allocator.free(self.optimization_flags);
            for (self.gpu_backend_flags) |flag| {
                self.allocator.free(flag);
            }
            self.allocator.free(self.gpu_backend_flags);
            for (self.architecture_flags) |flag| {
                self.allocator.free(flag);
            }
            self.allocator.free(self.architecture_flags);
            for (self.memory_flags) |flag| {
                self.allocator.free(flag);
            }
            self.allocator.free(self.memory_flags);
            for (self.threading_flags) |flag| {
                self.allocator.free(flag);
            }
            self.allocator.free(self.threading_flags);
        }
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .target_configs = std.HashMap(TargetKey, CrossCompilationTarget, TargetKey.hash, TargetKey.eql).init(allocator),
            .build_configs = std.HashMap(TargetKey, BuildConfig, TargetKey.hash, TargetKey.eql).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var target_iterator = self.target_configs.iterator();
        while (target_iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.target_configs.deinit();

        var build_iterator = self.build_configs.iterator();
        while (build_iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.build_configs.deinit();
    }

    /// Register a cross-compilation target
    pub fn registerTarget(self: *Self, target: CrossCompilationTarget) !void {
        const key = TargetKey{
            .arch = target.arch,
            .os = target.os,
            .abi = target.abi,
        };

        try self.target_configs.put(key, target);

        // Generate build configuration for this target
        const build_config = try self.generateBuildConfig(target);
        try self.build_configs.put(key, build_config);
    }

    /// Get cross-compilation target for architecture
    pub fn getTarget(self: *Self, arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, abi: std.Target.Abi) ?*CrossCompilationTarget {
        const key = TargetKey{ .arch = arch, .os = os, .abi = abi };
        return self.target_configs.getPtr(key);
    }

    /// Get build configuration for target
    pub fn getBuildConfig(self: *Self, arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, abi: std.Target.Abi) ?*BuildConfig {
        const key = TargetKey{ .arch = arch, .os = os, .abi = abi };
        return self.build_configs.getPtr(key);
    }

    /// Generate build configuration for target
    fn generateBuildConfig(self: *Self, target: CrossCompilationTarget) !BuildConfig {
        const target_string = try self.generateTargetString(target);
        const optimization_flags = try self.generateOptimizationFlags(target);
        const gpu_backend_flags = try self.generateGPUBackendFlags(target);
        const architecture_flags = try self.generateArchitectureFlags(target);
        const memory_flags = try self.generateMemoryFlags(target);
        const threading_flags = try self.generateThreadingFlags(target);

        return BuildConfig{
            .target_string = target_string,
            .optimization_flags = optimization_flags,
            .gpu_backend_flags = gpu_backend_flags,
            .architecture_flags = architecture_flags,
            .memory_flags = memory_flags,
            .threading_flags = threading_flags,
            .allocator = self.allocator,
        };
    }

    /// Generate target string for Zig build
    fn generateTargetString(self: *Self, target: CrossCompilationTarget) ![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{s}-{s}-{s}", .{
            @tagName(target.arch),
            @tagName(target.os),
            @tagName(target.abi),
        });
    }

    /// Generate optimization flags for target
    fn generateOptimizationFlags(self: *Self, target: CrossCompilationTarget) ![]const []const u8 {
        var flags = std.ArrayList([]const u8).init(self.allocator);
        defer flags.deinit();

        switch (target.optimization_level) {
            .debug => {
                try flags.append(try self.allocator.dupe(u8, "-O0"));
                try flags.append(try self.allocator.dupe(u8, "-g"));
            },
            .release_safe => {
                try flags.append(try self.allocator.dupe(u8, "-O2"));
                try flags.append(try self.allocator.dupe(u8, "-DNDEBUG"));
            },
            .release_fast => {
                try flags.append(try self.allocator.dupe(u8, "-O3"));
                try flags.append(try self.allocator.dupe(u8, "-DNDEBUG"));
                try flags.append(try self.allocator.dupe(u8, "-ffast-math"));
            },
            .release_small => {
                try flags.append(try self.allocator.dupe(u8, "-Os"));
                try flags.append(try self.allocator.dupe(u8, "-DNDEBUG"));
                try flags.append(try self.allocator.dupe(u8, "-fno-unroll-loops"));
            },
            .size_optimized => {
                try flags.append(try self.allocator.dupe(u8, "-Oz"));
                try flags.append(try self.allocator.dupe(u8, "-DNDEBUG"));
            },
            .performance_optimized => {
                try flags.append(try self.allocator.dupe(u8, "-O3"));
                try flags.append(try self.allocator.dupe(u8, "-DNDEBUG"));
                try flags.append(try self.allocator.dupe(u8, "-ffast-math"));
                try flags.append(try self.allocator.dupe(u8, "-funroll-loops"));
            },
        }

        return flags.toOwnedSlice();
    }

    /// Generate GPU backend flags for target
    fn generateGPUBackendFlags(self: *Self, target: CrossCompilationTarget) ![]const []const u8 {
        var flags = std.ArrayList([]const u8).init(self.allocator);
        defer flags.deinit();

        switch (target.gpu_backend) {
            .webgpu => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_WEBGPU"));
            },
            .vulkan => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_VULKAN"));
            },
            .metal => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_METAL"));
            },
            .directx12 => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_DIRECTX12"));
            },
            .opengl => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_OPENGL"));
            },
            .cuda => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_CUDA"));
            },
            .opencl => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_OPENCL"));
            },
            .spirv => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_SPIRV"));
            },
            .wasm_webgpu => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_WASM_WEBGPU"));
            },
            .mobile_metal => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_MOBILE_METAL"));
            },
            .mobile_vulkan => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_MOBILE_VULKAN"));
            },
            .embedded_opengl => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_EMBEDDED_OPENGL"));
            },
            .auto => {
                try flags.append(try self.allocator.dupe(u8, "-DGPU_BACKEND_AUTO"));
            },
        }

        return flags.toOwnedSlice();
    }

    /// Generate architecture-specific flags
    fn generateArchitectureFlags(self: *Self, target: CrossCompilationTarget) ![]const []const u8 {
        var flags = std.ArrayList([]const u8).init(self.allocator);
        defer flags.deinit();

        // Architecture-specific optimizations
        switch (target.arch) {
            .aarch64 => {
                if (target.features.supports_neon) {
                    try flags.append(try self.allocator.dupe(u8, "-march=armv8-a+simd"));
                }
                if (target.features.supports_arm_sve) {
                    try flags.append(try self.allocator.dupe(u8, "-march=armv8-a+sve"));
                }
            },
            .x86_64 => {
                if (target.features.supports_avx) {
                    try flags.append(try self.allocator.dupe(u8, "-mavx2"));
                }
                if (target.features.supports_sse) {
                    try flags.append(try self.allocator.dupe(u8, "-msse4.2"));
                }
            },
            .riscv64 => {
                if (target.features.supports_riscv_vector) {
                    try flags.append(try self.allocator.dupe(u8, "-march=rv64gcv"));
                }
            },
            .wasm32, .wasm64 => {
                if (target.features.supports_wasm_simd) {
                    try flags.append(try self.allocator.dupe(u8, "-msimd128"));
                }
            },
            else => {},
        }

        // SIMD support
        if (target.features.supports_simd) {
            try flags.append(try self.allocator.dupe(u8, "-DSIMD_ENABLED"));
        }

        return flags.toOwnedSlice();
    }

    /// Generate memory model flags
    fn generateMemoryFlags(self: *Self, target: CrossCompilationTarget) ![]const []const u8 {
        var flags = std.ArrayList([]const u8).init(self.allocator);
        defer flags.deinit();

        switch (target.memory_model) {
            .relaxed => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_relaxed"));
            },
            .acquire_release => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_acquire_release"));
            },
            .sequential_consistent => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_sequential_consistent"));
            },
            .wasm_linear => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_wasm_linear"));
            },
            .gpu_shared => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_gpu_shared"));
            },
            .gpu_private => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_gpu_private"));
            },
            .gpu_constant => {
                try flags.append(try self.allocator.dupe(u8, "-Dmemory_model_gpu_constant"));
            },
        }

        // Memory size limits
        if (target.features.max_memory_size > 0) {
            const memory_flag = try std.fmt.allocPrint(self.allocator, "-Dmax_memory_size={}", .{target.features.max_memory_size});
            try flags.append(memory_flag);
        }

        return flags.toOwnedSlice();
    }

    /// Generate threading model flags
    fn generateThreadingFlags(self: *Self, target: CrossCompilationTarget) ![]const []const u8 {
        var flags = std.ArrayList([]const u8).init(self.allocator);
        defer flags.deinit();

        switch (target.threading_model) {
            .single_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_single"));
            },
            .multi_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_multi"));
            },
            .async_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_async"));
            },
            .gpu_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_gpu"));
            },
            .wasm_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_wasm"));
            },
            .embedded_threaded => {
                try flags.append(try self.allocator.dupe(u8, "-Dthreading_model_embedded"));
            },
        }

        // Thread count limits
        if (target.features.max_thread_count > 0) {
            const thread_flag = try std.fmt.allocPrint(self.allocator, "-Dmax_thread_count={}", .{target.features.max_thread_count});
            try flags.append(thread_flag);
        }

        return flags.toOwnedSlice();
    }
};

/// Predefined cross-compilation targets
pub const PredefinedTargets = struct {
    /// WebAssembly target for web deployment
    pub fn wasmTarget(allocator: std.mem.Allocator) !CrossCompilationTarget {
        return CrossCompilationTarget{
            .arch = .wasm32,
            .os = .freestanding,
            .abi = .musl,
            .gpu_backend = .wasm_webgpu,
            .optimization_level = .size_optimized,
            .features = TargetFeatures{
                .supports_simd = true,
                .supports_neon = false,
                .supports_avx = false,
                .supports_sse = false,
                .supports_wasm_simd = true,
                .supports_riscv_vector = false,
                .supports_arm_sve = false,
                .supports_gpu_compute = true,
                .supports_unified_memory = false,
                .supports_shared_memory = false,
                .supports_atomic_operations = true,
                .supports_threading = true,
                .supports_async_operations = true,
                .max_memory_size = 4 * 1024 * 1024 * 1024, // 4GB
                .max_thread_count = 4,
                .cache_line_size = 64,
                .page_size = 65536, // 64KB
                .allocator = allocator,
            },
            .memory_model = .wasm_linear,
            .threading_model = .wasm_threaded,
            .allocator = allocator,
        };
    }

    /// ARM64 target for mobile and embedded systems
    pub fn arm64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
        return CrossCompilationTarget{
            .arch = .aarch64,
            .os = os,
            .abi = .gnu,
            .gpu_backend = if (os == .ios or os == .macos) .mobile_metal else .mobile_vulkan,
            .optimization_level = .performance_optimized,
            .features = TargetFeatures{
                .supports_simd = true,
                .supports_neon = true,
                .supports_avx = false,
                .supports_sse = false,
                .supports_wasm_simd = false,
                .supports_riscv_vector = false,
                .supports_arm_sve = true,
                .supports_gpu_compute = true,
                .supports_unified_memory = true,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_threading = true,
                .supports_async_operations = true,
                .max_memory_size = 16 * 1024 * 1024 * 1024, // 16GB
                .max_thread_count = 8,
                .cache_line_size = 64,
                .page_size = 4096,
                .allocator = allocator,
            },
            .memory_model = .acquire_release,
            .threading_model = .multi_threaded,
            .allocator = allocator,
        };
    }

    /// RISC-V target for embedded and HPC systems
    pub fn riscv64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
        return CrossCompilationTarget{
            .arch = .riscv64,
            .os = os,
            .abi = .gnu,
            .gpu_backend = .vulkan,
            .optimization_level = .performance_optimized,
            .features = TargetFeatures{
                .supports_simd = true,
                .supports_neon = false,
                .supports_avx = false,
                .supports_sse = false,
                .supports_wasm_simd = false,
                .supports_riscv_vector = true,
                .supports_arm_sve = false,
                .supports_gpu_compute = true,
                .supports_unified_memory = false,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_threading = true,
                .supports_async_operations = true,
                .max_memory_size = 32 * 1024 * 1024 * 1024, // 32GB
                .max_thread_count = 16,
                .cache_line_size = 64,
                .page_size = 4096,
                .allocator = allocator,
            },
            .memory_model = .acquire_release,
            .threading_model = .multi_threaded,
            .allocator = allocator,
        };
    }

    /// x86_64 target for desktop systems
    pub fn x86_64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
        return CrossCompilationTarget{
            .arch = .x86_64,
            .os = os,
            .abi = .gnu,
            .gpu_backend = switch (os) {
                .windows => .directx12,
                .macos => .metal,
                else => .vulkan,
            },
            .optimization_level = .performance_optimized,
            .features = TargetFeatures{
                .supports_simd = true,
                .supports_neon = false,
                .supports_avx = true,
                .supports_sse = true,
                .supports_wasm_simd = false,
                .supports_riscv_vector = false,
                .supports_arm_sve = false,
                .supports_gpu_compute = true,
                .supports_unified_memory = false,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_threading = true,
                .supports_async_operations = true,
                .max_memory_size = 64 * 1024 * 1024 * 1024, // 64GB
                .max_thread_count = 32,
                .cache_line_size = 64,
                .page_size = 4096,
                .allocator = allocator,
            },
            .memory_model = .acquire_release,
            .threading_model = .multi_threaded,
            .allocator = allocator,
        };
    }
};

/// Cross-compilation utility functions
pub const CrossCompilationUtils = struct {
    /// Check if target architecture supports GPU acceleration
    pub fn supportsGPUAcceleration(arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag) bool {
        return switch (arch) {
            .x86_64, .aarch64, .riscv64 => true,
            .wasm32, .wasm64 => os == .freestanding, // WebAssembly with WebGPU
            else => false,
        };
    }

    /// Get recommended GPU backend for target
    pub fn getRecommendedGPUBackend(arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag) GPUBackend {
        return switch (os) {
            .windows => .directx12,
            .macos, .ios => .metal,
            .linux, .android => .vulkan,
            .freestanding => if (arch == .wasm32 or arch == .wasm64) .wasm_webgpu else .auto,
            else => .vulkan,
        };
    }

    /// Check if target supports SIMD operations
    pub fn supportsSIMD(arch: std.Target.Cpu.Arch) bool {
        return switch (arch) {
            .x86_64 => true, // AVX, SSE
            .aarch64 => true, // NEON, SVE
            .riscv64 => true, // RISC-V Vector
            .wasm32, .wasm64 => true, // WASM SIMD
            else => false,
        };
    }

    /// Get optimal memory alignment for target
    pub fn getOptimalMemoryAlignment(arch: std.Target.Cpu.Arch) u32 {
        return switch (arch) {
            .x86_64, .aarch64, .riscv64 => 64, // Cache line size
            .wasm32, .wasm64 => 8, // WASM alignment
            else => 16, // Default alignment
        };
    }

    /// Get optimal thread count for target
    pub fn getOptimalThreadCount(arch: std.Target.Cpu.Arch, _: std.Target.Os.Tag) u32 {
        return switch (arch) {
            .x86_64 => 16,
            .aarch64 => 8,
            .riscv64 => 4,
            .wasm32, .wasm64 => 4,
            else => 2,
        };
    }
};

/// Log cross-compilation target information
pub fn logCrossCompilationTarget(target: *const CrossCompilationTarget) void {
    std.log.info("🎯 Cross-Compilation Target:", .{});
    std.log.info("  - Architecture: {s}", .{@tagName(target.arch)});
    std.log.info("  - OS: {s}", .{@tagName(target.os)});
    std.log.info("  - ABI: {s}", .{@tagName(target.abi)});
    std.log.info("  - GPU Backend: {s}", .{@tagName(target.gpu_backend)});
    std.log.info("  - Optimization: {s}", .{@tagName(target.optimization_level)});
    std.log.info("  - Memory Model: {s}", .{@tagName(target.memory_model)});
    std.log.info("  - Threading Model: {s}", .{@tagName(target.threading_model)});
    std.log.info("  - Features:", .{});
    std.log.info("    * SIMD: {}", .{target.features.supports_simd});
    std.log.info("    * GPU Compute: {}", .{target.features.supports_gpu_compute});
    std.log.info("    * Unified Memory: {}", .{target.features.supports_unified_memory});
    std.log.info("    * Threading: {}", .{target.features.supports_threading});
    std.log.info("    * Max Memory: {} GB", .{target.features.max_memory_size / (1024 * 1024 * 1024)});
    std.log.info("    * Max Threads: {}", .{target.features.max_thread_count});
    std.log.info("    * Cache Line: {} bytes", .{target.features.cache_line_size});
    std.log.info("    * Page Size: {} bytes", .{target.features.page_size});
}

test "cross-compilation target creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test WebAssembly target
    const wasm_target = try PredefinedTargets.wasmTarget(allocator);
    defer wasm_target.deinit();
    logCrossCompilationTarget(&wasm_target);

    // Test ARM64 target
    const arm64_target = try PredefinedTargets.arm64Target(allocator, .linux);
    defer arm64_target.deinit();
    logCrossCompilationTarget(&arm64_target);

    // Test RISC-V target
    const riscv_target = try PredefinedTargets.riscv64Target(allocator, .linux);
    defer riscv_target.deinit();
    logCrossCompilationTarget(&riscv_target);

    // Test x86_64 target
    const x86_target = try PredefinedTargets.x86_64Target(allocator, .linux);
    defer x86_target.deinit();
    logCrossCompilationTarget(&x86_target);
}

test "cross-compilation manager" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = CrossCompilationManager.init(allocator);
    defer manager.deinit();

    // Register targets
    const wasm_target = try PredefinedTargets.wasmTarget(allocator);
    defer wasm_target.deinit();
    try manager.registerTarget(wasm_target);

    const arm64_target = try PredefinedTargets.arm64Target(allocator, .linux);
    defer arm64_target.deinit();
    try manager.registerTarget(arm64_target);

    // Test target retrieval
    const retrieved_target = manager.getTarget(.wasm32, .freestanding, .musl);
    try std.testing.expect(retrieved_target != null);

    const retrieved_build_config = manager.getBuildConfig(.wasm32, .freestanding, .musl);
    try std.testing.expect(retrieved_build_config != null);

    if (retrieved_build_config) |config| {
        std.log.info("Build config for WASM: {s}", .{config.target_string});
        std.log.info("Optimization flags: {d}", .{config.optimization_flags.len});
        std.log.info("GPU backend flags: {d}", .{config.gpu_backend_flags.len});
        std.log.info("Architecture flags: {d}", .{config.architecture_flags.len});
        std.log.info("Memory flags: {d}", .{config.memory_flags.len});
        std.log.info("Threading flags: {d}", .{config.threading_flags.len});
    }
}
