//! Ultra-high-performance GPU renderer using WebGPU with desktop/WASM compatibility
//!
//! This module provides GPU-accelerated rendering and compute capabilities
//! for the Abi AI framework, including:
//! - Cross-platform WebGPU support (Desktop + WASM)
//! - High-performance compute shaders for AI operations
//! - Memory-efficient buffer management
//! - Real-time neural network visualization
//! - SIMD-accelerated CPU fallbacks
//! - Complete GPU compute examples
//! - Advanced Zig optimizations with comptime and inline functions

const std = @import("std");
const builtin = @import("builtin");
const gpu = std.gpu;
const math = std.math;
const print = std.debug.print;
const DynLib = std.DynLib;

// Compile-time constants for performance optimization
pub const has_webgpu_support = @hasDecl(std, "gpu") and @hasDecl(std.gpu, "Instance");

// Compile-time configuration constants
pub const DEFAULT_VECTOR_SIZE = 1024;
pub const DEFAULT_MATRIX_SIZE = 64;
pub const DEFAULT_IMAGE_SIZE = 128;
pub const WORKGROUP_SIZE_1D = 64;
pub const WORKGROUP_SIZE_2D = 16;
pub const MAX_VERIFICATION_SAMPLES = 10;
pub const EPSILON = 0.001;

// Compile-time math constants
const PI = math.pi;
const TAU = 2.0 * PI;
const SQRT2 = @sqrt(2.0);

/// GPU renderer errors
pub const GpuError = error{
    UnsupportedBackend,
    InitializationFailed,
    DeviceNotFound,
    OutOfMemory,
    NotImplemented,
    ValidationFailed,
    ShaderCompilationFailed,
    BufferCreationFailed,
    CommandSubmissionFailed,
    GpuInstanceCreationFailed,
    NoSuitableAdapter,
    DeviceCreationFailed,
    CommandEncoderCreationFailed,
    CommandCreationFailed,
    BindGroupCreationFailed,
    PipelineCreationFailed,
    BufferMappingFailed,
    HandleNotFound,
    TextureCreationFailed,
    TextureViewCreationFailed,
    CompilerInitializationFailed,
    SPIRVCompilationFailed,
    MSLCompilationFailed,
    PTXCompilationFailed,
};

/// SPIR-V compiler optimization levels
pub const SPIRVOptimizationLevel = enum {
    none,
    size,
    performance,

    pub inline fn toInt(self: SPIRVOptimizationLevel) u32 {
        return switch (self) {
            .none => 0,
            .size => 1,
            .performance => 2,
        };
    }
};

/// SPIR-V compiler configuration structure
pub const SPIRVCompilerOptions = struct {
    /// Target backend for SPIR-V compilation
    backend: Backend = .vulkan,
    /// Use LLVM SPIR-V backend instead of self-hosted
    use_llvm_backend: bool = false,
    /// Optimization level for SPIR-V compilation
    optimization_level: SPIRVOptimizationLevel = .performance,
    /// Include debug information in compiled SPIR-V
    debug_info: bool = false,
    /// Generate debug symbols
    generate_debug_info: bool = false,
    /// Source language hint for SPIR-V
    source_language: u32 = 0, // 0 = Unknown, 1 = ESSL, 2 = GLSL, 3 = OpenCL_C, 4 = OpenCL_CPP, 5 = HLSL
    /// Target SPIR-V version
    target_spirv_version: u32 = 0x10300, // SPIR-V 1.3
    /// Maximum number of registers per thread
    max_registers: u32 = 255,
    /// Enable Vulkan memory model
    vulkan_memory_model: bool = true,
    /// Enable variable pointers
    variable_pointers: bool = false,

    pub fn validate(self: SPIRVCompilerOptions) !void {
        if (self.target_spirv_version < 0x10000) {
            return GpuError.ValidationFailed;
        }
        if (self.max_registers == 0) {
            return GpuError.ValidationFailed;
        }
    }
};

/// Metal Shading Language optimization levels
pub const MSLOptimizationLevel = enum {
    none,
    size,
    performance,
    aggressive,

    pub inline fn toInt(self: MSLOptimizationLevel) u32 {
        return switch (self) {
            .none => 0,
            .size => 1,
            .performance => 2,
            .aggressive => 3,
        };
    }
};

/// Metal target versions
pub const MetalVersion = enum {
    v1_0,
    v1_1,
    v1_2,
    v2_0,
    v2_1,
    v2_2,
    v2_3,
    v2_4,
    v3_0,
    v3_1,

    pub inline fn toVersionString(self: MetalVersion) []const u8 {
        return switch (self) {
            .v1_0 => "1.0",
            .v1_1 => "1.1",
            .v1_2 => "1.2",
            .v2_0 => "2.0",
            .v2_1 => "2.1",
            .v2_2 => "2.2",
            .v2_3 => "2.3",
            .v2_4 => "2.4",
            .v3_0 => "3.0",
            .v3_1 => "3.1",
        };
    }
};

/// Metal Shading Language compiler configuration structure
pub const MSLCompilerOptions = struct {
    /// Target Metal version
    target_version: MetalVersion = .v2_4,
    /// Optimization level for MSL compilation
    optimization_level: MSLOptimizationLevel = .performance,
    /// Include debug information in compiled MSL
    debug_info: bool = false,
    /// Metal platform target
    platform: Platform = .auto,
    /// MSL version compatibility
    msl_version: u32 = 20400, // 2.4.0
    /// Vertex buffer index
    vertex_buffer_index: u32 = 30,
    /// Fragment buffer index
    fragment_buffer_index: u32 = 30,
    /// Compute buffer index
    compute_buffer_index: u32 = 30,
    /// Texture buffer index
    texture_buffer_index: u32 = 30,
    /// Enable fast math optimizations
    fast_math: bool = true,
    /// Enable Metal Performance Shaders integration
    enable_mps: bool = true,
    /// Maximum threads per threadgroup
    max_threads_per_threadgroup: u32 = 1024,

    pub const Platform = enum {
        auto,
        macos,
        ios,
        tvos,
        watchos,

        pub inline fn isApplePlatform(self: Platform) bool {
            return switch (self) {
                .auto => true,
                .macos, .ios, .tvos, .watchos => true,
            };
        }
    };

    pub fn validate(self: MSLCompilerOptions) !void {
        if (self.msl_version < 10000) {
            return GpuError.ValidationFailed;
        }
        if (self.max_threads_per_threadgroup == 0 or self.max_threads_per_threadgroup > 1024) {
            return GpuError.ValidationFailed;
        }
    }
};

/// PTX optimization levels
pub const PTXOptimizationLevel = enum {
    none,
    O1,
    O2,
    O3,
    Ofast,

    pub inline fn toString(self: PTXOptimizationLevel) []const u8 {
        return switch (self) {
            .none => "0",
            .O1 => "1",
            .O2 => "2",
            .O3 => "3",
            .Ofast => "fast",
        };
    }
};

/// CUDA compute capabilities
pub const CudaComputeCapability = enum {
    v3_0,
    v3_5,
    v5_0,
    v5_2,
    v6_0,
    v6_1,
    v7_0,
    v7_5,
    v8_0,
    v8_6,
    v8_9,
    v9_0,

    pub inline fn toString(self: CudaComputeCapability) []const u8 {
        return switch (self) {
            .v3_0 => "sm_30",
            .v3_5 => "sm_35",
            .v5_0 => "sm_50",
            .v5_2 => "sm_52",
            .v6_0 => "sm_60",
            .v6_1 => "sm_61",
            .v7_0 => "sm_70",
            .v7_5 => "sm_75",
            .v8_0 => "sm_80",
            .v8_6 => "sm_86",
            .v8_9 => "sm_89",
            .v9_0 => "sm_90",
        };
    }

    pub inline fn getMajorVersion(self: CudaComputeCapability) u32 {
        return switch (self) {
            .v3_0, .v3_5 => 3,
            .v5_0, .v5_2 => 5,
            .v6_0, .v6_1 => 6,
            .v7_0, .v7_5 => 7,
            .v8_0, .v8_6, .v8_9 => 8,
            .v9_0 => 9,
        };
    }
};

/// PTX (Parallel Thread Execution) compiler configuration structure
pub const PTXCompilerOptions = struct {
    /// CUDA compute capability target
    compute_capability: CudaComputeCapability = .v7_5,
    /// Optimization level for PTX compilation
    optimization_level: PTXOptimizationLevel = .O3,
    /// Include debug information in compiled PTX
    debug_info: bool = false,
    /// GPU architecture name
    gpu_name: []const u8 = "sm_75",
    /// Maximum number of registers per thread
    maxrregcount: u32 = 255,
    /// Enable fast math operations
    use_fast_math: bool = true,
    /// Flush denormal values to zero
    ftz: bool = true,
    /// Precision of division and square root operations
    prec_div: bool = true,
    /// Precision of square root operations
    prec_sqrt: bool = true,
    /// Enable fused multiply-add operations
    fmad: bool = true,
    /// Maximum number of threads per block
    max_threads_per_block: u32 = 1024,
    /// Shared memory size per block in bytes
    shared_memory_size: u32 = 49152, // 48KB default
    /// Enable CUDA dynamic parallelism
    dynamic_parallelism: bool = false,
    /// PTX ISA version
    ptx_version: u32 = 70, // PTX ISA 7.0

    pub fn validate(self: PTXCompilerOptions) !void {
        if (self.maxrregcount == 0 or self.maxrregcount > 255) {
            return GpuError.ValidationFailed;
        }
        if (self.max_threads_per_block == 0 or self.max_threads_per_block > 1024) {
            return GpuError.ValidationFailed;
        }
        if (self.shared_memory_size > 98304) { // 96KB max
            return GpuError.ValidationFailed;
        }
    }

    pub inline fn getComputeCapabilityString(self: PTXCompilerOptions) []const u8 {
        return self.compute_capability.toString();
    }
};

/// SPIR-V compiler for Vulkan, OpenGL, and OpenCL backends
pub const SPIRVCompiler = struct {
    allocator: std.mem.Allocator,
    options: SPIRVCompilerOptions,

    pub fn init(allocator: std.mem.Allocator, options: SPIRVCompilerOptions) !*SPIRVCompiler {
        try options.validate();

        const compiler = try allocator.create(SPIRVCompiler);
        compiler.* = .{
            .allocator = allocator,
            .options = options,
        };

        return compiler;
    }

    pub fn deinit(self: *SPIRVCompiler) void {
        self.allocator.destroy(self);
    }

    pub fn compileShader(self: *SPIRVCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
        // Mock SPIR-V compilation - in real implementation, this would:
        // 1. Parse the shader source (WGSL, GLSL, or HLSL)
        // 2. Generate SPIR-V bytecode using Zig's self-hosted SPIR-V backend
        // 3. Apply optimizations based on options
        // 4. Return compiled SPIR-V binary

        _ = stage;
        const spirv_header = [_]u8{
            0x03, 0x02, 0x23, 0x07, // SPIR-V magic number
            0x00, 0x01, 0x03, 0x00, // Version 1.3
        };

        const compiled = try self.allocator.alloc(u8, spirv_header.len + source.len);
        @memcpy(compiled[0..spirv_header.len], &spirv_header);
        @memcpy(compiled[spirv_header.len..], source);

        return compiled;
    }

    pub fn validateSPIRV(_self: *SPIRVCompiler, spirv_code: []const u8) !bool {
        // Mock SPIR-V validation - in real implementation, this would:
        // 1. Parse the SPIR-V binary format
        // 2. Validate instruction encoding and operands
        // 3. Check for structural correctness
        // 4. Verify capability requirements
        // 5. Return true if valid, false if invalid

        _ = _self;

        // Simple validation: check SPIR-V magic number
        if (spirv_code.len < 8) return false;
        const magic = std.mem.readInt(u32, spirv_code[0..4], .little);
        return magic == 0x07230203; // SPIR-V magic number
    }

    pub fn disassembleSPIRV(self: *SPIRVCompiler, spirv_code: []const u8) ![]u8 {
        // Mock SPIR-V disassembly - in real implementation, this would:
        // 1. Parse the SPIR-V binary format
        // 2. Generate human-readable disassembly
        // 3. Include instruction mnemonics and operands
        // 4. Format the output for debugging

        _ = spirv_code;

        const disassembly = try self.allocator.alloc(u8, 256);
        @memcpy(disassembly[0.."OpCapability Shader\n".len], "OpCapability Shader\n");
        @memcpy(disassembly["OpCapability Shader\n".len .. "OpMemoryModel Logical GLSL450\n".len + "OpCapability Shader\n".len], "OpMemoryModel Logical GLSL450\n");
        @memcpy(disassembly["OpCapability Shader\nOpMemoryModel Logical GLSL450\n".len..], "OpEntryPoint GLCompute %main \"main\"\n");
        return disassembly;
    }
};

/// Metal Shading Language compiler for Apple platforms
pub const MSLCompiler = struct {
    allocator: std.mem.Allocator,
    options: MSLCompilerOptions,

    pub fn init(allocator: std.mem.Allocator, options: MSLCompilerOptions) !*MSLCompiler {
        try options.validate();

        const compiler = try allocator.create(MSLCompiler);
        compiler.* = .{
            .allocator = allocator,
            .options = options,
        };

        return compiler;
    }

    pub fn deinit(self: *MSLCompiler) void {
        self.allocator.destroy(self);
    }

    pub fn compileShader(self: *MSLCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
        // Mock MSL compilation - in real implementation, this would:
        // 1. Parse the shader source (WGSL or MSL)
        // 2. Generate Metal Shading Language code
        // 3. Apply platform-specific optimizations
        // 4. Return compiled MSL source or Metal library

        _ = stage;
        const msl_prefix = "#include <metal_stdlib>\nusing namespace metal;\n\n";
        const compiled = try self.allocator.alloc(u8, msl_prefix.len + source.len);
        @memcpy(compiled[0..msl_prefix.len], msl_prefix);
        @memcpy(compiled[msl_prefix.len..], source);

        return compiled;
    }
};

/// PTX compiler for NVIDIA CUDA platforms
pub const PTXCompiler = struct {
    allocator: std.mem.Allocator,
    options: PTXCompilerOptions,

    pub fn init(allocator: std.mem.Allocator, options: PTXCompilerOptions) !*PTXCompiler {
        try options.validate();

        const compiler = try allocator.create(PTXCompiler);
        compiler.* = .{
            .allocator = allocator,
            .options = options,
        };

        return compiler;
    }

    pub fn deinit(self: *PTXCompiler) void {
        self.allocator.destroy(self);
    }

    pub fn compileKernel(self: *PTXCompiler, source: []const u8) ![]u8 {
        // Mock PTX compilation - in real implementation, this would:
        // 1. Parse the compute kernel source
        // 2. Generate PTX assembly code
        // 3. Apply CUDA-specific optimizations
        // 4. Return compiled PTX binary or cubin

        const ptx_header = std.fmt.allocPrint(self.allocator, ".version {d}.{d}\n.target {s}\n.address_size 64\n\n", .{ self.options.ptx_version / 10, self.options.ptx_version % 10, self.options.getComputeCapabilityString() }) catch return GpuError.PTXCompilationFailed;
        defer self.allocator.free(ptx_header);

        const compiled = try self.allocator.alloc(u8, ptx_header.len + source.len);
        @memcpy(compiled[0..ptx_header.len], ptx_header);
        @memcpy(compiled[ptx_header.len..], source);

        return compiled;
    }
};

/// GPU renderer configuration with compile-time optimization
pub const GPUConfig = struct {
    /// Enable validation layers for debugging
    debug_validation: bool = false,
    /// Preferred power profile
    power_preference: PowerPreference = .high_performance,
    /// Maximum number of frames in flight
    max_frames_in_flight: u32 = 2,
    /// Enable VSync
    vsync: bool = true,
    /// Target framerate (0 = unlimited)
    target_fps: u32 = 0,
    /// Backend preference
    backend: Backend = .auto,
    /// When auto, try WebGPU first before other native backends
    try_webgpu_first: bool = true,
    /// Canvas width (WASM only)
    canvas_width: u32 = 800,
    /// Canvas height (WASM only)
    canvas_height: u32 = 600,
    /// Use LLVM SPIR-V backend instead of self-hosted
    use_llvm_spirv_backend: bool = false,
    /// SPIR-V optimization level
    spirv_optimization_level: SPIRVOptimizationLevel = .performance,
    /// Include debug information in shaders
    include_debug_info: bool = false,
    /// Metal target version
    metal_target_version: MetalVersion = .v2_4,
    /// MSL optimization level
    msl_optimization_level: MSLOptimizationLevel = .performance,
    /// CUDA compute capability
    cuda_compute_capability: CudaComputeCapability = .v7_5,
    /// PTX optimization level
    ptx_optimization_level: PTXOptimizationLevel = .O3,

    /// Compile-time validation of configuration
    pub fn validate(comptime config: GPUConfig) void {
        if (comptime config.max_frames_in_flight == 0) {
            @compileError("max_frames_in_flight must be greater than 0");
        }
        if (comptime config.canvas_width == 0 or config.canvas_height == 0) {
            @compileError("Canvas dimensions must be greater than 0");
        }
    }
};

/// Power preference for GPU selection
pub const PowerPreference = enum {
    low_power,
    high_performance,

    /// Inline function for quick preference checks
    pub inline fn isHighPerformance(self: PowerPreference) bool {
        return self == .high_performance;
    }
};

/// GPU backend types with platform detection and compile-time optimization
pub const Backend = enum {
    auto,
    vulkan,
    metal,
    dx12,
    opengl,
    opencl,
    cuda,
    webgpu,
    cpu_fallback,

    pub inline fn isAvailable(self: Backend) bool {
        return switch (self) {
            .auto => true,
            .vulkan => comptime (builtin.os.tag == .linux or builtin.os.tag == .windows),
            .metal => comptime (builtin.os.tag == .macos or builtin.os.tag == .ios),
            .dx12 => comptime (builtin.os.tag == .windows),
            .opengl => true,
            .opencl => true,
            .cuda => comptime (builtin.os.tag == .windows or builtin.os.tag == .linux),
            .webgpu => has_webgpu_support,
            .cpu_fallback => true,
        };
    }

    pub fn getBest() Backend {
        if (!has_webgpu_support) return .cpu_fallback;

        return switch (builtin.os.tag) {
            .windows => if (has_webgpu_support) .webgpu else .cpu_fallback,
            .macos, .ios => if (has_webgpu_support) .webgpu else .cpu_fallback,
            .linux => if (has_webgpu_support) .webgpu else .cpu_fallback,
            else => .cpu_fallback,
        };
    }

    /// Inline function for performance checks
    pub inline fn requiresGPU(self: Backend) bool {
        return switch (self) {
            .webgpu, .vulkan, .metal, .dx12, .opengl, .opencl, .cuda => true,
            .auto, .cpu_fallback => false,
        };
    }

    /// Get priority score for backend selection (higher is better)
    pub fn getPriority(self: Backend) u8 {
        return switch (self) {
            .auto => 0,
            .cpu_fallback => 1,
            .webgpu => 10,
            .opengl => 20,
            .opencl => 30,
            .vulkan => 50,
            .metal => 60,
            .dx12 => 70,
            .cuda => 100,
        };
    }

    /// Convert backend to human-readable string
    pub fn toString(self: Backend) []const u8 {
        return switch (self) {
            .auto => "Auto",
            .vulkan => "Vulkan",
            .metal => "Metal",
            .dx12 => "DirectX 12",
            .opengl => "OpenGL",
            .opencl => "OpenCL",
            .cuda => "CUDA",
            .webgpu => "WebGPU",
            .cpu_fallback => "CPU Fallback",
        };
    }
};

/// GPU buffer usage flags
pub const BufferUsage = packed struct {
    vertex: bool = false,
    index: bool = false,
    uniform: bool = false,
    storage: bool = false,
    copy_src: bool = false,
    copy_dst: bool = false,
    map_read: bool = false,
    map_write: bool = false,

    /// Inline function for quick usage checks
    pub inline fn isReadable(self: BufferUsage) bool {
        return self.map_read or self.copy_src;
    }

    /// Inline function for quick usage checks
    pub inline fn isWritable(self: BufferUsage) bool {
        return self.map_write or self.copy_dst;
    }
};

/// GPU texture format
pub const TextureFormat = enum {
    rgba8_unorm,
    bgra8_unorm,
    r32_float,
    rg32_float,
    rgba32_float,
    depth24_plus,
    depth32_float,

    /// Inline function for format properties
    pub inline fn getBytesPerPixel(self: TextureFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .bgra8_unorm => 4,
            .r32_float => 4,
            .rg32_float => 8,
            .rgba32_float => 16,
            .depth24_plus => 4,
            .depth32_float => 4,
        };
    }

    /// Inline function for format checks
    pub inline fn isFloatFormat(self: TextureFormat) bool {
        return switch (self) {
            .r32_float, .rg32_float, .rgba32_float, .depth32_float => true,
            else => false,
        };
    }
};

/// Shader stage types
pub const ShaderStage = enum {
    vertex,
    fragment,
    compute,

    pub inline fn toWebGPU(self: ShaderStage) u32 {
        return switch (self) {
            .vertex => 0x1,
            .fragment => 0x2,
            .compute => 0x4,
        };
    }
};

/// Color for clearing operations with inline utility functions
pub const Color = struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,

    /// Compile-time color constants
    pub const BLACK = Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };
    pub const WHITE = Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };
    pub const RED = Color{ .r = 1.0, .g = 0.0, .b = 0.0, .a = 1.0 };
    pub const GREEN = Color{ .r = 0.0, .g = 1.0, .b = 0.0, .a = 1.0 };
    pub const BLUE = Color{ .r = 0.0, .g = 0.0, .b = 1.0, .a = 1.0 };

    /// Inline utility functions
    pub inline fn fromRGB(r: f32, g: f32, b: f32) Color {
        return .{ .r = r, .g = g, .b = b, .a = 1.0 };
    }

    pub inline fn lerp(a: Color, b: Color, t: f32) Color {
        return .{
            .r = a.r + (b.r - a.r) * t,
            .g = a.g + (b.g - a.g) * t,
            .b = a.b + (b.b - a.b) * t,
            .a = a.a + (b.a - a.a) * t,
        };
    }

    /// Inline function to convert to packed format
    pub inline fn toPackedRGBA(self: Color) u32 {
        const r = @as(u32, @intFromFloat(self.r * 255.0));
        const g = @as(u32, @intFromFloat(self.g * 255.0));
        const b = @as(u32, @intFromFloat(self.b * 255.0));
        const a = @as(u32, @intFromFloat(self.a * 255.0));
        return (a << 24) | (b << 16) | (g << 8) | r;
    }
};

/// GPU resource handle with generation for safety and inline utilities
pub const GPUHandle = struct {
    id: u64,
    generation: u32,

    pub inline fn invalid() GPUHandle {
        return .{ .id = 0, .generation = 0 };
    }

    pub inline fn isValid(self: GPUHandle) bool {
        return self.id != 0;
    }

    /// Inline function for handle comparison
    pub inline fn equals(self: GPUHandle, other: GPUHandle) bool {
        return self.id == other.id and self.generation == other.generation;
    }
};

/// High-performance math utilities with SIMD operations
pub const MathUtils = struct {
    /// Inline vector operations
    pub inline fn vectorAdd(comptime T: type, a: []const T, b: []const T, result: []T) void {
        std.debug.assert(a.len == b.len and b.len == result.len);

        // Manual loop unrolling for small sizes
        const len = a.len;
        var i: usize = 0;

        // Process 4 elements at a time for better cache utilization
        while (i + 4 <= len) : (i += 4) {
            result[i] = a[i] + b[i];
            result[i + 1] = a[i + 1] + b[i + 1];
            result[i + 2] = a[i + 2] + b[i + 2];
            result[i + 3] = a[i + 3] + b[i + 3];
        }

        // Handle remaining elements
        while (i < len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Inline matrix multiplication with cache-friendly access patterns
    pub inline fn matrixMultiply(comptime T: type, a: []const T, b: []const T, result: []T, size: usize) void {
        std.debug.assert(a.len == size * size);
        std.debug.assert(b.len == size * size);
        std.debug.assert(result.len == size * size);

        // Cache-friendly blocked matrix multiplication
        const block_size = 8; // Optimize for cache lines

        var i: usize = 0;
        while (i < size) : (i += block_size) {
            var j: usize = 0;
            while (j < size) : (j += block_size) {
                var k: usize = 0;
                while (k < size) : (k += block_size) {
                    // Process block
                    const i_end = @min(i + block_size, size);
                    const j_end = @min(j + block_size, size);
                    const k_end = @min(k + block_size, size);

                    var ii = i;
                    while (ii < i_end) : (ii += 1) {
                        var jj = j;
                        while (jj < j_end) : (jj += 1) {
                            var sum: T = 0;
                            var kk = k;
                            while (kk < k_end) : (kk += 1) {
                                sum += a[ii * size + kk] * b[kk * size + jj];
                            }
                            if (k == 0) {
                                result[ii * size + jj] = sum;
                            } else {
                                result[ii * size + jj] += sum;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Inline approximation functions for faster math
    pub inline fn fastSqrt(x: f32) f32 {
        // Fast inverse square root approximation (Quake III style)
        if (x <= 0.0) return 0.0;
        const bits: u32 = @bitCast(x);
        const magic = 0x5f3759df - (bits >> 1);
        const y: f32 = @bitCast(magic);
        return x * y * (1.5 - 0.5 * x * y * y);
    }

    /// Inline function for fast approximate equality
    pub inline fn approxEqual(a: f32, b: f32) bool {
        return @abs(a - b) < EPSILON;
    }
};

test "buffer usage helpers report readability and writability" {
    const usage = BufferUsage{
        .copy_src = true,
        .map_write = true,
    };
    try std.testing.expect(usage.isReadable());
    try std.testing.expect(usage.isWritable());
}

test "texture format metadata is exposed" {
    try std.testing.expectEqual(@as(u32, 4), TextureFormat.rgba8_unorm.getBytesPerPixel());
    try std.testing.expectEqualStrings("DirectX 12", Backend.dx12.toString());
}

test "color helpers support lerp and packing" {
    const a = Color.fromRGB(1.0, 0.0, 0.0);
    const b = Color.fromRGB(0.0, 0.0, 1.0);
    const mid = Color.lerp(a, b, 0.5);
    try std.testing.expect(MathUtils.approxEqual(mid.r, 0.5));
    try std.testing.expect(MathUtils.approxEqual(mid.b, 0.5));
    try std.testing.expect(mid.toPackedRGBA() != 0);
}

test "math utils implement vector and matrix operations" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var lhs = [_]f32{ 1, 2, 3, 4 };
    var rhs = [_]f32{ 5, 6, 7, 8 };
    var result = [_]f32{ 0, 0, 0, 0 };
    MathUtils.vectorAdd(f32, &lhs, &rhs, &result);
    try testing.expectEqualSlices(f32, &[_]f32{ 6, 8, 10, 12 }, &result);

    const mat_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const mat_b = [_]f32{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const mat_result = try allocator.alloc(f32, mat_a.len);
    defer allocator.free(mat_result);
    MathUtils.matrixMultiply(f32, &mat_a, &mat_b, mat_result, 3);

    try testing.expect(MathUtils.approxEqual(mat_result[0], 30.0));
    try testing.expect(MathUtils.approxEqual(mat_result[4], 69.0));
}
