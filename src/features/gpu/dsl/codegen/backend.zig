//! Code Generator Backend Interface
//!
//! Defines the common interface for all code generators that translate
//! kernel IR to backend-specific source code.

const std = @import("std");
const kernel = @import("../kernel.zig");
const gpu_backend = @import("../../backend.zig");

/// Errors that can occur during code generation.
pub const CodegenError = error{
    /// Type not supported by target backend.
    UnsupportedType,
    /// Operation not supported by target backend.
    UnsupportedOperation,
    /// Feature not supported by target backend.
    UnsupportedFeature,
    /// Out of memory.
    OutOfMemory,
    /// Invalid IR structure.
    InvalidIR,
    /// Backend not available.
    BackendUnavailable,
    /// Compilation failed.
    CompilationFailed,
};

/// Generated source code with metadata.
pub const GeneratedSource = struct {
    /// The generated source code.
    code: []const u8,
    /// Entry point function name in the generated code.
    entry_point: []const u8,
    /// Target backend.
    backend: gpu_backend.Backend,
    /// Source language (for debugging/logging).
    language: Language,
    /// Optional SPIR-V binary (for Vulkan/OpenGL).
    spirv_binary: ?[]const u8 = null,

    pub const Language = enum {
        cuda,
        glsl,
        wgsl,
        msl,
        spirv,
        hlsl,
    };

    /// Free the generated source.
    pub fn deinit(self: *GeneratedSource, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.entry_point);
        if (self.spirv_binary) |spirv| {
            allocator.free(spirv);
        }
    }
};

/// Code generator interface.
/// Each backend implements this to generate code from kernel IR.
pub const CodeGenerator = struct {
    /// Pointer to implementation-specific data.
    impl: *anyopaque,

    /// Function pointer for generating code.
    generateFn: *const fn (
        *anyopaque,
        std.mem.Allocator,
        *const kernel.KernelIR,
    ) CodegenError!GeneratedSource,

    /// Function pointer for checking if a feature is supported.
    supportsFeatureFn: *const fn (*anyopaque, kernel.FeatureFlags) bool,

    /// Function pointer for getting the target backend.
    getBackendFn: *const fn (*anyopaque) gpu_backend.Backend,

    /// Generate source code from kernel IR.
    pub fn generate(
        self: *CodeGenerator,
        allocator: std.mem.Allocator,
        ir: *const kernel.KernelIR,
    ) CodegenError!GeneratedSource {
        return self.generateFn(self.impl, allocator, ir);
    }

    /// Check if the generator supports all required features.
    pub fn supportsFeatures(self: *CodeGenerator, features: kernel.FeatureFlags) bool {
        return self.supportsFeatureFn(self.impl, features);
    }

    /// Get the target backend.
    pub fn getBackend(self: *CodeGenerator) gpu_backend.Backend {
        return self.getBackendFn(self.impl);
    }
};

/// Backend capability information.
pub const BackendCapabilities = struct {
    /// Maximum workgroup size (x * y * z).
    max_workgroup_size: u32 = 1024,
    /// Maximum shared memory in bytes.
    max_shared_memory: u32 = 49152, // 48KB default
    /// Maximum buffers per shader.
    max_buffers: u32 = 16,
    /// Maximum uniforms per shader.
    max_uniforms: u32 = 16,
    /// Supports 16-bit floats.
    supports_fp16: bool = false,
    /// Supports 64-bit floats.
    supports_fp64: bool = false,
    /// Supports 64-bit integers.
    supports_int64: bool = false,
    /// Supports subgroup operations.
    supports_subgroups: bool = false,
    /// Supports dynamic shared memory.
    supports_dynamic_shared_memory: bool = true,
    /// Supports atomics.
    supports_atomics: bool = true,
    /// Supports mesh shaders (Metal 3+ / Vulkan mesh shading).
    supports_mesh_shaders: bool = false,
    /// Supports ray tracing (Metal 3+ / Vulkan RT).
    supports_ray_tracing: bool = false,

    /// Get default capabilities for a backend.
    pub fn forBackend(backend: gpu_backend.Backend) BackendCapabilities {
        return switch (backend) {
            .cuda => .{
                .max_workgroup_size = 1024,
                .max_shared_memory = 49152,
                .supports_fp16 = true,
                .supports_fp64 = true,
                .supports_int64 = true,
                .supports_subgroups = true, // warps
                .supports_dynamic_shared_memory = true,
            },
            .vulkan => .{
                .max_workgroup_size = 1024,
                .max_shared_memory = 32768,
                .supports_fp16 = true,
                .supports_fp64 = false, // extension
                .supports_subgroups = true,
            },
            .metal => .{
                .max_workgroup_size = 1024,
                .max_shared_memory = 32768,
                .supports_fp16 = true,
                .supports_fp64 = false,
                .supports_subgroups = true, // simdgroups
                .supports_mesh_shaders = true, // Metal 3+ (Apple7+)
                .supports_ray_tracing = true, // Metal 3+ (Apple7+)
            },
            .webgpu => .{
                .max_workgroup_size = 256,
                .max_shared_memory = 16384,
                .supports_fp16 = false, // extension
                .supports_fp64 = false,
                .supports_subgroups = false,
            },
            .opengl, .opengles => .{
                .max_workgroup_size = 1024,
                .max_shared_memory = 32768,
                .supports_fp16 = false,
                .supports_fp64 = true, // OpenGL
                .supports_subgroups = false,
            },
            .stdgpu => .{
                .max_workgroup_size = 256,
                .max_shared_memory = 16384,
                .supports_fp64 = true,
            },
            .webgl2 => .{
                .max_workgroup_size = 0, // no compute
                .max_shared_memory = 0,
                .supports_atomics = false,
            },
            .fpga => .{
                .max_workgroup_size = 256,
                .max_shared_memory = 64 * 1024, // FPGA PLRAM/BRAM
                .supports_fp16 = true,
                .supports_fp64 = true,
                .supports_int64 = true,
                .supports_subgroups = false,
                .supports_dynamic_shared_memory = false,
            },
            .tpu => .{
                .max_workgroup_size = 1024,
                .max_shared_memory = 128 * 1024,
                .supports_fp16 = true,
                .supports_fp64 = false,
                .supports_int64 = true,
                .supports_subgroups = true,
                .supports_dynamic_shared_memory = true,
            },
            .simulated => .{
                .max_workgroup_size = 256,
                .max_shared_memory = 16 * 1024,
                .supports_fp16 = true,
                .supports_fp64 = true,
                .supports_int64 = true,
            },
        };
    }
};

/// Validate that kernel IR is compatible with backend capabilities.
pub fn validateForBackend(
    ir: *const kernel.KernelIR,
    capabilities: BackendCapabilities,
) ValidationResult {
    var result = ValidationResult{};

    // Check workgroup size
    const total_size = ir.totalWorkgroupSize();
    if (total_size > capabilities.max_workgroup_size) {
        result.workgroup_size_exceeded = true;
    }

    // Check shared memory
    if (ir.sharedMemoryBytes()) |size| {
        if (size > capabilities.max_shared_memory) {
            result.shared_memory_exceeded = true;
        }
    }

    // Check buffer count
    if (ir.buffers.len > capabilities.max_buffers) {
        result.buffer_count_exceeded = true;
    }

    // Check uniform count
    if (ir.uniforms.len > capabilities.max_uniforms) {
        result.uniform_count_exceeded = true;
    }

    // Check features
    if (ir.required_features.fp16 and !capabilities.supports_fp16) {
        result.fp16_unsupported = true;
    }
    if (ir.required_features.fp64 and !capabilities.supports_fp64) {
        result.fp64_unsupported = true;
    }
    if (ir.required_features.int64 and !capabilities.supports_int64) {
        result.int64_unsupported = true;
    }
    if (ir.required_features.subgroups and !capabilities.supports_subgroups) {
        result.subgroups_unsupported = true;
    }
    if (ir.required_features.dynamic_shared_memory and !capabilities.supports_dynamic_shared_memory) {
        result.dynamic_shared_memory_unsupported = true;
    }
    if (ir.required_features.atomics and !capabilities.supports_atomics) {
        result.atomics_unsupported = true;
    }

    return result;
}

/// Result of backend validation.
pub const ValidationResult = struct {
    workgroup_size_exceeded: bool = false,
    shared_memory_exceeded: bool = false,
    buffer_count_exceeded: bool = false,
    uniform_count_exceeded: bool = false,
    fp16_unsupported: bool = false,
    fp64_unsupported: bool = false,
    int64_unsupported: bool = false,
    subgroups_unsupported: bool = false,
    dynamic_shared_memory_unsupported: bool = false,
    atomics_unsupported: bool = false,

    pub fn isValid(self: ValidationResult) bool {
        return !self.workgroup_size_exceeded and
            !self.shared_memory_exceeded and
            !self.buffer_count_exceeded and
            !self.uniform_count_exceeded and
            !self.fp16_unsupported and
            !self.fp64_unsupported and
            !self.int64_unsupported and
            !self.subgroups_unsupported and
            !self.dynamic_shared_memory_unsupported and
            !self.atomics_unsupported;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "BackendCapabilities.forBackend" {
    const cuda_caps = BackendCapabilities.forBackend(.cuda);
    try std.testing.expect(cuda_caps.supports_fp64);
    try std.testing.expect(cuda_caps.supports_subgroups);

    const webgpu_caps = BackendCapabilities.forBackend(.webgpu);
    try std.testing.expect(!webgpu_caps.supports_fp64);
    try std.testing.expect(!webgpu_caps.supports_subgroups);
}

test "validateForBackend" {
    const ir = kernel.KernelIR.empty("test");
    const caps = BackendCapabilities.forBackend(.cuda);
    const result = validateForBackend(&ir, caps);
    try std.testing.expect(result.isValid());
}

test {
    std.testing.refAllDecls(@This());
}
