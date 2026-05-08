//! Unified Kernel Compiler
//!
//! Compiles kernel IR to backend-specific source code.
//! This is the main entry point for kernel compilation.

const std = @import("std");
const kernel_mod = @import("kernel.zig");
const backend_mod = @import("codegen/backend.zig");
const generic = @import("codegen/generic.zig");
const gpu_backend = @import("../backend.zig");
const kernel_types = @import("../kernel_types.zig");

pub const CompileError = backend_mod.CodegenError || error{
    UnsupportedBackend,
    ValidationFailed,
};

/// Compile options.
pub const CompileOptions = struct {
    /// Validate IR before compilation.
    validate: bool = true,
    /// Target backend capabilities (for validation).
    capabilities: ?backend_mod.BackendCapabilities = null,
    /// Generate debug info in output.
    debug_info: bool = false,
};

/// Compile kernel IR to a specific backend.
pub fn compile(
    allocator: std.mem.Allocator,
    ir: *const kernel_mod.KernelIR,
    target_backend: gpu_backend.Backend,
    options: CompileOptions,
) CompileError!backend_mod.GeneratedSource {
    // Validate IR if requested
    if (options.validate) {
        const validation = ir.validate();
        if (!validation.isValid()) {
            return CompileError.ValidationFailed;
        }

        // Validate against backend capabilities if provided
        if (options.capabilities) |caps| {
            const backend_validation = backend_mod.validateForBackend(ir, caps);
            if (!backend_validation.isValid()) {
                return CompileError.ValidationFailed;
            }
        }
    }

    // Generate source code for the target backend
    return generateSource(allocator, ir, target_backend);
}

/// Generate source code for a specific backend.
fn generateSource(
    allocator: std.mem.Allocator,
    ir: *const kernel_mod.KernelIR,
    target: gpu_backend.Backend,
) CompileError!backend_mod.GeneratedSource {
    return switch (target) {
        .cuda => blk: {
            var gen = generic.CudaGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .vulkan, .opengl, .opengles => blk: {
            // Generic GLSL generator produces Vulkan-style output (#version 450)
            var gen = generic.GlslGenerator.init(allocator);
            defer gen.deinit();
            var result = try gen.generate(ir);
            result.backend = target;
            break :blk result;
        },
        .metal => blk: {
            var gen = generic.MslGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .webgpu, .webgl2 => blk: {
            var gen = generic.WgslGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .fpga, .tpu, .intel_arc, .stdgpu, .simulated => blk: {
            // Fallback to CPU simulation
            var gen = generic.CudaGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
    };
}
