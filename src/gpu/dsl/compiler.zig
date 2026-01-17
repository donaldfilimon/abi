//! Unified Kernel Compiler
//!
//! Compiles kernel IR to backend-specific source code.
//! This is the main entry point for kernel compilation.

const std = @import("std");
const kernel_mod = @import("kernel.zig");
const backend_mod = @import("codegen/backend.zig");
const cuda = @import("codegen/cuda.zig");
const glsl = @import("codegen/glsl.zig");
const wgsl = @import("codegen/wgsl.zig");
const msl = @import("codegen/msl.zig");
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
            var gen = cuda.CudaGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .vulkan => blk: {
            var gen = glsl.GlslGenerator.init(allocator, .vulkan);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .opengl => blk: {
            var gen = glsl.GlslGenerator.init(allocator, .opengl);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .opengles => blk: {
            var gen = glsl.GlslGenerator.init(allocator, .opengles);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .metal => blk: {
            var gen = msl.MslGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .webgpu => blk: {
            var gen = wgsl.WgslGenerator.init(allocator);
            defer gen.deinit();
            break :blk try gen.generate(ir);
        },
        .stdgpu => blk: {
            // stdgpu uses GLSL-like format
            var gen = glsl.GlslGenerator.init(allocator, .vulkan);
            defer gen.deinit();
            var result = try gen.generate(ir);
            result.backend = .stdgpu;
            break :blk result;
        },
        .webgl2 => return CompileError.UnsupportedBackend,
    };
}

/// Compile kernel IR to a KernelSource for use with the existing kernel system.
pub fn compileToKernelSource(
    allocator: std.mem.Allocator,
    ir: *const kernel_mod.KernelIR,
    target_backend: gpu_backend.Backend,
    options: CompileOptions,
) CompileError!kernel_types.KernelSource {
    const source = try compile(allocator, ir, target_backend, options);

    // KernelSource takes ownership of strings
    return kernel_types.KernelSource{
        .name = try allocator.dupe(u8, ir.name),
        .source = source.code,
        .entry_point = source.entry_point,
        .backend = target_backend,
    };
}

/// Compile kernel IR to all available backends.
pub fn compileAll(
    allocator: std.mem.Allocator,
    ir: *const kernel_mod.KernelIR,
    options: CompileOptions,
) ![]backend_mod.GeneratedSource {
    const available = try gpu_backend.availableBackends(allocator);
    defer allocator.free(available);

    var sources = std.ArrayListUnmanaged(backend_mod.GeneratedSource){};
    errdefer {
        for (sources.items) |*src| {
            src.deinit(allocator);
        }
        sources.deinit(allocator);
    }

    for (available) |target| {
        if (gpu_backend.backendSupportsKernels(target)) {
            const src = compile(allocator, ir, target, options) catch continue;
            try sources.append(allocator, src);
        }
    }

    return sources.toOwnedSlice(allocator);
}

/// Get the best available backend for compilation.
pub fn getBestBackend(allocator: std.mem.Allocator) !gpu_backend.Backend {
    const available = try gpu_backend.availableBackends(allocator);
    defer allocator.free(available);

    // Priority order: CUDA > Vulkan > Metal > WebGPU > OpenGL > stdgpu
    const priority = [_]gpu_backend.Backend{
        .cuda,
        .vulkan,
        .metal,
        .webgpu,
        .opengl,
        .opengles,
        .stdgpu,
    };

    for (priority) |preferred| {
        for (available) |avail| {
            if (avail == preferred and gpu_backend.backendSupportsKernels(avail)) {
                return avail;
            }
        }
    }

    return error.BackendUnavailable;
}

/// Check if a backend supports kernel compilation.
pub fn backendSupportsCompilation(target: gpu_backend.Backend) bool {
    return switch (target) {
        .webgl2 => false, // WebGL2 doesn't support compute shaders
        else => true,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "compile empty kernel to CUDA" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = kernel_mod.KernelIR.empty("test_kernel");
    var result = try compile(allocator, &ir, .cuda, .{});
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "__global__") != null);
}

test "compile empty kernel to GLSL" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = kernel_mod.KernelIR.empty("test_kernel");
    var result = try compile(allocator, &ir, .vulkan, .{});
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 450") != null);
}

test "compile empty kernel to WGSL" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = kernel_mod.KernelIR.empty("test_kernel");
    var result = try compile(allocator, &ir, .webgpu, .{});
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "@compute") != null);
}

test "compile empty kernel to MSL" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = kernel_mod.KernelIR.empty("test_kernel");
    var result = try compile(allocator, &ir, .metal, .{});
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "kernel void") != null);
}

test "backendSupportsCompilation" {
    try std.testing.expect(backendSupportsCompilation(.cuda));
    try std.testing.expect(backendSupportsCompilation(.vulkan));
    try std.testing.expect(backendSupportsCompilation(.webgpu));
    try std.testing.expect(!backendSupportsCompilation(.webgl2));
}
