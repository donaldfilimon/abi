const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const backend_mod = @import("../backend.zig");
const simulated = @import("../factory/simulated.zig");
const backend_shared = @import("shared.zig");

pub const Backend = backend_mod.Backend;

pub fn createVTable(allocator: std.mem.Allocator, backend: Backend) interface.BackendError!interface.Backend {
    if (!isEnabledAtBuild(backend)) {
        return interface.BackendError.NotAvailable;
    }

    return switch (backend) {
        .cuda => createCuda(allocator),
        .vulkan => createVulkan(allocator),
        .metal => createMetal(allocator),
        .webgpu => createWebGPU(allocator),
        .opengl => createOpenGL(allocator),
        .opengles => createOpenGLES(allocator),
        .fpga => createFpga(allocator),
        .tpu => interface.BackendError.NotAvailable,
        .stdgpu, .simulated => simulated.createSimulatedVTable(allocator),
        .webgl2 => simulated.createSimulatedVTable(allocator),
    };
}

pub fn isEnabledAtBuild(backend: Backend) bool {
    return switch (backend) {
        .cuda => build_options.gpu_cuda and backend_shared.dynlibSupported,
        .vulkan => build_options.gpu_vulkan,
        .stdgpu => build_options.gpu_stdgpu,
        .metal => build_options.gpu_metal,
        .webgpu => build_options.gpu_webgpu,
        .opengl => build_options.gpu_opengl,
        .opengles => build_options.gpu_opengles,
        .webgl2 => build_options.gpu_webgl2,
        .fpga => if (@hasDecl(build_options, "gpu_fpga")) build_options.gpu_fpga else false,
        .tpu => if (@hasDecl(build_options, "gpu_tpu")) build_options.gpu_tpu else false,
        .simulated => build_options.enable_gpu,
    };
}

fn createCuda(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
        const cuda_vtable = @import("cuda/vtable.zig");
        return cuda_vtable.createCudaVTable(allocator);
    }
    return interface.BackendError.NotAvailable;
}

fn createVulkan(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    if (comptime !build_options.gpu_vulkan) return interface.BackendError.NotAvailable;
    const vulkan = @import("vulkan.zig");
    return vulkan.createVulkanVTable(allocator);
}

fn createMetal(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    if (comptime !build_options.gpu_metal) return interface.BackendError.NotAvailable;
    const metal_vtable = @import("metal_vtable.zig");
    return metal_vtable.createMetalVTable(allocator);
}

fn createWebGPU(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    if (comptime !build_options.gpu_webgpu) return interface.BackendError.NotAvailable;
    const webgpu_vtable = @import("webgpu_vtable.zig");
    return webgpu_vtable.createWebGpuVTable(allocator);
}

fn createOpenGL(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const gl_backend = @import("gl/backend.zig");
    const gl_profile = @import("gl/profile.zig");
    return gl_backend.createVTableForProfile(allocator, gl_profile.Profile.desktop);
}

fn createOpenGLES(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const gl_backend = @import("gl/backend.zig");
    const gl_profile = @import("gl/profile.zig");
    return gl_backend.createVTableForProfile(allocator, gl_profile.Profile.es);
}

fn createFpga(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    if (comptime !build_options.gpu_fpga) return interface.BackendError.NotAvailable;
    const fpga_vtable = @import("fpga/vtable.zig");
    return fpga_vtable.createFpgaVTable(allocator);
}

fn assertRegistryTag(comptime backend: Backend) void {
    _ = switch (backend) {
        .cuda,
        .vulkan,
        .stdgpu,
        .metal,
        .webgpu,
        .opengl,
        .opengles,
        .webgl2,
        .fpga,
        .tpu,
        .simulated,
        => {},
    };
}

comptime {
    for (std.meta.tags(Backend)) |backend| {
        assertRegistryTag(backend);
    }
}

test "createVTable rejects disabled backends" {
    inline for (std.meta.tags(Backend)) |backend| {
        if (!isEnabledAtBuild(backend)) {
            try std.testing.expectError(
                interface.BackendError.NotAvailable,
                createVTable(std.testing.allocator, backend),
            );
        }
    }
}

test "stdgpu and webgl2 are strict when disabled" {
    if (!isEnabledAtBuild(.stdgpu)) {
        try std.testing.expectError(
            interface.BackendError.NotAvailable,
            createVTable(std.testing.allocator, .stdgpu),
        );
    }
    if (!isEnabledAtBuild(.webgl2)) {
        try std.testing.expectError(
            interface.BackendError.NotAvailable,
            createVTable(std.testing.allocator, .webgl2),
        );
    }
}
