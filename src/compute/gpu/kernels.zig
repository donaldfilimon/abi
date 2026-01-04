//! GPU kernel execution and compilation
//!
//! Provides kernel compilation, execution, and synchronization primitives
//! for CUDA, Vulkan, Metal, WebGPU, and OpenGL-family backends.

const std = @import("std");
const backend = @import("backend.zig");
const build_options = @import("build_options");
const types = @import("kernel_types.zig");

pub const KernelError = types.KernelError;
pub const KernelSource = types.KernelSource;
pub const KernelConfig = types.KernelConfig;
pub const Stream = types.Stream;

const BackendOps = struct {
    compile: *const fn (std.mem.Allocator, KernelSource) KernelError!*anyopaque,
    launch: *const fn (
        std.mem.Allocator,
        *anyopaque,
        KernelConfig,
        []const ?*const anyopaque,
    ) KernelError!void,
    destroy: *const fn (std.mem.Allocator, *anyopaque) void,
};

fn backendOps(which_backend: backend.Backend) ?BackendOps {
    return switch (which_backend) {
        .cuda => if (comptime build_options.gpu_cuda) blk: {
            const cuda_module = @import("backends/cuda.zig");
            break :blk .{
                .compile = cuda_module.compileKernel,
                .launch = cuda_module.launchKernel,
                .destroy = cuda_module.destroyKernel,
            };
        } else null,
        .vulkan => if (comptime build_options.gpu_vulkan) blk: {
            const vulkan_module = @import("backends/vulkan.zig");
            break :blk .{
                .compile = vulkan_module.compileKernel,
                .launch = vulkan_module.launchKernel,
                .destroy = vulkan_module.destroyKernel,
            };
        } else null,
        .metal => if (comptime build_options.gpu_metal) blk: {
            const metal_module = @import("backends/metal.zig");
            break :blk .{
                .compile = metal_module.compileKernel,
                .launch = metal_module.launchKernel,
                .destroy = metal_module.destroyKernel,
            };
        } else null,
        .webgpu => if (comptime build_options.gpu_webgpu) blk: {
            const webgpu_module = @import("backends/webgpu.zig");
            break :blk .{
                .compile = webgpu_module.compileKernel,
                .launch = webgpu_module.launchKernel,
                .destroy = webgpu_module.destroyKernel,
            };
        } else null,
        .opengl => if (comptime build_options.gpu_opengl) blk: {
            const opengl_module = @import("backends/opengl.zig");
            break :blk .{
                .compile = opengl_module.compileKernel,
                .launch = opengl_module.launchKernel,
                .destroy = opengl_module.destroyKernel,
            };
        } else null,
        .opengles => if (comptime build_options.gpu_opengles) blk: {
            const opengles_module = @import("backends/opengles.zig");
            break :blk .{
                .compile = opengles_module.compileKernel,
                .launch = opengles_module.launchKernel,
                .destroy = opengles_module.destroyKernel,
            };
        } else null,
        .webgl2 => if (comptime build_options.gpu_webgl2) blk: {
            const webgl2_module = @import("backends/webgl2.zig");
            break :blk .{
                .compile = webgl2_module.compileKernel,
                .launch = webgl2_module.launchKernel,
                .destroy = webgl2_module.destroyKernel,
            };
        } else null,
    };
}

pub const CompiledKernel = struct {
    name: []const u8,
    backend: backend.Backend,
    handle: ?*anyopaque,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompiledKernel) void {
        self.allocator.free(self.name);
        if (self.handle) |handle| {
            if (backendOps(self.backend)) |ops| {
                ops.destroy(self.allocator, handle);
            }
        }
        self.* = undefined;
    }

    pub fn launch(
        self: *const CompiledKernel,
        allocator: std.mem.Allocator,
        config: KernelConfig,
        args: []const ?*const anyopaque,
    ) KernelError!void {
        const ops = backendOps(self.backend) orelse return KernelError.UnsupportedBackend;
        const handle = self.handle orelse return KernelError.InvalidArguments;
        try ops.launch(allocator, handle, config, args);
    }
};

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: KernelSource,
) KernelError!CompiledKernel {
    if (!backend.backendSupportsKernels(source.backend)) {
        return KernelError.UnsupportedBackend;
    }

    const name = allocator.dupe(u8, source.name) catch return KernelError.CompilationFailed;
    errdefer allocator.free(name);

    const ops = backendOps(source.backend) orelse return KernelError.UnsupportedBackend;
    const handle = try ops.compile(allocator, source);

    return .{
        .name = name,
        .backend = source.backend,
        .handle = handle,
        .allocator = allocator,
    };
}

const KernelBuilder = fn (std.mem.Allocator) anyerror!KernelSource;

const cuda_kernel_builders = [_]KernelBuilder{
    createCudaVectorAddKernel,
    createCudaMatMulKernel,
    createCudaReduceKernel,
};
const vulkan_kernel_builders = [_]KernelBuilder{
    createVulkanVectorAddKernel,
    createVulkanMatMulKernel,
};
const metal_kernel_builders = [_]KernelBuilder{
    createMetalVectorAddKernel,
    createMetalMatMulKernel,
};
const webgpu_kernel_builders = [_]KernelBuilder{
    createWebGpuVectorAddKernel,
    createWebGpuReduceKernel,
};
const opengl_kernel_builders = [_]KernelBuilder{
    createOpenGlVectorAddKernel,
};
const opengles_kernel_builders = [_]KernelBuilder{
    createOpenGlesVectorAddKernel,
};
const webgl2_kernel_builders = [_]KernelBuilder{};

const backend_kernel_builders = [_][]const KernelBuilder{
    cuda_kernel_builders[0..],
    vulkan_kernel_builders[0..],
    metal_kernel_builders[0..],
    webgpu_kernel_builders[0..],
    opengl_kernel_builders[0..],
    opengles_kernel_builders[0..],
    webgl2_kernel_builders[0..],
};

fn kernelBuildersForBackend(which_backend: backend.Backend) []const KernelBuilder {
    return backend_kernel_builders[@intFromEnum(which_backend)];
}

pub fn createDefaultKernels(allocator: std.mem.Allocator) ![]KernelSource {
    var kernels = std.ArrayListUnmanaged(KernelSource).empty;
    errdefer {
        for (kernels.items) |*kernel| {
            kernel.deinit(allocator);
        }
        kernels.deinit(allocator);
    }

    const backends = try backend.availableBackends(allocator);
    defer allocator.free(backends);

    for (backends) |available_backend| {
        for (kernelBuildersForBackend(available_backend)) |builder| {
            try kernels.append(allocator, try builder(allocator));
        }
    }

    return kernels.toOwnedSlice(allocator);
}

fn createCudaVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\__global__ void vector_add(const float* a, const float* b, float* c, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {
        \\        c[idx] = a[idx] + b[idx];
        \\    }
        \\}
    );
    const entry_point = try allocator.dupe(u8, "vector_add");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .cuda,
    };
}

fn createCudaMatMulKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "matmul");
    const source = try allocator.dupe(u8,
        \\__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
        \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
        \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (row < M && col < N) {
        \\        float sum = 0.0f;
        \\        for (int k = 0; k < K; k++) {
        \\            sum += A[row * K + k] * B[k * N + col];
        \\        }
        \\        C[row * N + col] = sum;
        \\    }
        \\}
    );
    const entry_point = try allocator.dupe(u8, "matmul");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .cuda,
    };
}

fn createCudaReduceKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "reduce_sum");
    const source = try allocator.dupe(u8,
        \\__global__ void reduce_sum(const float* input, float* output, int n) {
        \\    extern __shared__ float sdata[];
        \\    unsigned int tid = threadIdx.x;
        \\    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    sdata[tid] = (i < n) ? input[i] : 0.0f;
        \\    __syncthreads();
        \\    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        \\        if (tid < s) sdata[tid] += sdata[tid + s];
        \\        __syncthreads();
        \\    }
        \\    if (tid == 0) output[blockIdx.x] = sdata[0];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "reduce_sum");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .cuda,
    };
}

fn createVulkanVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\#version 450
        \\layout(local_size_x = 256) in;
        \\layout(binding = 0) readonly buffer A { float a[]; };
        \\layout(binding = 1) readonly buffer B { float b[]; };
        \\layout(binding = 2) writeonly buffer C { float c[]; };
        \\void main() {
        \\    uint idx = gl_GlobalInvocationID.x;
        \\    c[idx] = a[idx] + b[idx];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .vulkan,
    };
}

fn createVulkanMatMulKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "matmul");
    const source = try allocator.dupe(u8,
        \\#version 450
        \\layout(local_size_x = 16, local_size_y = 16) in;
        \\layout(binding = 0) readonly buffer A { float a[]; };
        \\layout(binding = 1) readonly buffer B { float b[]; };
        \\layout(binding = 2) writeonly buffer C { float c[]; };
        \\layout(push_constant) uniform Params {
        \\    uint M, N, K;
        \\} params;
        \\void main() {
        \\    uint row = gl_GlobalInvocationID.y;
        \\    uint col = gl_GlobalInvocationID.x;
        \\    if (row < params.M && col < params.N) {
        \\        float sum = 0.0;
        \\        for (uint k = 0; k < params.K; k++) {
        \\            sum += a[row * params.K + k] * b[k * params.N + col];
        \\        }
        \\        c[row * params.N + col] = sum;
        \\    }
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .vulkan,
    };
}

fn createMetalVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void vector_add(const device float* a [[buffer(0)]],
        \\                         const device float* b [[buffer(1)]],
        \\                         device float* c [[buffer(2)]],
        \\                         uint idx [[thread_position_in_grid]]) {
        \\    c[idx] = a[idx] + b[idx];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "vector_add");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .metal,
    };
}

fn createMetalMatMulKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "matmul");
    const source = try allocator.dupe(u8,
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void matmul(const device float* A [[buffer(0)]],
        \\                    const device float* B [[buffer(1)]],
        \\                    device float* C [[buffer(2)]],
        \\                    constant uint& M [[buffer(3)]],
        \\                    constant uint& N [[buffer(4)]],
        \\                    constant uint& K [[buffer(5)]],
        \\                    uint2 gid [[thread_position_in_grid]]) {
        \\    uint row = gid.y;
        \\    uint col = gid.x;
        \\    if (row < M && col < N) {
        \\        float sum = 0.0f;
        \\        for (uint k = 0; k < K; k++) {
        \\            sum += A[row * K + k] * B[k * N + col];
        \\        }
        \\        C[row * N + col] = sum;
        \\    }
        \\}
    );
    const entry_point = try allocator.dupe(u8, "matmul");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .metal,
    };
}

fn createWebGpuVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\@group(0) @binding(0) var<storage, read> a : array<f32>;
        \\@group(0) @binding(1) var<storage, read> b : array<f32>;
        \\@group(0) @binding(2) var<storage, read_write> c : array<f32>;
        \\@compute @workgroup_size(256)
        \\fn main(@builtin(global_invocation_id) idx : vec3<u32>) {
        \\    c[idx.x] = a[idx.x] + b[idx.x];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .webgpu,
    };
}

fn createWebGpuReduceKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "reduce_sum");
    const source = try allocator.dupe(u8,
        \\@group(0) @binding(0) var<storage, read> input : array<f32>;
        \\@group(0) @binding(1) var<storage, read_write> output : array<f32>;
        \\var<workgroup> sdata : array<f32, 256>;
        \\@compute @workgroup_size(256)
        \\fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        \\         @builtin(local_invocation_id) local_id : vec3<u32>,
        \\         @builtin(num_workgroups) num_workgroups : vec3<u32>,
        \\         @builtin(workgroup_id) workgroup_id : vec3<u32>) {
        \\    let n = arrayLength(&input);
        \\    let i = workgroup_id.x * 256u + local_id.x;
        \\    sdata[local_id.x] = select(0.0, input[i], i < n);
        \\    workgroupBarrier();
        \\    var s = 128u;
        \\    loop {
        \\        if (local_id.x < s) {
        \\            sdata[local_id.x] += sdata[local_id.x + s];
        \\        }
        \\        workgroupBarrier();
        \\        if (s <= 1u) { break; }
        \\        s /= 2u;
        \\    }
        \\    if (local_id.x == 0u) {
        \\        output[workgroup_id.x] = sdata[0];
        \\    }
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .webgpu,
    };
}

fn createOpenGlVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\#version 430
        \\layout(local_size_x = 256) in;
        \\layout(std430, binding = 0) readonly buffer A { float a[]; };
        \\layout(std430, binding = 1) readonly buffer B { float b[]; };
        \\layout(std430, binding = 2) writeonly buffer C { float c[]; };
        \\void main() {
        \\    uint idx = gl_GlobalInvocationID.x;
        \\    c[idx] = a[idx] + b[idx];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .opengl,
    };
}

fn createOpenGlesVectorAddKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "vector_add");
    const source = try allocator.dupe(u8,
        \\#version 310 es
        \\layout(local_size_x = 256) in;
        \\layout(std430, binding = 0) readonly buffer A { float a[]; };
        \\layout(std430, binding = 1) readonly buffer B { float b[]; };
        \\layout(std430, binding = 2) writeonly buffer C { float c[]; };
        \\void main() {
        \\    uint idx = gl_GlobalInvocationID.x;
        \\    c[idx] = a[idx] + b[idx];
        \\}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .opengles,
    };
}

fn createWebGl2PlaceholderKernel(allocator: std.mem.Allocator) !KernelSource {
    const name = try allocator.dupe(u8, "noop");
    const source = try allocator.dupe(u8,
        \\#version 300 es
        \\void main() {}
    );
    const entry_point = try allocator.dupe(u8, "main");
    return .{
        .name = name,
        .source = source,
        .entry_point = entry_point,
        .backend = .webgl2,
    };
}

test "create default kernels for available backends" {
    const allocator = std.testing.allocator;
    const available = try backend.availableBackends(allocator);
    defer allocator.free(available);
    const kernels = try createDefaultKernels(allocator);
    defer {
        for (kernels) |*kernel| {
            kernel.deinit(allocator);
        }
        allocator.free(kernels);
    }
    if (available.len == 0) {
        try std.testing.expectEqual(@as(usize, 0), kernels.len);
        return;
    }
    try std.testing.expect(kernels.len > 0);
}

fn toAnyConstPtr(ptr: anytype) *const anyopaque {
    return @ptrCast(@constCast(ptr));
}

fn findKernelByName(
    allocator: std.mem.Allocator,
    which_backend: backend.Backend,
    name: []const u8,
) !?KernelSource {
    for (kernelBuildersForBackend(which_backend)) |builder| {
        var source = try builder(allocator);
        if (std.ascii.eqlIgnoreCase(source.name, name)) {
            return source;
        }
        source.deinit(allocator);
    }
    return null;
}

test "vector_add kernels execute on all enabled backends" {
    const allocator = std.testing.allocator;
    const input_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    const expected = [_]f32{ 5.0, 5.0, 5.0, 5.0 };

    for (std.enums.values(backend.Backend)) |which_backend| {
        if (!backend.isEnabled(which_backend)) continue;
        if (!backend.backendSupportsKernels(which_backend)) continue;

        var source = (try findKernelByName(allocator, which_backend, "vector_add")) orelse {
            continue;
        };
        defer source.deinit(allocator);

        var compiled = try compileKernel(allocator, source);
        defer compiled.deinit();

        var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const count: u32 = @intCast(input_a.len);
        const args = [_]?*const anyopaque{
            toAnyConstPtr(&input_a[0]),
            toAnyConstPtr(&input_b[0]),
            toAnyConstPtr(&output[0]),
            toAnyConstPtr(&count),
        };
        try compiled.launch(allocator, .{}, &args);
        try std.testing.expectEqualSlices(f32, &expected, &output);
    }
}

test "reduce_sum kernels execute when available" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const count: u32 = @intCast(input.len);

    for (std.enums.values(backend.Backend)) |which_backend| {
        if (!backend.isEnabled(which_backend)) continue;
        if (!backend.backendSupportsKernels(which_backend)) continue;

        var source = (try findKernelByName(allocator, which_backend, "reduce_sum")) orelse {
            continue;
        };
        defer source.deinit(allocator);

        var compiled = try compileKernel(allocator, source);
        defer compiled.deinit();

        var output = [_]f32{0.0};
        const args = [_]?*const anyopaque{
            toAnyConstPtr(&input[0]),
            toAnyConstPtr(&output[0]),
            toAnyConstPtr(&count),
        };
        try compiled.launch(allocator, .{}, &args);
        try std.testing.expectApproxEqAbs(@as(f32, 10.0), output[0], 0.0001);
    }
}

test "kernel builder coverage matches backend support" {
    for (std.enums.values(backend.Backend)) |which_backend| {
        const builders = kernelBuildersForBackend(which_backend);
        const supports = backend.backendSupportsKernels(which_backend);
        if (supports) {
            try std.testing.expect(builders.len > 0);
        } else {
            try std.testing.expectEqual(@as(usize, 0), builders.len);
        }
    }
}

test "kernel compilation stubs cover supported backends" {
    const allocator = std.testing.allocator;
    for (std.enums.values(backend.Backend)) |which_backend| {
        if (!backend.isEnabled(which_backend)) continue;
        if (!backend.backendSupportsKernels(which_backend)) continue;
        const builders = kernelBuildersForBackend(which_backend);
        if (builders.len == 0) continue;

        var source = try builders[0](allocator);
        defer source.deinit(allocator);

        var compiled = try compileKernel(allocator, source);
        defer compiled.deinit();

        _ = compiled.launch(allocator, .{}, &.{}) catch |err| {
            switch (err) {
                KernelError.LaunchFailed, KernelError.InvalidArguments, KernelError.UnsupportedBackend => {},
                else => return err,
            }
            return;
        };
    }
}

test "webgl2 kernels are unsupported" {
    const allocator = std.testing.allocator;
    var source = try createWebGl2PlaceholderKernel(allocator);
    defer source.deinit(allocator);
    try std.testing.expectError(KernelError.UnsupportedBackend, compileKernel(allocator, source));
}

test "stream synchronization works" {
    var stream = Stream.init(.cuda);
    stream.stop();
    stream.synchronize();
    try std.testing.expect(!stream.running.load(.acquire));
}
