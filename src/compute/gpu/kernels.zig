//! GPU kernel execution and compilation
//!
//! Provides kernel compilation, execution, and synchronization primitives
//! for CUDA, Vulkan, Metal, and WebGPU backends.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const memory = @import("memory.zig");
const mod = @import("mod.zig");

pub const KernelError = error{
    CompilationFailed,
    LaunchFailed,
    InvalidArguments,
    UnsupportedBackend,
    MissingDevice,
};

pub const KernelSource = struct {
    name: []const u8,
    source: []const u8,
    entry_point: []const u8 = "main",
    backend: mod.Backend,

    pub fn deinit(self: *KernelSource, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.source);
        allocator.free(self.entry_point);
        self.* = undefined;
    }
};

pub const KernelConfig = struct {
    grid_dim: [3]u32 = .{ 1, 1, 1 },
    block_dim: [3]u32 = .{ 1, 1, 1 },
    shared_memory_bytes: u32 = 0,
    stream: ?*Stream = null,
};

pub const Stream = struct {
    id: u64,
    backend: mod.Backend,
    running: std.atomic.Value(bool),

    pub fn init(backend: mod.Backend) Stream {
        return .{
            .id = std.time.nanoTimestamp(),
            .backend = backend,
            .running = std.atomic.Value(bool).init(true),
        };
    }

    pub fn synchronize(self: *Stream) void {
        while (self.running.load(.acquire)) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn stop(self: *Stream) void {
        self.running.store(false, .release);
    }
};

pub const CompiledKernel = struct {
    name: []const u8,
    backend: mod.Backend,
    handle: ?*anyopaque,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompiledKernel) void {
        self.allocator.free(self.name);
        if (self.handle) |handle| {
            switch (self.backend) {
                .cuda => if (comptime build_options.gpu_cuda) {
                    const cuda_module = @import("backends/cuda.zig");
                    cuda_module.destroyKernel(handle);
                },
                .vulkan => if (comptime build_options.gpu_vulkan) {
                    const vulkan_module = @import("backends/vulkan.zig");
                    vulkan_module.destroyKernel(handle);
                },
                .metal => if (comptime build_options.gpu_metal) {
                    const metal_module = @import("backends/metal.zig");
                    metal_module.destroyKernel(handle);
                },
                .webgpu => if (comptime build_options.gpu_webgpu) {
                    const webgpu_module = @import("backends/webgpu.zig");
                    webgpu_module.destroyKernel(handle);
                },
                else => {},
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
        switch (self.backend) {
            .cuda => if (comptime build_options.gpu_cuda) {
                const cuda_module = @import("backends/cuda.zig");
                try cuda_module.launchKernel(
                    allocator,
                    self.handle orelse return KernelError.InvalidArguments,
                    config,
                    args,
                );
            } else {
                return KernelError.UnsupportedBackend;
            },
            .vulkan => if (comptime build_options.gpu_vulkan) {
                const vulkan_module = @import("backends/vulkan.zig");
                try vulkan_module.launchKernel(
                    allocator,
                    self.handle orelse return KernelError.InvalidArguments,
                    config,
                    args,
                );
            } else {
                return KernelError.UnsupportedBackend;
            },
            .metal => if (comptime build_options.gpu_metal) {
                const metal_module = @import("backends/metal.zig");
                try metal_module.launchKernel(
                    allocator,
                    self.handle orelse return KernelError.InvalidArguments,
                    config,
                    args,
                );
            } else {
                return KernelError.UnsupportedBackend;
            },
            .webgpu => if (comptime build_options.gpu_webgpu) {
                const webgpu_module = @import("backends/webgpu.zig");
                try webgpu_module.launchKernel(
                    allocator,
                    self.handle orelse return KernelError.InvalidArguments,
                    config,
                    args,
                );
            } else {
                return KernelError.UnsupportedBackend;
            },
            else => return KernelError.UnsupportedBackend,
        }
    }
};

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: KernelSource,
) KernelError!CompiledKernel {
    const name = try allocator.dupe(u8, source.name);
    errdefer allocator.free(name);

    var handle: ?*anyopaque = null;
    switch (source.backend) {
        .cuda => if (comptime build_options.gpu_cuda) {
            const cuda_module = @import("backends/cuda.zig");
            handle = try cuda_module.compileKernel(allocator, source);
        } else {
            return KernelError.UnsupportedBackend;
        },
        .vulkan => if (comptime build_options.gpu_vulkan) {
            const vulkan_module = @import("backends/vulkan.zig");
            handle = try vulkan_module.compileKernel(allocator, source);
        } else {
            return KernelError.UnsupportedBackend;
        },
        .metal => if (comptime build_options.gpu_metal) {
            const metal_module = @import("backends/metal.zig");
            handle = try metal_module.compileKernel(allocator, source);
        } else {
            return KernelError.UnsupportedBackend;
        },
        .webgpu => if (comptime build_options.gpu_webgpu) {
            const webgpu_module = @import("backends/webgpu.zig");
            handle = try webgpu_module.compileKernel(allocator, source);
        } else {
            return KernelError.UnsupportedBackend;
        },
        else => return KernelError.UnsupportedBackend,
    }

    return .{
        .name = name,
        .backend = source.backend,
        .handle = handle,
        .allocator = allocator,
    };
}

pub fn createDefaultKernels(allocator: std.mem.Allocator) ![]KernelSource {
    var kernels = std.ArrayListUnmanaged(KernelSource).empty;
    errdefer {
        for (kernels.items) |*kernel| {
            kernel.deinit(allocator);
        }
        kernels.deinit(allocator);
    }

    const backends = try mod.availableBackends(allocator);
    defer allocator.free(backends);

    for (backends) |backend| {
        switch (backend) {
            .cuda => {
                try kernels.append(allocator, try createCudaVectorAddKernel(allocator));
                try kernels.append(allocator, try createCudaMatMulKernel(allocator));
                try kernels.append(allocator, try createCudaReduceKernel(allocator));
            },
            .vulkan => {
                try kernels.append(allocator, try createVulkanVectorAddKernel(allocator));
                try kernels.append(allocator, try createVulkanMatMulKernel(allocator));
            },
            .metal => {
                try kernels.append(allocator, try createMetalVectorAddKernel(allocator));
                try kernels.append(allocator, try createMetalMatMulKernel(allocator));
            },
            .webgpu => {
                try kernels.append(allocator, try createWebGpuVectorAddKernel(allocator));
                try kernels.append(allocator, try createWebGpuReduceKernel(allocator));
            },
            else => {},
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

test "create default kernels for available backends" {
    const allocator = std.testing.allocator;
    const kernels = try createDefaultKernels(allocator);
    defer {
        for (kernels) |*kernel| {
            kernel.deinit(allocator);
        }
        allocator.free(kernels);
    }
    try std.testing.expect(kernels.len > 0);
}

test "stream synchronization works" {
    var stream = Stream.init(.cuda);
    stream.stop();
    stream.synchronize();
    try std.testing.expect(!stream.running.load(.acquire));
}
