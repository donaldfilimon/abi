//! Unified Accelerator Interface
//!
//! Abstract interface for all compute backends: GPU (CUDA, Vulkan, Metal),
//! AI accelerators (TPU, NPU), and CPU (SIMD, scalar).

const std = @import("std");
const builtin = @import("builtin");

const hardware_detection = @import("hardware_detection.zig");
pub const BackendType = hardware_detection.BackendType;

/// Memory allocation on an accelerator device
pub const DeviceMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    backend: BackendType,

    pub fn isValid(self: DeviceMemory) bool {
        return self.ptr != null;
    }
};

/// High-level Tensor abstraction
pub const Tensor = struct {
    data: DeviceMemory,
    shape: []const usize,
    strides: []const usize,
    data_type: DataType,
    allocator: std.mem.Allocator,

    pub const DataType = enum { f16, f32, f64, i32, i64, u8 };

    pub fn init(allocator: std.mem.Allocator, accel: *Accelerator, shape: []const usize, dtype: DataType) !Tensor {
        var count: usize = 1;
        const my_shape = try allocator.alloc(usize, shape.len);
        const my_strides = try allocator.alloc(usize, shape.len);

        var stride: usize = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            my_shape[i] = shape[i];
            my_strides[i] = stride;
            count *= shape[i];
            stride *= shape[i];
        }

        const elem_size: usize = switch (dtype) {
            .f16 => 2,
            .f32, .i32 => 4,
            .f64, .i64 => 8,
            .u8 => 1,
        };

        const mem = try accel.alloc(count * elem_size);

        return Tensor{
            .data = mem,
            .shape = my_shape,
            .strides = my_strides,
            .data_type = dtype,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor, accel: *Accelerator) void {
        accel.free(&self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    pub fn elementCount(self: Tensor) usize {
        var count: usize = 1;
        for (self.shape) |dim| count *= dim;
        return count;
    }
};

/// Unified accelerator interface
pub const Accelerator = struct {
    backend: BackendType,
    allocator: std.mem.Allocator,
    device_id: u32,
    name: []const u8,

    pub fn alloc(self: *Accelerator, size: usize) !DeviceMemory {
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const ptr = self.allocator.alloc(u8, size) catch return error.OutOfMemory;
            return DeviceMemory{
                .ptr = @ptrCast(ptr.ptr),
                .size = size,
                .backend = self.backend,
            };
        }
        return error.BackendNotImplemented;
    }

    pub fn free(self: *Accelerator, mem: *DeviceMemory) void {
        if (mem.ptr == null) return;
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const slice: [*]u8 = @ptrCast(mem.ptr.?);
            self.allocator.free(slice[0..mem.size]);
        }
        mem.ptr = null;
    }

    pub fn copyToDevice(self: *Accelerator, dst: DeviceMemory, src: []const u8) !void {
        if (dst.ptr == null) return error.InvalidMemory;
        if (src.len > dst.size) return error.BufferTooSmall;
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const dst_slice: [*]u8 = @ptrCast(dst.ptr.?);
            @memcpy(dst_slice[0..src.len], src);
        }
    }

    pub fn copyFromDevice(self: *Accelerator, dst: []u8, src: DeviceMemory) !void {
        if (src.ptr == null) return error.InvalidMemory;
        if (dst.len > src.size) return error.BufferTooSmall;
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const src_slice: [*]const u8 = @ptrCast(src.ptr.?);
            @memcpy(dst, src_slice[0..dst.len]);
        }
    }
};

/// Tensor operations on accelerator
pub const TensorOps = struct {
    accel: *Accelerator,

    pub fn init(accel: *Accelerator) TensorOps {
        return .{ .accel = accel };
    }

    pub fn matmul(self: *TensorOps, c: Tensor, a: Tensor, b: Tensor) void {
        const m = a.shape[0];
        const k = a.shape[1];
        const n = b.shape[1];

        if (self.accel.backend == .cpu_simd or self.accel.backend == .cpu_fallback) {
            const a_ptr: [*]const f32 = @ptrCast(@alignCast(a.data.ptr.?));
            const b_ptr: [*]const f32 = @ptrCast(@alignCast(b.data.ptr.?));
            const c_ptr: [*]f32 = @ptrCast(@alignCast(c.data.ptr.?));

            for (0..m) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0;
                    for (0..k) |l| {
                        sum += a_ptr[i * k + l] * b_ptr[l * n + j];
                    }
                    c_ptr[i * n + j] = sum;
                }
            }
        }
    }

    pub fn conv2d(self: *TensorOps, output: Tensor, input: Tensor, kernel: Tensor, stride: usize, padding: usize) void {
        if (self.accel.backend == .cpu_fallback) {
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.data.ptr.?));
            const k_ptr: [*]const f32 = @ptrCast(@alignCast(kernel.data.ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(output.data.ptr.?));

            const N = input.shape[0];
            const Cin = input.shape[1];
            const H = input.shape[2];
            const W = input.shape[3];

            const Cout = kernel.shape[0];
            const KH = kernel.shape[2];
            const KW = kernel.shape[3];

            const H_out = output.shape[2];
            const W_out = output.shape[3];

            for (0..N) |n| {
                for (0..Cout) |co| {
                    for (0..H_out) |ho| {
                        for (0..W_out) |wo| {
                            var sum: f32 = 0;
                            const h_start = ho * stride;
                            const w_start = wo * stride;

                            for (0..Cin) |ci| {
                                for (0..KH) |kh| {
                                    for (0..KW) |kw| {
                                        const h_in = @as(isize, @intCast(h_start + kh)) - @as(isize, @intCast(padding));
                                        const w_in = @as(isize, @intCast(w_start + kw)) - @as(isize, @intCast(padding));

                                        if (h_in >= 0 and h_in < H and w_in >= 0 and w_in < W) {
                                            const val = in_ptr[n * Cin * H * W + ci * H * W + @as(usize, @intCast(h_in * @as(isize, @intCast(W)) + w_in))];
                                            const w_val = k_ptr[co * Cin * KH * KW + ci * KH * KW + kh * KW + kw];
                                            sum += val * w_val;
                                        }
                                    }
                                }
                            }
                            out_ptr[n * Cout * H_out * W_out + co * H_out * W_out + ho * W_out + wo] = sum;
                        }
                    }
                }
            }
        }
    }

    pub fn relu(self: *TensorOps, output: Tensor, input: Tensor) void {
        if (self.accel.backend == .cpu_fallback) {
            const count = input.elementCount();
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.data.ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(output.data.ptr.?));
            for (0..count) |i| {
                out_ptr[i] = @max(0, in_ptr[i]);
            }
        }
    }

    pub fn dropout(self: *TensorOps, output: Tensor, input: Tensor, rate: f32, training: bool) !void {
        if (self.accel.backend == .cpu_fallback) {
            const count = input.elementCount();
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.data.ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(output.data.ptr.?));

            if (training) {
                var prng = std.Random.DefaultPrng.init(0);
                const random = prng.random();
                const scale = 1.0 / (1.0 - rate);

                for (0..count) |i| {
                    if (random.float(f32) > rate) {
                        out_ptr[i] = in_ptr[i] * scale;
                    } else {
                        out_ptr[i] = 0;
                    }
                }
            } else {
                @memcpy(out_ptr[0..count], in_ptr[0..count]);
            }
        }
    }

    pub fn softmax(self: *TensorOps, output: Tensor, input: Tensor) void {
        if (self.accel.backend == .cpu_fallback) {
            const count = input.elementCount();
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.data.ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(output.data.ptr.?));

            var max_val: f32 = in_ptr[0];
            for (1..count) |i| max_val = @max(max_val, in_ptr[i]);

            var sum: f32 = 0;
            for (0..count) |i| {
                out_ptr[i] = @exp(in_ptr[i] - max_val);
                sum += out_ptr[i];
            }

            for (0..count) |i| out_ptr[i] /= sum;
        }
    }

    pub fn dotProduct(self: *TensorOps, a: Tensor, b: Tensor) f32 {
        if (self.accel.backend == .cpu_fallback) {
            const count = a.elementCount(); // Assume same shape
            const a_ptr: [*]const f32 = @ptrCast(@alignCast(a.data.ptr.?));
            const b_ptr: [*]const f32 = @ptrCast(@alignCast(b.data.ptr.?));

            var sum: f32 = 0;
            for (0..count) |i| sum += a_ptr[i] * b_ptr[i];
            return sum;
        }
        return 0;
    }

    pub fn batchNorm(self: *TensorOps, output: Tensor, input: Tensor, mean: Tensor, var_: Tensor, gamma: Tensor, beta: Tensor, eps: f32) void {
        // CPU Fallback for BN inference (training requires more)
        if (self.accel.backend == .cpu_fallback) {
            const count = input.elementCount();
            const C = input.shape[1]; // Assume [N, C, H, W] or [N, C]

            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.data.ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(output.data.ptr.?));
            const m_ptr: [*]const f32 = @ptrCast(@alignCast(mean.data.ptr.?));
            const v_ptr: [*]const f32 = @ptrCast(@alignCast(var_.data.ptr.?));
            const g_ptr: [*]const f32 = @ptrCast(@alignCast(gamma.data.ptr.?));
            const b_ptr: [*]const f32 = @ptrCast(@alignCast(beta.data.ptr.?));

            // Simple 1D loop mapping, ignoring strides for now (assume packed)
            // Logic to map index to channel c depends on shape
            // For simplicity, handle [N, C] case
            if (input.shape.len == 2) {
                const N = input.shape[0];
                for (0..N) |n| {
                    for (0..C) |c| {
                        const idx = n * C + c;
                        const norm = (in_ptr[idx] - m_ptr[c]) / @sqrt(v_ptr[c] + eps);
                        out_ptr[idx] = g_ptr[c] * norm + b_ptr[c];
                    }
                }
            } else {
                // Fallback: copy
                @memcpy(out_ptr[0..count], in_ptr[0..count]);
            }
        }
    }
};

pub fn createBestAccelerator(allocator: std.mem.Allocator) Accelerator {
    return Accelerator{
        .backend = .cpu_fallback,
        .allocator = allocator,
        .device_id = 0,
        .name = "CPU Fallback",
    };
}
