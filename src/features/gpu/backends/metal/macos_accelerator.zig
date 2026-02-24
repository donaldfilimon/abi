//! macOS Unified Acceleration Pipeline
//!
//! Routes LLM inference operations across Apple's three compute tiers:
//!   1. Accelerate (AMX) — CPU-side matrix ops via vBLAS/vDSP
//!   2. MPS (GPU) — Metal Performance Shaders for large tensor ops
//!   3. CoreML (Neural Engine) — full model inference on the ANE
//!
//! The router selects the optimal backend based on operation class and data size.
//! Thresholds are configurable and reflect that GPU dispatch overhead only pays off
//! above certain sizes, while the Neural Engine excels at full-model inference.
//!
//! ## Routing Table
//! ```
//! OperationClass          → AcceleratorBackend
//! ──────────────────────────────────────────────
//! elementwise_small        → .accelerate_cpu (AMX)
//! matmul_small             → .accelerate_cpu (AMX)
//! activation               → .accelerate_cpu (AMX)
//! embedding_lookup         → .accelerate_cpu (AMX)
//! elementwise_large        → .mps_gpu
//! matmul_large             → .mps_gpu
//! attention                → .mps_gpu
//! full_model               → .coreml_ne
//! ```

const std = @import("std");
const builtin = @import("builtin");
const accelerate = @import("accelerate.zig");
const mps = @import("mps.zig");
const coreml = @import("coreml.zig");
const gpu_family = @import("gpu_family.zig");

/// Backend selection for a given operation.
pub const AcceleratorBackend = enum {
    /// Apple AMX via Accelerate framework (vBLAS/vDSP)
    accelerate_cpu,
    /// Metal Performance Shaders on GPU
    mps_gpu,
    /// CoreML on Neural Engine
    coreml_ne,
    /// No acceleration available — fall back to scalar
    none,

    pub fn label(self: AcceleratorBackend) []const u8 {
        return switch (self) {
            .accelerate_cpu => "Accelerate (AMX)",
            .mps_gpu => "MPS (GPU)",
            .coreml_ne => "CoreML (Neural Engine)",
            .none => "none",
        };
    }
};

/// Classification of operations for routing decisions.
pub const OperationClass = enum {
    elementwise_small,
    elementwise_large,
    matmul_small,
    matmul_large,
    activation,
    attention,
    embedding_lookup,
    full_model,
};

/// Configuration for the macOS accelerator routing thresholds.
pub const AcceleratorConfig = struct {
    /// Matrix dimension threshold: matmul with M*N*K above this uses GPU.
    matmul_gpu_threshold: usize = 4096,
    /// Element count threshold: elementwise ops above this use GPU.
    elementwise_gpu_threshold: usize = 1024,
    /// Prefer Neural Engine for full-model inference when available.
    prefer_neural_engine: bool = true,
    /// Force a specific backend (overrides routing).
    force_backend: ?AcceleratorBackend = null,
};

/// Unified macOS acceleration pipeline.
/// Routes operations across Accelerate, MPS, and CoreML based on size and type.
pub const MacOSAccelerator = struct {
    config: AcceleratorConfig,
    feature_set: gpu_family.MetalFeatureSet,
    has_accelerate: bool,
    has_mps: bool,
    has_coreml: bool,

    /// Opaque Metal device handle (MTLDevice ID). Set by initMetal().
    metal_device: ?*anyopaque = null,
    /// Opaque Metal command queue handle. Set by initMetal().
    command_queue: ?*anyopaque = null,
    /// Whether MPS runtime is initialized and ready for dispatch.
    mps_ready: bool = false,

    /// Total operations dispatched per backend (for profiling).
    stats_accelerate: u64 = 0,
    stats_mps: u64 = 0,
    stats_coreml: u64 = 0,
    stats_none: u64 = 0,

    pub fn init(config: AcceleratorConfig) MacOSAccelerator {
        const is_macos = builtin.os.tag == .macos;
        const features = if (is_macos)
            gpu_family.buildFeatureSet(.unknown) // Will be updated on real device detection
        else
            gpu_family.MetalFeatureSet{};

        return .{
            .config = config,
            .feature_set = features,
            .has_accelerate = is_macos and accelerate.is_available,
            .has_mps = is_macos and features.supports_mps,
            .has_coreml = is_macos,
        };
    }

    /// Initialize with a known GPU family (from device detection).
    pub fn initWithFamily(config: AcceleratorConfig, family: gpu_family.MetalGpuFamily) MacOSAccelerator {
        const features = gpu_family.buildFeatureSet(family);
        return .{
            .config = config,
            .feature_set = features,
            .has_accelerate = accelerate.is_available,
            .has_mps = features.supports_mps,
            .has_coreml = builtin.os.tag == .macos,
        };
    }

    /// Initialize Metal device and command queue for MPS dispatch.
    /// Call this before using matmul() with MPS routing.
    /// On non-macOS platforms, returns error.PlatformNotSupported.
    pub fn initMetal(self: *MacOSAccelerator) !void {
        if (comptime builtin.os.tag != .macos) return error.PlatformNotSupported;

        // MTLCreateSystemDefaultDevice() returns an MTLDevice ID
        const create_device = @extern(*const fn () callconv(.C) ?*anyopaque, .{
            .name = "MTLCreateSystemDefaultDevice",
            .library_name = "Metal",
        });
        self.metal_device = create_device() orelse return error.MetalDeviceNotFound;

        // Create command queue: [device newCommandQueue]
        const sel_new_queue = mps.sel_register("newCommandQueue");
        const msg_send: *const fn (*anyopaque, *anyopaque) callconv(.C) ?*anyopaque = @ptrCast(&mps.objc_msgSend);
        self.command_queue = msg_send(self.metal_device.?, sel_new_queue) orelse return error.CommandQueueCreationFailed;

        // Initialize MPS with the device
        mps.init(self.metal_device.?, null, null) catch |err| {
            std.log.warn("MPS init failed (GPU dispatch unavailable): {}", .{err});
        };
        self.mps_ready = true;
    }

    /// Release Metal device and command queue.
    pub fn deinitMetal(self: *MacOSAccelerator) void {
        if (comptime builtin.os.tag != .macos) return;

        if (self.command_queue) |queue| {
            const sel_release = mps.sel_register("release");
            const msg_send: *const fn (*anyopaque, *anyopaque) callconv(.C) void = @ptrCast(&mps.objc_msgSend);
            msg_send(queue, sel_release);
            self.command_queue = null;
        }
        if (self.metal_device) |device| {
            const sel_release = mps.sel_register("release");
            const msg_send: *const fn (*anyopaque, *anyopaque) callconv(.C) void = @ptrCast(&mps.objc_msgSend);
            msg_send(device, sel_release);
            self.metal_device = null;
        }
        self.mps_ready = false;
    }

    /// Core routing: select the best backend for a given operation and size.
    pub fn selectBackend(self: *const MacOSAccelerator, op: OperationClass) AcceleratorBackend {
        // Forced backend override
        if (self.config.force_backend) |forced| return forced;

        return switch (op) {
            // Small/CPU-friendly ops → Accelerate (AMX)
            .elementwise_small, .matmul_small, .activation, .embedding_lookup => {
                if (self.has_accelerate) return .accelerate_cpu;
                return .none;
            },
            // Large tensor ops → MPS (GPU)
            .elementwise_large, .matmul_large, .attention => {
                if (self.has_mps) return .mps_gpu;
                if (self.has_accelerate) return .accelerate_cpu;
                return .none;
            },
            // Full model inference → CoreML (Neural Engine)
            .full_model => {
                if (self.config.prefer_neural_engine and self.has_coreml and self.feature_set.has_neural_engine) {
                    return .coreml_ne;
                }
                if (self.has_mps) return .mps_gpu;
                if (self.has_accelerate) return .accelerate_cpu;
                return .none;
            },
        };
    }

    /// Classify a matmul operation based on dimensions.
    pub fn classifyMatmul(self: *const MacOSAccelerator, m: usize, n: usize, k: usize) OperationClass {
        const total = m * n * k;
        if (total >= self.config.matmul_gpu_threshold) return .matmul_large;
        return .matmul_small;
    }

    /// Classify an elementwise operation based on element count.
    pub fn classifyElementwise(self: *const MacOSAccelerator, num_elements: usize) OperationClass {
        if (num_elements >= self.config.elementwise_gpu_threshold) return .elementwise_large;
        return .elementwise_small;
    }

    /// High-level matrix multiply: routes to Accelerate or MPS based on size.
    pub fn matmul(
        self: *MacOSAccelerator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        m: u32,
        n: u32,
        k: u32,
    ) !AcceleratorBackend {
        const op = self.classifyMatmul(m, n, k);
        const backend = self.selectBackend(op);
        self.recordStat(backend);

        switch (backend) {
            .accelerate_cpu => {
                if (comptime accelerate.is_available) {
                    accelerate.sgemm(
                        .no_trans,
                        .no_trans,
                        @intCast(m),
                        @intCast(n),
                        @intCast(k),
                        1.0,
                        a.ptr,
                        @intCast(k),
                        b.ptr,
                        @intCast(n),
                        0.0,
                        @constCast(result.ptr),
                        @intCast(n),
                    );
                    return .accelerate_cpu;
                }
                return .none;
            },
            .mps_gpu => {
                // Try MPS GPU dispatch if Metal runtime is initialized.
                if (self.mps_ready and self.metal_device != null and self.command_queue != null) {
                    self.executeMpsMatmul(a, b, result, m, n, k) catch |err| {
                        // MPS dispatch failed — fall through to Accelerate CPU
                        std.log.warn("MPS matmul dispatch failed, falling back to Accelerate CPU: {}", .{err});
                    };
                    // If we get here without error, MPS handled it via shared memory.
                    // The result buffer is already populated (unified memory).
                }
                // Accelerate CPU fallback (also used if MPS dispatch fails)
                if (comptime accelerate.is_available) {
                    accelerate.sgemm(
                        .no_trans,
                        .no_trans,
                        @intCast(m),
                        @intCast(n),
                        @intCast(k),
                        1.0,
                        a.ptr,
                        @intCast(k),
                        b.ptr,
                        @intCast(n),
                        0.0,
                        @constCast(result.ptr),
                        @intCast(n),
                    );
                    return .accelerate_cpu;
                }
                return .none;
            },
            .coreml_ne, .none => return .none,
        }
    }

    /// High-level softmax: uses vDSP maxv + vForce exp + vDSP sum when Accelerate available.
    pub fn softmax(self: *MacOSAccelerator, data: []f32) AcceleratorBackend {
        const backend = self.selectBackend(.activation);
        self.recordStat(backend);
        const n = data.len;
        if (n == 0) return backend;

        if (comptime accelerate.is_available) {
            // Step 1: Find max (scalar — no vDSP_maxv wrapper yet)
            var max_val: f32 = -std.math.inf(f32);
            for (data) |v| {
                if (v > max_val) max_val = v;
            }

            // Step 2: Subtract max (numerical stability) via vDSP_vsadd
            accelerate.vsadd(data, -max_val, data) catch |err| {
                std.log.warn("accelerate vsadd failed in softmax, falling back to scalar: {}", .{err});
                for (data) |*v| v.* -= max_val;
            };

            // Step 3: Exponentiate via vForce vvexpf
            accelerate.vexp(data, data) catch |err| {
                std.log.warn("accelerate vexp failed in softmax, falling back to scalar: {}", .{err});
                for (data) |*v| v.* = @exp(v.*);
            };

            // Step 4: Sum (scalar — no vDSP_sve wrapper yet)
            var total: f32 = 0;
            for (data) |v| total += v;

            // Step 5: Normalize via vDSP_vsmul
            if (total > 0) {
                accelerate.vsmul(data, 1.0 / total, data) catch |err| {
                    std.log.warn("accelerate vsmul failed in softmax, falling back to scalar: {}", .{err});
                    const inv_total = 1.0 / total;
                    for (data) |*v| v.* *= inv_total;
                };
            }
        } else {
            // Scalar fallback
            var max_val: f32 = -std.math.inf(f32);
            for (data) |v| {
                if (v > max_val) max_val = v;
            }
            var sum: f32 = 0;
            for (data) |*v| {
                v.* = @exp(v.* - max_val);
                sum += v.*;
            }
            if (sum > 0) {
                const inv_sum = 1.0 / sum;
                for (data) |*v| {
                    v.* *= inv_sum;
                }
            }
        }
        return backend;
    }

    /// High-level RMSNorm: uses vDSP for dot product and scaling when Accelerate available.
    pub fn rmsnorm(self: *MacOSAccelerator, output: []f32, input: []const f32, weight: []const f32, eps: f32) AcceleratorBackend {
        const op = self.classifyElementwise(input.len);
        const backend = self.selectBackend(op);
        self.recordStat(backend);
        const len = @min(input.len, @min(output.len, weight.len));
        if (len == 0) return backend;

        if (comptime accelerate.is_available) {
            // Step 1: Sum of squares (scalar — no dotpr wrapper yet)
            var sum_sq: f32 = 0;
            for (input[0..len]) |v| sum_sq += v * v;

            // Step 2: Compute RMS scale
            const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(len)) + eps);

            // Step 3: input * weight → output via vDSP_vmul (slice-based)
            accelerate.vmul(input[0..len], weight[0..len], output[0..len]) catch |err| {
                std.log.warn("accelerate vmul failed in rmsnorm, falling back to scalar: {}", .{err});
                for (0..len) |i| output[i] = input[i] * weight[i];
            };

            // Step 4: Scale by RMS via vDSP_vsmul (slice-based)
            accelerate.vsmul(output[0..len], rms, output[0..len]) catch |err| {
                std.log.warn("accelerate vsmul failed in rmsnorm, falling back to scalar: {}", .{err});
                for (0..len) |i| output[i] *= rms;
            };
        } else {
            // Scalar fallback
            var sum_sq: f32 = 0;
            for (input) |v| {
                sum_sq += v * v;
            }
            const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(input.len)) + eps);
            for (0..len) |i| {
                output[i] = input[i] * rms * weight[i];
            }
        }
        return backend;
    }

    /// High-level SiLU activation: x * sigmoid(x). Uses vForce for exp when available.
    pub fn silu(self: *MacOSAccelerator, data: []f32) AcceleratorBackend {
        const backend = self.selectBackend(.activation);
        self.recordStat(backend);
        const n = data.len;
        if (n == 0) return backend;

        if (comptime accelerate.is_available) {
            // SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
            var tmp: [4096]f32 = undefined;
            const chunk = @min(n, tmp.len);
            var offset: usize = 0;
            while (offset < n) {
                const remaining = @min(chunk, n - offset);
                const src = data[offset..][0..remaining];
                const dst = tmp[0..remaining];
                // Step 1: Negate → tmp = -x
                accelerate.vneg(src, dst) catch {
                    // Fallback to scalar if vneg fails
                    for (0..remaining) |i| {
                        const sigmoid = 1.0 / (1.0 + @exp(-data[offset + i]));
                        data[offset + i] = data[offset + i] * sigmoid;
                    }
                    offset += remaining;
                    continue;
                };
                // Step 2: exp(-x) via vForce
                accelerate.vexp(dst, dst) catch |err| {
                    std.log.warn("accelerate vexp failed in silu, falling back to scalar: {}", .{err});
                    for (dst) |*v| v.* = @exp(v.*);
                };
                // Step 3: x * sigmoid(x)
                for (0..remaining) |i| {
                    const sigmoid = 1.0 / (1.0 + tmp[i]);
                    data[offset + i] = data[offset + i] * sigmoid;
                }
                offset += remaining;
            }
        } else {
            for (data) |*v| {
                const sigmoid = 1.0 / (1.0 + @exp(-v.*));
                v.* = v.* * sigmoid;
            }
        }
        return backend;
    }

    /// High-level GeLU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// Uses vForce tanh when available.
    pub fn gelu(self: *MacOSAccelerator, data: []f32) AcceleratorBackend {
        const backend = self.selectBackend(.activation);
        self.recordStat(backend);
        const n = data.len;
        if (n == 0) return backend;

        const sqrt_2_over_pi: f32 = 0.7978845608;

        if (comptime accelerate.is_available) {
            var tmp: [4096]f32 = undefined;
            const chunk = @min(n, tmp.len);
            var offset: usize = 0;
            while (offset < n) {
                const remaining = @min(chunk, n - offset);
                const dst = tmp[0..remaining];
                // Compute inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                for (0..remaining) |i| {
                    const x = data[offset + i];
                    dst[i] = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                }
                // tanh via vForce vvtanhf (slice-based)
                accelerate.vtanh(dst, dst) catch |err| {
                    std.log.warn("accelerate vtanh failed in gelu, falling back to scalar: {}", .{err});
                    for (dst) |*v| v.* = std.math.tanh(v.*);
                };
                // result = 0.5 * x * (1 + tanh(inner))
                for (0..remaining) |i| {
                    data[offset + i] = 0.5 * data[offset + i] * (1.0 + tmp[i]);
                }
                offset += remaining;
            }
        } else {
            for (data) |*v| {
                const x = v.*;
                const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                v.* = 0.5 * x * (1.0 + std.math.tanh(inner));
            }
        }
        return backend;
    }

    /// Execute matrix multiply on MPS GPU via Metal command buffer.
    /// Uses shared memory (Apple Silicon unified architecture) so the result
    /// is visible to CPU after command buffer completion.
    fn executeMpsMatmul(
        self: *MacOSAccelerator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        m: u32,
        n: u32,
        k: u32,
    ) !void {
        if (comptime builtin.os.tag != .macos) return error.PlatformNotSupported;

        const device = self.metal_device orelse return error.MetalDeviceNotFound;
        const queue = self.command_queue orelse return error.CommandQueueCreationFailed;

        // Create MPS matrix multiply kernel
        var mps_config = mps.MpsMatMul.Config{};
        mps_config.m = m;
        mps_config.n = n;
        mps_config.k = k;
        mps_config.alpha = 1.0;
        mps_config.beta = 0.0;
        var matmul_kernel = mps.MpsMatMul.create(device, mps_config) orelse return error.KernelCreationFailed;
        defer matmul_kernel.destroy();

        // Create command buffer from queue
        const sel_cmd_buf = mps.sel_register("commandBuffer");
        const msg_send_obj: *const fn (*anyopaque, *anyopaque) callconv(.C) ?*anyopaque = @ptrCast(&mps.objc_msgSend);
        const cmd_buf = msg_send_obj(queue, sel_cmd_buf) orelse return error.CommandBufferCreationFailed;

        // Encode the matmul operation
        matmul_kernel.encode(cmd_buf, a.ptr, b.ptr, @constCast(result.ptr)) catch return error.EncodeFailed;

        // Commit and wait
        const sel_commit = mps.sel_register("commit");
        const sel_wait = mps.sel_register("waitUntilCompleted");
        const msg_send_void: *const fn (*anyopaque, *anyopaque) callconv(.C) void = @ptrCast(&mps.objc_msgSend);
        msg_send_void(cmd_buf, sel_commit);
        msg_send_void(cmd_buf, sel_wait);
    }

    /// Load a CoreML model for full-model inference on Neural Engine.
    pub fn loadCoreMLModel(self: *const MacOSAccelerator, model_path: []const u8) !coreml.CoreMlModel {
        if (!self.has_coreml) return error.FrameworkNotAvailable;
        _ = model_path;
        // CoreML model loading requires Obj-C runtime initialization.
        // Returns error until coreml.init() has been called with runtime pointers.
        return error.FrameworkNotAvailable;
    }

    /// Run inference on a loaded CoreML model.
    pub fn runCoreMLInference(self: *MacOSAccelerator, model_obj: *coreml.CoreMlModel, input_data: []const f32) ![]f32 {
        if (!self.has_coreml) return error.FrameworkNotAvailable;
        self.recordStat(.coreml_ne);
        _ = model_obj;
        _ = input_data;
        return error.FrameworkNotAvailable;
    }

    /// Get dispatch statistics.
    pub fn getStats(self: *const MacOSAccelerator) struct {
        accelerate_ops: u64,
        mps_ops: u64,
        coreml_ops: u64,
        none_ops: u64,
        total_ops: u64,
    } {
        const total = self.stats_accelerate + self.stats_mps + self.stats_coreml + self.stats_none;
        return .{
            .accelerate_ops = self.stats_accelerate,
            .mps_ops = self.stats_mps,
            .coreml_ops = self.stats_coreml,
            .none_ops = self.stats_none,
            .total_ops = total,
        };
    }

    fn recordStat(self: *MacOSAccelerator, backend: AcceleratorBackend) void {
        switch (backend) {
            .accelerate_cpu => self.stats_accelerate += 1,
            .mps_gpu => self.stats_mps += 1,
            .coreml_ne => self.stats_coreml += 1,
            .none => self.stats_none += 1,
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "AcceleratorBackend labels" {
    try std.testing.expectEqualStrings("Accelerate (AMX)", AcceleratorBackend.accelerate_cpu.label());
    try std.testing.expectEqualStrings("MPS (GPU)", AcceleratorBackend.mps_gpu.label());
    try std.testing.expectEqualStrings("CoreML (Neural Engine)", AcceleratorBackend.coreml_ne.label());
    try std.testing.expectEqualStrings("none", AcceleratorBackend.none.label());
}

test "MacOSAccelerator init defaults" {
    const accel = MacOSAccelerator.init(.{});
    try std.testing.expectEqual(@as(usize, 4096), accel.config.matmul_gpu_threshold);
    try std.testing.expectEqual(@as(usize, 1024), accel.config.elementwise_gpu_threshold);
    try std.testing.expect(accel.config.prefer_neural_engine);
}

test "classifyMatmul threshold routing" {
    const accel = MacOSAccelerator.init(.{ .matmul_gpu_threshold = 4096 });
    // 8*8*8 = 512 < 4096 → small
    try std.testing.expectEqual(OperationClass.matmul_small, accel.classifyMatmul(8, 8, 8));
    // 32*32*32 = 32768 >= 4096 → large
    try std.testing.expectEqual(OperationClass.matmul_large, accel.classifyMatmul(32, 32, 32));
}

test "classifyElementwise threshold routing" {
    const accel = MacOSAccelerator.init(.{ .elementwise_gpu_threshold = 1024 });
    try std.testing.expectEqual(OperationClass.elementwise_small, accel.classifyElementwise(512));
    try std.testing.expectEqual(OperationClass.elementwise_large, accel.classifyElementwise(2048));
}

test "selectBackend forced override" {
    const accel = MacOSAccelerator.init(.{ .force_backend = .mps_gpu });
    try std.testing.expectEqual(AcceleratorBackend.mps_gpu, accel.selectBackend(.activation));
    try std.testing.expectEqual(AcceleratorBackend.mps_gpu, accel.selectBackend(.matmul_large));
}

test "softmax correctness" {
    var accel = MacOSAccelerator.init(.{});
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    _ = accel.softmax(&data);

    // Verify sum ~= 1.0
    var sum: f32 = 0;
    for (data) |v| sum += v;
    try std.testing.expect(@abs(sum - 1.0) < 1e-5);

    // Verify ordering preserved
    try std.testing.expect(data[2] > data[1]);
    try std.testing.expect(data[1] > data[0]);
}

test "softmax empty input" {
    var accel = MacOSAccelerator.init(.{});
    var data = [_]f32{};
    _ = accel.softmax(&data);
}

test "rmsnorm correctness" {
    var accel = MacOSAccelerator.init(.{});
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output: [4]f32 = undefined;
    _ = accel.rmsnorm(&output, &input, &weight, 1e-6);

    // With uniform weights, output should be input scaled by 1/rms
    // rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5)
    const expected_rms = 1.0 / @sqrt(7.5 + 1e-6);
    for (0..4) |i| {
        try std.testing.expect(@abs(output[i] - input[i] * expected_rms) < 1e-4);
    }
}

test "silu zero is zero" {
    var accel = MacOSAccelerator.init(.{});
    var data = [_]f32{ 0.0, 0.0 };
    _ = accel.silu(&data);
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try std.testing.expect(@abs(data[0]) < 1e-6);
    try std.testing.expect(@abs(data[1]) < 1e-6);
}

test "gelu symmetry" {
    var accel = MacOSAccelerator.init(.{});
    var pos = [_]f32{1.0};
    var neg = [_]f32{-1.0};
    _ = accel.gelu(&pos);
    _ = accel.gelu(&neg);
    // GeLU is NOT symmetric, but GeLU(x) + GeLU(-x) ≈ x for |x| small
    // GeLU(1.0) ≈ 0.8412, GeLU(-1.0) ≈ -0.1588
    try std.testing.expect(pos[0] > 0.8);
    try std.testing.expect(pos[0] < 0.9);
    try std.testing.expect(neg[0] > -0.2);
    try std.testing.expect(neg[0] < -0.1);
}

test "MPS fields initialized correctly" {
    const accel = MacOSAccelerator.init(.{});
    try std.testing.expect(accel.metal_device == null);
    try std.testing.expect(accel.command_queue == null);
    try std.testing.expect(!accel.mps_ready);
}

test "initMetal on non-macOS returns error" {
    if (comptime builtin.os.tag == .macos) return error.SkipZigTest;
    var accel = MacOSAccelerator.init(.{});
    try std.testing.expectError(error.PlatformNotSupported, accel.initMetal());
}

test "dispatch stats tracking" {
    var accel = MacOSAccelerator.init(.{});
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    _ = accel.softmax(&data);
    _ = accel.silu(&data);
    _ = accel.gelu(&data);
    const stats = accel.getStats();
    // Each activation call records one stat
    try std.testing.expectEqual(@as(u64, 3), stats.total_ops);
}

test {
    std.testing.refAllDecls(@This());
}
