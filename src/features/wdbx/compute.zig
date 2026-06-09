//! Dynamic compute backend selection (Compute Layer).
//!
//! Enumerates the CPU / GPU / NPU / TPU backends from the north-star spec and
//! selects one dynamically from the host target, always with a deterministic
//! CPU fallback. Native GPU/NPU/TPU dispatch is not linked in this build, so
//! those backends report `native=false` and execution runs the portable SIMD
//! CPU path — exactly how the existing GPU layer behaves. This wires the
//! *selection + capability reporting* honestly; it does not claim native
//! accelerator kernels that are not present.

const std = @import("std");
const builtin = @import("builtin");

pub const Backend = enum {
    cpu_scalar,
    cpu_avx2,
    cpu_avx512,
    cpu_neon,
    gpu_cuda,
    gpu_metal,
    gpu_vulkan,
    npu_ane,
    tpu_remote,

    pub fn name(self: Backend) []const u8 {
        return switch (self) {
            .cpu_scalar => "cpu-scalar",
            .cpu_avx2 => "cpu-avx2",
            .cpu_avx512 => "cpu-avx512",
            .cpu_neon => "cpu-neon",
            .gpu_cuda => "gpu-cuda",
            .gpu_metal => "gpu-metal",
            .gpu_vulkan => "gpu-vulkan",
            .npu_ane => "npu-ane",
            .tpu_remote => "tpu-remote",
        };
    }

    pub fn class(self: Backend) []const u8 {
        return switch (self) {
            .cpu_scalar, .cpu_avx2, .cpu_avx512, .cpu_neon => "CPU",
            .gpu_cuda, .gpu_metal, .gpu_vulkan => "GPU",
            .npu_ane => "NPU",
            .tpu_remote => "TPU",
        };
    }
};

pub const Capability = struct {
    backend: Backend,
    /// True only when native dispatch for this backend is linked. All
    /// accelerator backends are false in this build (CPU fallback active).
    native: bool,
    /// True when the backend is usable at all (CPU paths, or accelerators via
    /// their CPU fallback).
    available: bool,
};

/// The best CPU SIMD path the host target supports (compile-time feature
/// detection), used as the deterministic fallback for every backend.
pub fn bestCpuBackend() Backend {
    return switch (builtin.cpu.arch) {
        .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
            .cpu_avx512
        else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
            .cpu_avx2
        else
            .cpu_scalar,
        .aarch64, .aarch64_be => if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon))
            .cpu_neon
        else
            .cpu_scalar,
        else => .cpu_scalar,
    };
}

/// Whether this build targets an Apple-Silicon macOS host, where the Apple
/// Neural Engine is physically present. IMPORTANT: presence is NOT execution —
/// no CoreML/ANE dispatch path is linked (that requires Apple ObjC frameworks,
/// not pure Zig, and cannot be verified to run on the ANE vs CPU/GPU without
/// on-device profiling), so ANE-selected workloads use the CPU fallback. This
/// reports hardware availability honestly; it is not a claim of ANE execution.
pub fn aneHardwarePresent() bool {
    return builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos;
}

fn isAccelerator(b: Backend) bool {
    return switch (b) {
        .gpu_cuda, .gpu_metal, .gpu_vulkan, .npu_ane, .tpu_remote => true,
        else => false,
    };
}

pub fn capabilities() [9]Capability {
    var caps: [9]Capability = undefined;
    const cpu_best = bestCpuBackend();
    inline for (std.enums.values(Backend), 0..) |b, i| {
        const accel = isAccelerator(b);
        // CPU backends are available when the host actually supports that level;
        // accelerators are "available" via CPU fallback but never native here.
        const cpu_available = switch (b) {
            .cpu_scalar => true,
            .cpu_avx2 => cpu_best == .cpu_avx2 or cpu_best == .cpu_avx512,
            .cpu_avx512 => cpu_best == .cpu_avx512,
            .cpu_neon => cpu_best == .cpu_neon,
            else => false,
        };
        caps[i] = .{
            .backend = b,
            .native = false,
            .available = if (accel) true else cpu_available,
        };
    }
    return caps;
}

pub const Selection = struct {
    requested: Backend,
    effective: Backend,
    native: bool,
    message: []const u8,
};

/// Dynamically select a backend. If the requested backend has no native
/// dispatch (every accelerator in this build), fall back to the best CPU SIMD
/// path deterministically rather than failing.
pub fn select(requested: Backend) Selection {
    if (!isAccelerator(requested)) {
        return .{ .requested = requested, .effective = requested, .native = false, .message = "CPU SIMD path active" };
    }
    return .{
        .requested = requested,
        .effective = bestCpuBackend(),
        .native = false,
        .message = "native accelerator dispatch not linked; deterministic CPU fallback active",
    };
}

/// Host-optimal SIMD lane count for f32 (8 on AVX2, 16 on AVX512, 4 on NEON),
/// falling back to 4 where the compiler offers no suggestion. Matches the width
/// selection used by the HNSW distance kernel rather than a hardcoded vector.
pub const SIMD_LANES: usize = std.simd.suggestVectorLength(f32) orelse 4;

/// Execute a dot product on the selected backend using the host-matched SIMD
/// width. The result is identical across backends (the accelerators fall back to
/// the same CPU SIMD kernel), so CPU/GPU parity holds by construction.
pub fn dot(sel: Selection, a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.DimensionMismatch;
    _ = sel;
    const lanes = SIMD_LANES;
    const Vec = @Vector(lanes, f32);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const va: Vec = a[i..][0..lanes].*;
        const vb: Vec = b[i..][0..lanes].*;
        acc += va * vb;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

test "compute: selection falls back to a CPU SIMD path for accelerators" {
    const ane = select(.npu_ane);
    try std.testing.expect(!ane.native);
    try std.testing.expectEqual(ane.effective.class().len, "CPU".len);
    try std.testing.expectEqualStrings("CPU", ane.effective.class());

    const tpu = select(.tpu_remote);
    try std.testing.expectEqualStrings("CPU", tpu.effective.class());

    const cpu = select(.cpu_scalar);
    try std.testing.expectEqual(Backend.cpu_scalar, cpu.effective);
}

test "compute: ANE hardware detection matches the build target" {
    const expected = builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos;
    try std.testing.expectEqual(expected, aneHardwarePresent());
}

test "compute: capability table enumerates all backend classes" {
    const caps = capabilities();
    try std.testing.expectEqual(@as(usize, 9), caps.len);
    var has_npu = false;
    var has_tpu = false;
    for (caps) |c| {
        try std.testing.expect(!c.native); // nothing native in this build
        if (c.backend == .npu_ane) has_npu = true;
        if (c.backend == .tpu_remote) has_tpu = true;
    }
    try std.testing.expect(has_npu and has_tpu);
}

test "compute: CPU/GPU parity — dot product matches across backends" {
    const a = [_]f32{ 1, 2, 3, 4, 5 };
    const b = [_]f32{ 5, 4, 3, 2, 1 };
    const expected: f32 = 1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1; // 35
    const cpu = try dot(select(.cpu_scalar), &a, &b);
    const gpu = try dot(select(.gpu_metal), &a, &b);
    const npu = try dot(select(.npu_ane), &a, &b);
    try std.testing.expectApproxEqAbs(expected, cpu, 1e-5);
    try std.testing.expectApproxEqAbs(cpu, gpu, 1e-6);
    try std.testing.expectApproxEqAbs(cpu, npu, 1e-6);
}

test "compute: dot is correct across host SIMD width with a ragged tail" {
    // Length 17 is not a multiple of 4/8/16, so the scalar tail runs under any
    // host vector width — guards the width-adaptive loop + remainder handling.
    var a: [17]f32 = undefined;
    var b: [17]f32 = undefined;
    var expected: f32 = 0;
    for (&a, &b, 0..) |*ai, *bi, i| {
        ai.* = @floatFromInt(i + 1);
        bi.* = 2.0;
        expected += ai.* * bi.*;
    }
    try std.testing.expect(SIMD_LANES >= 4);
    try std.testing.expectApproxEqAbs(expected, try dot(select(.cpu_scalar), &a, &b), 1e-4);
}

test {
    std.testing.refAllDecls(@This());
}
