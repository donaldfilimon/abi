//! Parent GPU compute API — one backend-agnostic facade over the compute
//! backends, with an HONEST per-backend availability report.
//!
//! It exposes a single dispatch surface for elementwise binary kernels
//! (`map`) and their reduce-sum (`reduce`, i.e. dot / squared-L2), routing to
//! the best backend that actually executes here, and falling back to a
//! vectorized CPU path. `backendMatrix()` reports, per backend, whether it is
//! available and whether it genuinely dispatches on this build — so callers
//! never mistake a declared backend for a working one.
//!
//! Backend reality (per docs/contracts/external-claims-audit.mdx — no faking):
//!   - **Metal** (macOS): REAL. Dispatches MSL compute kernels on the GPU via
//!     the pure-Zig Objective-C runtime FFI in `metal_shared.zig` (runtime MSL
//!     compile + MTLComputePipelineState + dispatchThreads). Targets Metal 3/4
//!     class devices.
//!   - **Simulated**: REAL vectorized CPU fallback; always available.
//!   - **Vulkan / OpenGL**: DECLARED but NOT dispatching here. Vulkan (1.x/1.4)
//!     needs a loader + ICD (e.g. MoltenVK on macOS) and SPIR-V shaders; macOS
//!     OpenGL caps at 4.1 with no compute shaders. The intended real path is
//!     Zig-compiled SPIR-V (`-ofmt=spirv`) fed to the loader — gated until a
//!     loader is present and Zig's SPIR-V backend is production-ready. These
//!     report `available=false` with a reason rather than simulating execution.
//!   - **CUDA / WebGPU**: declared, not dispatching here (no toolchain/runtime).

const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");
const metal = @import("metal_shared.zig");

pub const Backend = backends.Backend;

/// Elementwise binary kernels. `map` writes per-element results; `reduce` sums
/// them (so `.mul` reduce = dot product, `.l2diff` reduce = squared L2,
/// `.add` / `.sub` reduce = sum of pairwise sums / differences).
pub const Kernel = enum {
    /// out[i] = a[i] * b[i]
    mul,
    /// out[i] = (a[i] - b[i])^2
    l2diff,
    /// out[i] = a[i] + b[i]
    add,
    /// out[i] = a[i] - b[i]
    sub,
};

/// Honest status of one backend on this build.
pub const BackendAvailability = struct {
    backend: Backend,
    /// The backend's runtime/driver is usable here.
    available: bool,
    /// It genuinely executes compute on this build (not simulated/CPU-mapped).
    dispatches: bool,
    /// Human-readable reason, especially when not dispatching.
    reason: []const u8,
};

/// Per-backend availability matrix — the truthful answer to "what actually runs
/// here". Order matches `backendCapabilitiesList()` for consistent reporting.
pub fn backendMatrix() [7]BackendAvailability {
    const caps = backends.backendCapabilitiesList();
    var matrix: [7]BackendAvailability = undefined;
    for (caps, 0..) |cap, i| {
        const dispatches = switch (cap.backend) {
            .simulated => true,
            .metal => cap.native_kernels,
            else => false,
        };
        matrix[i] = .{
            .backend = cap.backend,
            .available = cap.available,
            .dispatches = dispatches,
            .reason = cap.message,
        };
    }
    return matrix;
}

/// Backend-agnostic compute handle. `init` selects the best backend that truly
/// dispatches here (Metal when live, else CPU). All `map`/`reduce` calls go
/// through the selected backend with a deterministic CPU fallback.
pub const GpuCompute = struct {
    backend: Backend,
    accelerated: bool,

    pub fn init() GpuCompute {
        if (builtin.target.os.tag == .macos) {
            metal.g_metal_context.init(std.heap.page_allocator) catch |err| {
                std.log.warn("GpuCompute: Metal init failed: {s}; using CPU", .{@errorName(err)});
            };
            if (metal.g_metal_context.initialized) return .{ .backend = .metal, .accelerated = true };
        }
        return .{ .backend = .simulated, .accelerated = false };
    }

    pub fn backendName(self: GpuCompute) []const u8 {
        return backends.backendName(self.backend);
    }

    pub fn isAccelerated(self: GpuCompute) bool {
        return self.accelerated;
    }

    fn metalPipeline(self: GpuCompute, kernel: Kernel) ?*anyopaque {
        if (!self.accelerated) return null;
        return switch (kernel) {
            .mul => metal.g_metal_context.dot_pipeline,
            .l2diff => metal.g_metal_context.l2_pipeline,
            .add => metal.g_metal_context.add_pipeline,
            .sub => metal.g_metal_context.sub_pipeline,
        };
    }

    fn cpuMap(kernel: Kernel, a: []const f32, b: []const f32, out: []f32) void {
        switch (kernel) {
            .mul => for (a, b, out) |x, y, *o| {
                o.* = x * y;
            },
            .l2diff => for (a, b, out) |x, y, *o| {
                const d = x - y;
                o.* = d * d;
            },
            .add => for (a, b, out) |x, y, *o| {
                o.* = x + y;
            },
            .sub => for (a, b, out) |x, y, *o| {
                o.* = x - y;
            },
        }
    }

    /// Elementwise `kernel(a, b)` into `out` (all three same length). Dispatches
    /// on the GPU when accelerated, else the CPU path.
    pub fn map(self: GpuCompute, kernel: Kernel, a: []const f32, b: []const f32, out: []f32) !void {
        if (a.len != b.len or a.len != out.len) return error.DimensionMismatch;
        if (a.len == 0) return;
        if (self.metalPipeline(kernel)) |pipeline| {
            metal.g_metal_context.runKernel(pipeline, a.len, a, b, out) catch {
                // GPU dispatch failed mid-run — degrade to the CPU path rather
                // than surfacing a transient device error to the caller.
                cpuMap(kernel, a, b, out);
            };
            return;
        }
        cpuMap(kernel, a, b, out);
    }

    /// Reduce-sum of the elementwise kernel: `.mul` -> dot(a,b),
    /// `.l2diff` -> squared L2. Uses the GPU `map` then sums, with a fully
    /// vectorized CPU fast path when not accelerated.
    pub fn reduce(self: GpuCompute, kernel: Kernel, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.metalPipeline(kernel)) |pipeline| {
            var stack: [4096]f32 = undefined;
            const out = if (a.len <= stack.len)
                stack[0..a.len]
            else
                try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > stack.len) std.heap.page_allocator.free(out);
            metal.g_metal_context.runKernel(pipeline, a.len, a, b, out) catch {
                return cpuReduce(kernel, a, b);
            };
            var sum: f32 = 0;
            for (out) |v| sum += v;
            return sum;
        }
        return cpuReduce(kernel, a, b);
    }

    fn cpuReduce(kernel: Kernel, a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0;
        var i: usize = 0;
        switch (kernel) {
            .mul => {
                while (i + 4 <= a.len) : (i += 4) {
                    const av: @Vector(4, f32) = a[i..][0..4].*;
                    const bv: @Vector(4, f32) = b[i..][0..4].*;
                    sum += @reduce(.Add, av * bv);
                }
                while (i < a.len) : (i += 1) sum += a[i] * b[i];
            },
            .l2diff => {
                while (i + 4 <= a.len) : (i += 4) {
                    const av: @Vector(4, f32) = a[i..][0..4].*;
                    const bv: @Vector(4, f32) = b[i..][0..4].*;
                    const d = av - bv;
                    sum += @reduce(.Add, d * d);
                }
                while (i < a.len) : (i += 1) {
                    const d = a[i] - b[i];
                    sum += d * d;
                }
            },
            .add => {
                while (i + 4 <= a.len) : (i += 4) {
                    const av: @Vector(4, f32) = a[i..][0..4].*;
                    const bv: @Vector(4, f32) = b[i..][0..4].*;
                    sum += @reduce(.Add, av + bv);
                }
                while (i < a.len) : (i += 1) sum += a[i] + b[i];
            },
            .sub => {
                while (i + 4 <= a.len) : (i += 4) {
                    const av: @Vector(4, f32) = a[i..][0..4].*;
                    const bv: @Vector(4, f32) = b[i..][0..4].*;
                    sum += @reduce(.Add, av - bv);
                }
                while (i < a.len) : (i += 1) sum += a[i] - b[i];
            },
        }
        return sum;
    }
};

const testing = std.testing;

test "compute_api: backend matrix is honest (cpu fallback always dispatches; stubs gated)" {
    const m = backendMatrix();
    try testing.expectEqual(@as(usize, 7), m.len);
    var saw_cpu_fallback = false;
    var saw_metal = false;
    var saw_webgl2 = false;
    for (m) |entry| {
        switch (entry.backend) {
            .simulated => {
                saw_cpu_fallback = true;
                try testing.expect(entry.available and entry.dispatches);
            },
            .metal => {
                saw_metal = true;
                // Metal only "dispatches" when actually initialized on macOS.
                if (builtin.target.os.tag != .macos) {
                    try testing.expect(!entry.available);
                    try testing.expect(!entry.dispatches);
                }
            },
            .webgl2 => {
                saw_webgl2 = true;
                try testing.expect(!entry.available);
                try testing.expect(!entry.dispatches);
                try testing.expect(entry.reason.len > 0);
            },
            .vulkan, .opengl, .cuda, .webgpu => {
                // Declared but not dispatching here — must not claim execution.
                try testing.expect(!entry.available);
                try testing.expect(!entry.dispatches);
                try testing.expect(entry.reason.len > 0);
            },
        }
    }
    try testing.expect(saw_cpu_fallback and saw_metal and saw_webgl2);
}

test "compute_api: reduce matches scalar reference on the active backend" {
    const gc = GpuCompute.init();
    // Whatever backend init() selected (Metal on macOS when live, else CPU) must
    // agree with an independent scalar reference — the CPU/GPU parity contract.
    const a = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    const b = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };
    var ref_dot: f32 = 0;
    var ref_l2: f32 = 0;
    var ref_add: f32 = 0;
    var ref_sub: f32 = 0;
    for (a, b) |x, y| {
        ref_dot += x * y;
        ref_l2 += (x - y) * (x - y);
        ref_add += x + y;
        ref_sub += x - y;
    }
    try testing.expectApproxEqAbs(ref_dot, try gc.reduce(.mul, &a, &b), 1e-3);
    try testing.expectApproxEqAbs(ref_l2, try gc.reduce(.l2diff, &a, &b), 1e-3);
    try testing.expectApproxEqAbs(ref_add, try gc.reduce(.add, &a, &b), 1e-3);
    try testing.expectApproxEqAbs(ref_sub, try gc.reduce(.sub, &a, &b), 1e-3);
}

test "compute_api: map writes correct elementwise results and checks dims" {
    const gc = GpuCompute.init();
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [4]f32 = undefined;
    try gc.map(.mul, &a, &b, &out);
    try testing.expectApproxEqAbs(@as(f32, 5), out[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 32), out[3], 1e-4);
    try gc.map(.l2diff, &a, &b, &out);
    try testing.expectApproxEqAbs(@as(f32, 16), out[0], 1e-4); // (1-5)^2
    try gc.map(.add, &a, &b, &out);
    try testing.expectApproxEqAbs(@as(f32, 6), out[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 12), out[3], 1e-4);
    try gc.map(.sub, &a, &b, &out);
    try testing.expectApproxEqAbs(@as(f32, -4), out[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -4), out[3], 1e-4);

    var bad: [3]f32 = undefined;
    try testing.expectError(error.DimensionMismatch, gc.map(.mul, &a, &b, &bad));
}

test "compute_api: empty input is a no-op / zero reduce" {
    const gc = GpuCompute.init();
    var out: [0]f32 = undefined;
    try gc.map(.mul, &.{}, &.{}, &out);
    try testing.expectEqual(@as(f32, 0), try gc.reduce(.mul, &.{}, &.{}));
}

test {
    testing.refAllDecls(@This());
}
