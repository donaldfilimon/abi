//! ARM Thumb Bare Metal Deployment Example
//!
//! Demonstrates fully stack-based (zero heap allocation) usage of ABI
//! framework types, targeting ARM Thumb embedded cores.
//!
//! Key embedded patterns demonstrated:
//!   - All data lives on the stack — no allocator needed
//!   - Fixed-size tensor operations on stack buffers
//!   - SIMD vector arithmetic (dot product, element-wise add)
//!   - Math primitives (clamp, bounded arithmetic)
//!   - Compile-time feature and version queries
//!
//! Thumb deployment considerations:
//!   - Thumb cores are often Cortex-M class with 16-256 KB RAM.
//!     Keep stack buffers small and prefer in-place operations.
//!   - The Zig compiler emits Thumb-2 instructions when targeting
//!     `thumb-freestanding-none`.  SIMD intrinsics fall back to
//!     scalar loops on cores without NEON/MVE.
//!   - On real hardware, replace `std.debug.print` with UART or
//!     semihosting output.
//!
//! Run with: `zig build run-bare-metal-thumb`

const std = @import("std");
const abi = @import("abi");

const Tensor32 = abi.foundation.tensor.Tensor32;
const Shape = abi.foundation.tensor.Shape;
const primitives = abi.foundation.utils.primitives;

pub fn main(_: std.process.Init) !void {
    std.debug.print("\n=== ABI Bare-Metal ARM Thumb Demo ===\n", .{});
    std.debug.print("ABI version: {s}\n", .{abi.version()});

    // ── 1. Stack-only tensor pipeline ───────────────────────────────────
    // All buffers are stack-allocated.  On a Cortex-M4 with 64 KB RAM the
    // default 8 KB stack is more than enough for small inference layers.

    // Simulated 1x4 activation vector and weight/bias vectors
    var activations = [_]f32{ 0.8, -0.3, 1.2, -0.7 };
    var weights = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var biases = [_]f32{ 0.1, -0.1, 0.2, -0.2 };

    var weighted: [4]f32 = undefined;
    var biased: [4]f32 = undefined;
    var relu_out: [4]f32 = undefined;
    var softmax_out: [4]f32 = undefined;

    // Element-wise multiply (manual — tensors do add, not mul)
    for (&weighted, activations, weights) |*w, act, wgt| {
        w.* = act * wgt;
    }

    // Bias-add via tensor
    var t_weighted = Tensor32.init(weighted[0..], Shape.vec(4));
    var t_bias = Tensor32.init(biases[0..], Shape.vec(4));
    var t_biased = Tensor32.init(biased[0..], Shape.vec(4));
    Tensor32.add(&t_weighted, &t_bias, &t_biased);

    // ReLU activation
    var t_relu = Tensor32.init(relu_out[0..], Shape.vec(4));
    t_biased.relu(&t_relu);

    // Softmax normalization
    var t_sm = Tensor32.init(softmax_out[0..], Shape.vec(4));
    t_relu.softmax(&t_sm);

    std.debug.print("Weighted:  [{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n", .{
        weighted[0], weighted[1], weighted[2], weighted[3],
    });
    std.debug.print("Biased:    [{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n", .{
        biased[0], biased[1], biased[2], biased[3],
    });
    std.debug.print("ReLU:      [{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n", .{
        relu_out[0], relu_out[1], relu_out[2], relu_out[3],
    });
    std.debug.print("Softmax:   [{d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        softmax_out[0], softmax_out[1], softmax_out[2], softmax_out[3],
    });

    // ── 2. SIMD vector operations ───────────────────────────────────────
    // On Thumb targets without NEON, these compile to scalar loops.
    // On Cortex-M55 with MVE (Helium), the compiler can auto-vectorize.

    const query = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const key = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var attention: [4]f32 = undefined;

    abi.foundation.simd.vectorAdd(&query, &key, &attention);
    const similarity = abi.foundation.simd.vectorDot(&query, &key);

    std.debug.print("Query+Key: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{
        attention[0], attention[1], attention[2], attention[3],
    });
    std.debug.print("Dot similarity: {d:.3}\n", .{similarity});

    // ── 3. Bounded arithmetic ───────────────────────────────────────────
    // On embedded targets, clamp outputs to sensor/actuator ranges.
    const raw_output: f32 = similarity * 100.0;
    const pwm_duty = primitives.Math.clamp(f32, raw_output, 0.0, 255.0);
    std.debug.print("Raw output: {d:.1}  PWM duty (0-255): {d:.1}\n", .{ raw_output, pwm_duty });

    // ── 4. Activation energy summary ────────────────────────────────────
    const total_energy = t_relu.sum();
    const bounded_energy = primitives.Math.clamp(f32, total_energy, 0.0, 10.0);
    std.debug.print("Total activation energy: {d:.4} (bounded: {d:.4})\n", .{
        total_energy,
        bounded_energy,
    });

    std.debug.print("=== Demo complete ===\n", .{});
}
