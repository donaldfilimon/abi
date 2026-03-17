//! RISC-V 32 Bare Metal Deployment Example
//!
//! Demonstrates how to use ABI framework types in a freestanding / bare-metal
//! context targeting RISC-V 32-bit cores.  The patterns shown here are
//! portable across any `os.tag == .freestanding` target.
//!
//! Key bare-metal patterns demonstrated:
//!   - Fixed-buffer allocator instead of OS-backed heap
//!   - Tensor creation and element-wise math (add, ReLU, softmax)
//!   - Matrix multiply on small, statically-sized matrices
//!   - SIMD vector dot-product and addition
//!   - Compile-time version query (zero runtime cost)
//!
//! What is intentionally avoided:
//!   - `std.os`, `std.fs`, `std.net`, `std.process` (require an OS)
//!   - `abi.App.initMinimal` (needs the IO backend, which is a no-op on
//!     freestanding but also needs `std.Io.Threaded` on hosted builds)
//!   - Heap allocators that call into the OS (`DebugAllocator`, page allocator)
//!
//! Run with: `zig build run-bare-metal-riscv32`

const std = @import("std");
const abi = @import("abi");

const Tensor32 = abi.foundation.tensor.Tensor32;
const Shape = abi.foundation.tensor.Shape;
const Mat32 = abi.foundation.matrix.Mat32;
const primitives = abi.foundation.utils.primitives;

pub fn main(_: std.process.Init) !void {
    // ── 0. Allocator ────────────────────────────────────────────────────
    // On bare metal there is no OS heap.  A fixed-buffer allocator backed
    // by a static array is the simplest portable option.  Real firmware
    // would point this at a linker-script-defined RAM region.
    var buf: [8192]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const allocator = fba.allocator();

    std.debug.print("\n=== ABI Bare-Metal RISC-V 32 Demo ===\n", .{});
    std.debug.print("ABI version: {s}\n", .{abi.version()});

    // ── 1. Matrix multiply ──────────────────────────────────────────────
    // A(2x3) * B(3x2) => C(2x2)
    var a = try Mat32.alloc(allocator, 2, 3);
    defer a.free(allocator);
    var b = try Mat32.alloc(allocator, 3, 2);
    defer b.free(allocator);
    var c = try Mat32.alloc(allocator, 2, 2);
    defer c.free(allocator);

    // A = [[1,2,3],[4,5,6]]
    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(0, 2, 3.0);
    a.set(1, 0, 4.0);
    a.set(1, 1, 5.0);
    a.set(1, 2, 6.0);

    // B = [[7,8],[9,10],[11,12]]
    b.set(0, 0, 7.0);
    b.set(0, 1, 8.0);
    b.set(1, 0, 9.0);
    b.set(1, 1, 10.0);
    b.set(2, 0, 11.0);
    b.set(2, 1, 12.0);

    Mat32.multiply(&a, &b, &c);

    std.debug.print("Matrix C (expected [58,64;139,154]): [{d:.0}, {d:.0}; {d:.0}, {d:.0}]\n", .{
        c.at(0, 0),
        c.at(0, 1),
        c.at(1, 0),
        c.at(1, 1),
    });

    // ── 2. Tensor pipeline: bias-add -> ReLU -> softmax ─────────────────
    var input_data = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    var bias_data = [_]f32{ 0.5, 0.5, -0.5, 0.5 };
    var add_out: [4]f32 = undefined;
    var relu_out: [4]f32 = undefined;
    var softmax_out: [4]f32 = undefined;

    var input = Tensor32.init(input_data[0..], Shape.mat(2, 2));
    var bias = Tensor32.init(bias_data[0..], Shape.mat(2, 2));
    var added = Tensor32.init(add_out[0..], Shape.mat(2, 2));
    var relu = Tensor32.init(relu_out[0..], Shape.mat(2, 2));
    var sm = Tensor32.init(softmax_out[0..], Shape.mat(2, 2));

    Tensor32.add(&input, &bias, &added);
    added.relu(&relu);
    relu.softmax(&sm);

    std.debug.print("Tensor add:     [{d:.2}, {d:.2}, {d:.2}, {d:.2}]\n", .{
        add_out[0], add_out[1], add_out[2], add_out[3],
    });
    std.debug.print("Tensor ReLU:    [{d:.2}, {d:.2}, {d:.2}, {d:.2}]\n", .{
        relu_out[0], relu_out[1], relu_out[2], relu_out[3],
    });
    std.debug.print("Tensor softmax: [{d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        softmax_out[0], softmax_out[1], softmax_out[2], softmax_out[3],
    });

    // ── 3. SIMD vector ops ──────────────────────────────────────────────
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var vec_sum: [4]f32 = undefined;

    abi.foundation.simd.vectorAdd(&vec_a, &vec_b, &vec_sum);
    const dot = abi.foundation.simd.vectorDot(&vec_a, &vec_b);
    const clamped = primitives.Math.clamp(f32, dot, 0.0, 100.0);

    std.debug.print("SIMD add: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{
        vec_sum[0], vec_sum[1], vec_sum[2], vec_sum[3],
    });
    std.debug.print("SIMD dot: {d:.1}  clamped: {d:.1}\n", .{ dot, clamped });

    // ── 4. Total activation (scalar reduce) ─────────────────────────────
    const total = relu.sum();
    const bounded = primitives.Math.clamp(f32, total, 0.0, 1000.0);
    std.debug.print("Total ReLU activation: {d:.3} (bounded: {d:.3})\n", .{ total, bounded });

    std.debug.print("=== Demo complete ===\n", .{});
}
