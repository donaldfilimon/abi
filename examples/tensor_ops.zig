//! Tensor + Matrix + SIMD Example
//!
//! Demonstrates a small end-to-end numeric path using shared modules:
//! - Matrix multiply via `abi.shared.matrix.Mat32`
//! - Tensor transforms via `abi.shared.tensor.Tensor32`
//! - SIMD vector ops via `abi.simd.vectorAdd` / `abi.simd.vectorDot`
//! - timing/clamp helpers via `abi.shared.utils.primitives`
//!
//! Run with: `zig build run-tensor-ops`

const std = @import("std");
const abi = @import("abi");

const Mat32 = abi.shared.matrix.Mat32;
const Tensor32 = abi.shared.tensor.Tensor32;
const Shape = abi.shared.tensor.Shape;
const primitives = abi.shared.utils.primitives;

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== ABI Tensor Ops Demo ===\n", .{});
    std.debug.print("Platform: {s}\n", .{primitives.Platform.description()});

    // ---------------------------------------------------------------------
    // 1) Matrix multiply: A(2x3) * B(3x2) => C(2x2)
    // ---------------------------------------------------------------------
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

    std.debug.print("Matrix result C (expected [58,64;139,154]): [{d:.1}, {d:.1}; {d:.1}, {d:.1}]\n", .{
        c.at(0, 0),
        c.at(0, 1),
        c.at(1, 0),
        c.at(1, 1),
    });

    // ---------------------------------------------------------------------
    // 2) Tensor pipeline: bias add -> ReLU -> softmax
    // ---------------------------------------------------------------------
    var matrix_row_major = [_]f32{
        c.at(0, 0), c.at(0, 1),
        c.at(1, 0), c.at(1, 1),
    };
    var input = Tensor32.init(matrix_row_major[0..], Shape.mat(2, 2));

    var bias_data = [_]f32{
        1.0, -1.0,
        0.5, -0.5,
    };
    var bias = Tensor32.init(bias_data[0..], Shape.mat(2, 2));

    var added_data: [4]f32 = undefined;
    var added = Tensor32.init(added_data[0..], Shape.mat(2, 2));
    Tensor32.add(&input, &bias, &added);

    var relu_data: [4]f32 = undefined;
    var relu_out = Tensor32.init(relu_data[0..], Shape.mat(2, 2));
    added.relu(&relu_out);

    var softmax_data: [4]f32 = undefined;
    var softmax_out = Tensor32.init(softmax_data[0..], Shape.mat(2, 2));
    relu_out.softmax(&softmax_out);

    std.debug.print("Tensor add+ReLU: [{d:.2}, {d:.2}; {d:.2}, {d:.2}]\n", .{
        relu_out.flat()[0],
        relu_out.flat()[1],
        relu_out.flat()[2],
        relu_out.flat()[3],
    });
    std.debug.print("Tensor softmax rows: [{d:.4}, {d:.4}; {d:.4}, {d:.4}]\n", .{
        softmax_out.flat()[0],
        softmax_out.flat()[1],
        softmax_out.flat()[2],
        softmax_out.flat()[3],
    });

    // ---------------------------------------------------------------------
    // 3) SIMD pass on flattened tensors
    // ---------------------------------------------------------------------
    var simd_add: [4]f32 = undefined;
    abi.simd.vectorAdd(input.flat(), bias.flat(), simd_add[0..]);
    const alignment_dot = abi.simd.vectorDot(relu_out.flat(), softmax_out.flat());
    const clamped_dot = primitives.Math.clamp(f32, alignment_dot, 0.0, 1_000_000.0);

    std.debug.print("SIMD add(flat input+bias): [{d:.2}, {d:.2}, {d:.2}, {d:.2}]\n", .{
        simd_add[0],
        simd_add[1],
        simd_add[2],
        simd_add[3],
    });
    std.debug.print("SIMD dot(ReLU, softmax): {d:.4} (clamped: {d:.4})\n", .{
        alignment_dot,
        clamped_dot,
    });

    const total_activation = relu_out.sum();
    const bounded_activation = primitives.Math.clamp(f32, total_activation, 0.0, 1_000_000.0);
    std.debug.print("Total ReLU activation: {d:.3} (bounded: {d:.3})\n", .{
        total_activation,
        bounded_activation,
    });
}
