//! Matrix multiplication workload
//!
//! Work item that owns matrices and executes matmul.

const std = @import("std");

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0.0);
        return .{ .rows = rows, .cols = cols, .data = data, .allocator = allocator };
    }

    pub fn deinit(self: *Matrix) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

pub const MatrixMultiplication = struct {
    matrix_a: Matrix,
    matrix_b: Matrix,
    output: Matrix,

    pub fn init(matrix_a: Matrix, matrix_b: Matrix, output: Matrix) MatrixMultiplication {
        return .{
            .matrix_a = matrix_a,
            .matrix_b = matrix_b,
            .output = output,
        };
    }

    pub fn deinit(self: *MatrixMultiplication) void {
        self.matrix_a.deinit();
        self.matrix_b.deinit();
        self.output.deinit();
        self.* = undefined;
    }
};
