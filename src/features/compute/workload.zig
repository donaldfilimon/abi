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
    }

    pub fn get(self: Matrix, row: usize, col: usize) f32 {
        return self.data[row * self.cols + col];
    }

    pub fn set(self: *Matrix, row: usize, col: usize, value: f32) void {
        self.data[row * self.cols + col] = value;
    }
};

pub fn matmul(out: *Matrix, a: Matrix, b: Matrix) void {
    std.debug.assert(a.cols == b.rows);
    std.debug.assert(out.rows == a.rows and out.cols == b.cols);

    for (0..out.rows) |i| {
        for (0..out.cols) |j| {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.get(i, k) * b.get(k, j);
            }
            out.set(i, j, sum);
        }
    }
}

pub fn matmulBlocked(out: *Matrix, a: Matrix, b: Matrix, block: usize) void {
    std.debug.assert(a.cols == b.rows);
    std.debug.assert(out.rows == a.rows and out.cols == b.cols);

    @memset(out.data, 0.0);
    var ii: usize = 0;
    while (ii < out.rows) : (ii += block) {
        var kk: usize = 0;
        while (kk < a.cols) : (kk += block) {
            var jj: usize = 0;
            while (jj < out.cols) : (jj += block) {
                const i_max = @min(ii + block, out.rows);
                const k_max = @min(kk + block, a.cols);
                const j_max = @min(jj + block, out.cols);

                for (ii..i_max) |i| {
                    for (kk..k_max) |k| {
                        const a_val = a.get(i, k);
                        for (jj..j_max) |j| {
                            out.data[i * out.cols + j] += a_val * b.get(k, j);
                        }
                    }
                }
            }
        }
    }
}

pub fn denseForward(
    output: []f32,
    input: []const f32,
    weights: Matrix,
    bias: []const f32,
) void {
    std.debug.assert(weights.cols == input.len);
    std.debug.assert(weights.rows == output.len);
    std.debug.assert(bias.len == output.len);

    for (0..weights.rows) |i| {
        var sum = bias[i];
        for (0..weights.cols) |j| {
            sum += weights.get(i, j) * input[j];
        }
        output[i] = sum;
    }
}

test "matrix multiply" {
    var a = try Matrix.init(std.testing.allocator, 2, 2);
    defer a.deinit();
    var b = try Matrix.init(std.testing.allocator, 2, 2);
    defer b.deinit();
    var out = try Matrix.init(std.testing.allocator, 2, 2);
    defer out.deinit();

    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(1, 0, 3.0);
    a.set(1, 1, 4.0);

    b.set(0, 0, 5.0);
    b.set(0, 1, 6.0);
    b.set(1, 0, 7.0);
    b.set(1, 1, 8.0);

    matmul(&out, a, b);
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), out.get(0, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out.get(0, 1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), out.get(1, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), out.get(1, 1), 0.001);
}
