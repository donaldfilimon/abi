// ============================================================================
// ABI Framework — Dense Matrix Operations
// Adapted from abi-system-v2.0/matrix.zig
// ============================================================================
//
// Column-major dense matrix with SIMD-accelerated operations.
// Supports: multiply (naive + tiled), transpose, matvec, norms.
// ============================================================================

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

// ─── Dense Matrix ──────────────────────────────────────────────────────────

pub fn Matrix(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64)
            @compileError("Matrix only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        data: []T,
        rows: usize,
        cols: usize,
        stride: usize,

        pub fn fromSlice(data: []T, rows: usize, cols: usize) Self {
            const s = rows;
            std.debug.assert(data.len >= s * cols);
            return Self{ .data = data, .rows = rows, .cols = cols, .stride = s };
        }

        pub fn alloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
            const n = rows * cols;
            const data = try allocator.alloc(T, n);
            @memset(data, 0);
            return Self{ .data = data, .rows = rows, .cols = cols, .stride = rows };
        }

        pub fn free(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        // ── Element Access ──────────────────────────────────────────

        pub inline fn at(self: *const Self, row_idx: usize, col_idx: usize) T {
            std.debug.assert(row_idx < self.rows and col_idx < self.cols);
            return self.data[col_idx * self.stride + row_idx];
        }

        pub inline fn atPtr(self: *Self, row_idx: usize, col_idx: usize) *T {
            std.debug.assert(row_idx < self.rows and col_idx < self.cols);
            return &self.data[col_idx * self.stride + row_idx];
        }

        pub fn set(self: *Self, row_idx: usize, col_idx: usize, val: T) void {
            self.atPtr(row_idx, col_idx).* = val;
        }

        pub fn col(self: *const Self, c: usize) []const T {
            std.debug.assert(c < self.cols);
            const start = c * self.stride;
            return self.data[start .. start + self.rows];
        }

        pub fn colMut(self: *Self, c: usize) []T {
            std.debug.assert(c < self.cols);
            const start = c * self.stride;
            return self.data[start .. start + self.rows];
        }

        // ── Fill ────────────────────────────────────────────────────

        pub fn fill(self: *Self, val: T) void {
            @memset(self.data[0 .. self.stride * self.cols], val);
        }

        pub fn zero(self: *Self) void {
            self.fill(0);
        }

        pub fn identity(self: *Self) void {
            std.debug.assert(self.rows == self.cols);
            self.zero();
            for (0..self.rows) |i| self.set(i, i, 1);
        }

        // ── Matrix Multiply ─────────────────────────────────────────

        pub fn multiply(a: *const Self, b: *const Self, c: *Self) void {
            std.debug.assert(a.cols == b.rows);
            std.debug.assert(c.rows == a.rows and c.cols == b.cols);

            const m = a.rows;
            const n = b.cols;
            const k = a.cols;

            c.zero();

            const tile = if (m >= 64 and n >= 64 and k >= 64) @as(usize, 32) else m;

            if (tile < m) {
                tiledMultiply(a, b, c, m, n, k, tile);
            } else {
                naiveMultiply(a, b, c, m, n, k);
            }
        }

        fn naiveMultiply(a: *const Self, b: *const Self, c: *Self, m: usize, n: usize, k: usize) void {
            _ = m;
            for (0..n) |j| {
                for (0..k) |p| {
                    const b_pj = b.at(p, j);
                    const a_col_data = a.col(p);
                    const c_col_data = c.colMut(j);
                    vectorizedSaxpy(T, b_pj, a_col_data, c_col_data);
                }
            }
        }

        fn tiledMultiply(a: *const Self, b: *const Self, c: *Self, m: usize, n: usize, k: usize, tile: usize) void {
            var jj: usize = 0;
            while (jj < n) : (jj += tile) {
                const j_end = @min(jj + tile, n);
                var pp: usize = 0;
                while (pp < k) : (pp += tile) {
                    const p_end = @min(pp + tile, k);
                    for (jj..j_end) |j| {
                        for (pp..p_end) |p| {
                            const b_pj = b.at(p, j);
                            for (0..m) |i| {
                                c.atPtr(i, j).* += a.at(i, p) * b_pj;
                            }
                        }
                    }
                }
            }
        }

        // ── Matrix-Vector Product ───────────────────────────────────

        pub fn matvec(a: *const Self, x: []const T, y: []T) void {
            std.debug.assert(x.len == a.cols);
            std.debug.assert(y.len == a.rows);

            @memset(y, 0);

            for (0..a.cols) |j| {
                const x_j = x[j];
                const a_col_data = a.col(j);
                vectorizedSaxpy(T, x_j, a_col_data, y);
            }
        }

        // ── Transpose ───────────────────────────────────────────────

        pub fn transpose(a: *const Self, b: *Self) void {
            std.debug.assert(b.rows == a.cols and b.cols == a.rows);
            for (0..a.rows) |i| {
                for (0..a.cols) |j| {
                    b.set(j, i, a.at(i, j));
                }
            }
        }

        pub fn transposeInPlace(self: *Self) void {
            std.debug.assert(self.rows == self.cols);
            for (0..self.rows) |i| {
                for (i + 1..self.cols) |j| {
                    const tmp = self.at(i, j);
                    self.set(i, j, self.at(j, i));
                    self.set(j, i, tmp);
                }
            }
        }

        // ── Norms ───────────────────────────────────────────────────

        pub fn frobeniusNorm(self: *const Self) T {
            var s: T = 0;
            const total = self.stride * self.cols;
            for (self.data[0..total]) |v| s += v * v;
            return @sqrt(s);
        }

        pub fn maxAbs(self: *const Self) T {
            var result: T = 0;
            const total = self.stride * self.cols;
            for (self.data[0..total]) |v| {
                result = @max(result, @abs(v));
            }
            return result;
        }

        // ── Element-Wise ────────────────────────────────────────────

        pub fn addMatrix(a: *const Self, b: *const Self, c: *Self) void {
            std.debug.assert(a.rows == b.rows and a.cols == b.cols);
            std.debug.assert(c.rows == a.rows and c.cols == a.cols);
            const n = a.stride * a.cols;
            for (0..n) |i| c.data[i] = a.data[i] + b.data[i];
        }

        pub fn scaleInPlace(self: *Self, scalar_val: T) void {
            const n = self.stride * self.cols;
            for (self.data[0..n]) |*v| v.* *= scalar_val;
        }

        // ── Diagnostics ─────────────────────────────────────────────

        pub fn approxEqual(a: *const Self, b: *const Self, tolerance: T) bool {
            if (a.rows != b.rows or a.cols != b.cols) return false;
            for (0..a.rows) |i| {
                for (0..a.cols) |j| {
                    if (@abs(a.at(i, j) - b.at(i, j)) > tolerance) return false;
                }
            }
            return true;
        }
    };
}

// ─── Vectorized SAXPY Helper ───────────────────────────────────────────────

fn vectorizedSaxpy(comptime T: type, a: T, x: []const T, y: []T) void {
    const n = @min(x.len, y.len);

    const lanes = VectorSize;
    const VecT = @Vector(lanes, T);
    const va: VecT = @splat(a);

    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const vx: VecT = x[i..][0..lanes].*;
        var vy: VecT = y[i..][0..lanes].*;
        vy += va * vx;
        y[i..][0..lanes].* = vy;
    }

    while (i < n) : (i += 1) {
        y[i] += a * x[i];
    }
}

// ─── Convenience Aliases ───────────────────────────────────────────────────

pub const Mat32 = Matrix(f32);
pub const Mat64 = Matrix(f64);

// ─── Tests ──────────────────────────────────────────────────────────────────

test "Matrix alloc, set, at" {
    const alloc = std.testing.allocator;
    var m = try Mat32.alloc(alloc, 3, 3);
    defer m.free(alloc);

    m.set(0, 0, 1.0);
    m.set(1, 2, 5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), m.at(0, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), m.at(1, 2), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), m.at(2, 2), 0.001);
}

test "Matrix identity" {
    const alloc = std.testing.allocator;
    var m = try Mat32.alloc(alloc, 3, 3);
    defer m.free(alloc);

    m.identity();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), m.at(0, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), m.at(1, 1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), m.at(0, 1), 0.001);
}

test "Matrix multiply 2x2" {
    const alloc = std.testing.allocator;
    var a = try Mat32.alloc(alloc, 2, 2);
    defer a.free(alloc);
    var b = try Mat32.alloc(alloc, 2, 2);
    defer b.free(alloc);
    var c = try Mat32.alloc(alloc, 2, 2);
    defer c.free(alloc);

    // A = [[1,2],[3,4]] (column-major: col0=[1,3], col1=[2,4])
    a.set(0, 0, 1);
    a.set(0, 1, 2);
    a.set(1, 0, 3);
    a.set(1, 1, 4);

    // B = [[5,6],[7,8]]
    b.set(0, 0, 5);
    b.set(0, 1, 6);
    b.set(1, 0, 7);
    b.set(1, 1, 8);

    // C = A*B = [[19,22],[43,50]]
    Mat32.multiply(&a, &b, &c);
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), c.at(0, 0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), c.at(0, 1), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), c.at(1, 0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), c.at(1, 1), 0.01);
}

test "Matrix transpose" {
    const alloc = std.testing.allocator;
    var a = try Mat32.alloc(alloc, 2, 3);
    defer a.free(alloc);
    var b = try Mat32.alloc(alloc, 3, 2);
    defer b.free(alloc);

    a.set(0, 0, 1);
    a.set(0, 1, 2);
    a.set(0, 2, 3);
    a.set(1, 0, 4);
    a.set(1, 1, 5);
    a.set(1, 2, 6);

    Mat32.transpose(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.at(0, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), b.at(0, 1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), b.at(1, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), b.at(1, 1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), b.at(2, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), b.at(2, 1), 0.001);
}

test "Matrix matvec" {
    const alloc = std.testing.allocator;
    // A = [[1,2],[3,4],[5,6]], x = [1,1]
    var a = try Mat32.alloc(alloc, 3, 2);
    defer a.free(alloc);

    a.set(0, 0, 1);
    a.set(0, 1, 2);
    a.set(1, 0, 3);
    a.set(1, 1, 4);
    a.set(2, 0, 5);
    a.set(2, 1, 6);

    var x = [_]f32{ 1.0, 1.0 };
    var y: [3]f32 = undefined;
    a.matvec(&x, &y);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), y[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), y[2], 0.01);
}

test "Matrix frobenius norm" {
    const alloc = std.testing.allocator;
    var m = try Mat32.alloc(alloc, 2, 2);
    defer m.free(alloc);

    m.set(0, 0, 3.0);
    m.set(1, 1, 4.0);
    // frobenius = sqrt(9 + 16) = 5
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), m.frobeniusNorm(), 0.01);
}

test "Matrix scaleInPlace" {
    const alloc = std.testing.allocator;
    var m = try Mat32.alloc(alloc, 2, 2);
    defer m.free(alloc);

    m.set(0, 0, 2.0);
    m.set(1, 1, 3.0);
    m.scaleInPlace(10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), m.at(0, 0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), m.at(1, 1), 0.01);
}
