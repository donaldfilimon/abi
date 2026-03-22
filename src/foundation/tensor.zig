//! ═══════════════════════════════════════════════════════════════════════════
//! ABI Framework — Tensor: N-Dimensional Array Primitives
//! Adapted from abi-system-v2.0/tensor.zig
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! Dense tensor storage for embedding vectors, attention matrices, and
//! inference buffers. Row-major layout with optional SIMD acceleration.
//!
//! Supported operations:
//!   - Shape manipulation (reshape, view, row)
//!   - Element-wise arithmetic (add, mul, scale, relu, gelu)
//!   - Reductions (sum, mean, max, argmax — per-axis or global)
//!   - Broadcasting (NumPy-style shape promotion)
//!   - Softmax (numerically stable, per-row for attention scores)
//!
//! Memory: Tensors borrow their data buffer from an external allocator.
//!         No implicit heap allocation — caller provides storage.
//! ═══════════════════════════════════════════════════════════════════════════

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// Maximum tensor rank (dimensions). 8 covers all practical ML shapes.
pub const max_rank = 8;

// ─── Shape ─────────────────────────────────────────────────────────────────

pub const Shape = struct {
    dims: [max_rank]usize = .{0} ** max_rank,
    rank: u8 = 0,

    pub fn init(dims: []const usize) Shape {
        var s = Shape{};
        s.rank = @intCast(@min(dims.len, max_rank));
        for (0..s.rank) |i| s.dims[i] = dims[i];
        return s;
    }

    pub fn scalar() Shape {
        return .{ .rank = 0 };
    }

    pub fn vec(n: usize) Shape {
        return init(&.{n});
    }

    pub fn mat(rows: usize, cols: usize) Shape {
        return init(&.{ rows, cols });
    }

    pub fn totalElements(self: *const Shape) usize {
        if (self.rank == 0) return 1; // scalar
        var total: usize = 1;
        for (0..self.rank) |i| total *= self.dims[i];
        return total;
    }

    pub fn eql(a: *const Shape, b: *const Shape) bool {
        if (a.rank != b.rank) return false;
        for (0..a.rank) |i| {
            if (a.dims[i] != b.dims[i]) return false;
        }
        return true;
    }

    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.rank];
    }

    /// Check if two shapes are broadcast-compatible (NumPy rules)
    pub fn broadcastable(a: *const Shape, b: *const Shape) bool {
        const max_r = @max(a.rank, b.rank);
        var i: usize = 0;
        while (i < max_r) : (i += 1) {
            const da = if (i < a.rank) a.dims[a.rank - 1 - i] else 1;
            const db = if (i < b.rank) b.dims[b.rank - 1 - i] else 1;
            if (da != db and da != 1 and db != 1) return false;
        }
        return true;
    }

    /// Compute broadcast output shape
    pub fn broadcastShape(a: *const Shape, b: *const Shape) Shape {
        var result = Shape{};
        result.rank = @max(a.rank, b.rank);
        var i: usize = 0;
        while (i < result.rank) : (i += 1) {
            const da = if (i < a.rank) a.dims[a.rank - 1 - i] else 1;
            const db = if (i < b.rank) b.dims[b.rank - 1 - i] else 1;
            result.dims[result.rank - 1 - i] = @max(da, db);
        }
        return result;
    }
};

// ─── Tensor ────────────────────────────────────────────────────────────────

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: Shape,
        strides: [max_rank]usize = .{0} ** max_rank,

        // ── Construction ────────────────────────────────────────────

        pub fn init(data: []T, shape: Shape) Self {
            var t = Self{ .data = data, .shape = shape };
            t.computeStrides();
            std.debug.assert(data.len >= shape.totalElements());
            return t;
        }

        pub fn alloc(allocator: std.mem.Allocator, shape: Shape) !Self {
            const n = shape.totalElements();
            const data = try allocator.alloc(T, n);
            @memset(data, 0);
            var t = Self{ .data = data, .shape = shape };
            t.computeStrides();
            return t;
        }

        pub fn free(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        /// Create a vector tensor from a slice (borrows data)
        pub fn fromSlice(data: []T) Self {
            return init(data, Shape.vec(data.len));
        }

        fn computeStrides(self: *Self) void {
            if (self.shape.rank == 0) return;
            self.strides[self.shape.rank - 1] = 1;
            var i: usize = self.shape.rank - 1;
            while (i > 0) {
                i -= 1;
                self.strides[i] = self.strides[i + 1] * self.shape.dims[i + 1];
            }
        }

        // ── Element Access ──────────────────────────────────────────

        pub fn at(self: *const Self, indices: []const usize) T {
            return self.data[self.flatIndex(indices)];
        }

        pub fn set(self: *Self, indices: []const usize, val: T) void {
            self.data[self.flatIndex(indices)] = val;
        }

        fn flatIndex(self: *const Self, indices: []const usize) usize {
            std.debug.assert(indices.len == self.shape.rank);
            var idx: usize = 0;
            for (0..self.shape.rank) |i| {
                std.debug.assert(indices[i] < self.shape.dims[i]);
                idx += indices[i] * self.strides[i];
            }
            return idx;
        }

        pub fn flat(self: *const Self) []const T {
            return self.data[0..self.shape.totalElements()];
        }

        pub fn flatMut(self: *Self) []T {
            return self.data[0..self.shape.totalElements()];
        }

        // ── Fill ────────────────────────────────────────────────────

        pub fn fill(self: *Self, val: T) void {
            @memset(self.flatMut(), val);
        }

        pub fn zeros(self: *Self) void {
            self.fill(0);
        }

        pub fn ones(self: *Self) void {
            self.fill(1);
        }

        pub fn arange(self: *Self) void {
            for (self.flatMut(), 0..) |*v, i| {
                v.* = @floatFromInt(i);
            }
        }

        // ── Shape Manipulation ──────────────────────────────────────

        pub fn reshape(self: *Self, new_shape: Shape) !void {
            if (new_shape.totalElements() != self.shape.totalElements())
                return error.ShapeMismatch;
            self.shape = new_shape;
            self.computeStrides();
        }

        pub fn view(self: *const Self, new_shape: Shape) !Self {
            if (new_shape.totalElements() != self.shape.totalElements())
                return error.ShapeMismatch;
            var t = Self{ .data = self.data, .shape = new_shape };
            t.computeStrides();
            return t;
        }

        pub fn row(self: *const Self, r: usize) Self {
            std.debug.assert(self.shape.rank == 2 and r < self.shape.dims[0]);
            const cols = self.shape.dims[1];
            const start = r * cols;
            return init(self.data[start .. start + cols], Shape.vec(cols));
        }

        // ── Element-Wise Operations ─────────────────────────────────

        pub fn add(a: *const Self, b: *const Self, out: *Self) void {
            std.debug.assert(a.shape.eql(&b.shape) and a.shape.eql(&out.shape));
            const n = a.shape.totalElements();
            simdBinaryOp(T, a.data[0..n], b.data[0..n], out.data[0..n], .add_op);
        }

        pub fn mul(a: *const Self, b: *const Self, out: *Self) void {
            std.debug.assert(a.shape.eql(&b.shape) and a.shape.eql(&out.shape));
            const n = a.shape.totalElements();
            simdBinaryOp(T, a.data[0..n], b.data[0..n], out.data[0..n], .mul_op);
        }

        pub fn scaleInPlace(self: *Self, scalar_val: T) void {
            const n = self.shape.totalElements();
            const lanes = comptime VectorSize;
            const V = @Vector(lanes, T);
            const vs: V = @splat(scalar_val);

            var i: usize = 0;
            while (i + lanes <= n) : (i += lanes) {
                var v: V = self.data[i..][0..lanes].*;
                v *= vs;
                self.data[i..][0..lanes].* = v;
            }
            while (i < n) : (i += 1) self.data[i] *= scalar_val;
        }

        pub fn relu(self: *const Self, out: *Self) void {
            const n = self.shape.totalElements();
            for (0..n) |i| {
                out.data[i] = @max(self.data[i], 0);
            }
        }

        /// GELU activation: x * Phi(x) ≈ 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
        pub fn gelu(self: *const Self, out: *Self) void {
            const n = self.shape.totalElements();
            const sqrt_2_over_pi: T = 0.7978845608;
            const coeff: T = 0.044715;

            for (0..n) |i| {
                const x = self.data[i];
                const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                out.data[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
            }
        }

        // ── Reductions ──────────────────────────────────────────────

        pub fn sum(self: *const Self) T {
            const n = self.shape.totalElements();
            const lanes = comptime VectorSize;
            const V = @Vector(lanes, T);
            var acc: V = @splat(@as(T, 0));

            var i: usize = 0;
            while (i + lanes <= n) : (i += lanes) {
                acc += self.data[i..][0..lanes].*;
            }
            var total: T = @reduce(.Add, acc);
            while (i < n) : (i += 1) total += self.data[i];
            return total;
        }

        pub fn mean(self: *const Self) T {
            const n = self.shape.totalElements();
            return self.sum() / @as(T, @floatFromInt(n));
        }

        pub fn max(self: *const Self) T {
            const data = self.flat();
            var result = data[0];
            for (data[1..]) |v| result = @max(result, v);
            return result;
        }

        pub fn argmax(self: *const Self) usize {
            const data = self.flat();
            var best: usize = 0;
            var best_val = data[0];
            for (data[1..], 1..) |v, i| {
                if (v > best_val) {
                    best_val = v;
                    best = i;
                }
            }
            return best;
        }

        pub fn sumAxis0(self: *const Self, out: *Self) void {
            std.debug.assert(self.shape.rank == 2);
            const rows = self.shape.dims[0];
            const cols = self.shape.dims[1];
            std.debug.assert(out.shape.totalElements() >= cols);

            out.zeros();
            for (0..rows) |r| {
                for (0..cols) |c_idx| {
                    out.data[c_idx] += self.data[r * cols + c_idx];
                }
            }
        }

        pub fn sumAxis1(self: *const Self, out: *Self) void {
            std.debug.assert(self.shape.rank == 2);
            const rows = self.shape.dims[0];
            const cols = self.shape.dims[1];
            std.debug.assert(out.shape.totalElements() >= rows);

            out.zeros();
            for (0..rows) |r| {
                var row_sum: T = 0;
                for (0..cols) |c_idx| row_sum += self.data[r * cols + c_idx];
                out.data[r] = row_sum;
            }
        }

        // ── Softmax ─────────────────────────────────────────────────

        pub fn softmax(self: *const Self, out: *Self) void {
            std.debug.assert(self.shape.eql(&out.shape));

            if (self.shape.rank == 1) {
                softmaxRow(self.flat(), out.flatMut());
                return;
            }

            std.debug.assert(self.shape.rank == 2);
            const rows = self.shape.dims[0];
            const cols = self.shape.dims[1];
            for (0..rows) |r| {
                const start = r * cols;
                softmaxRow(
                    self.data[start .. start + cols],
                    out.data[start .. start + cols],
                );
            }
        }

        fn softmaxRow(input: []const T, output: []T) void {
            const n = input.len;
            var max_val = input[0];
            for (input[1..]) |v| max_val = @max(max_val, v);

            var exp_sum: T = 0;
            for (0..n) |i| {
                const exp_val = @exp(input[i] - max_val);
                output[i] = exp_val;
                exp_sum += exp_val;
            }

            const inv_sum = 1.0 / exp_sum;
            for (0..n) |i| output[i] *= inv_sum;
        }

        // ── Approximate Equality ────────────────────────────────────

        pub fn approxEqual(a: *const Self, b: *const Self, tolerance: T) bool {
            if (!a.shape.eql(&b.shape)) return false;
            const n = a.shape.totalElements();
            for (0..n) |i| {
                if (@abs(a.data[i] - b.data[i]) > tolerance) return false;
            }
            return true;
        }

        // ── Diagnostics ─────────────────────────────────────────────

        pub fn print(self: *const Self, writer: anytype) !void {
            try writer.print("Tensor({s}) shape=[", .{@typeName(T)});
            for (0..self.shape.rank) |i| {
                if (i > 0) try writer.writeByte(',');
                try writer.print("{d}", .{self.shape.dims[i]});
            }
            try writer.writeAll("]");

            if (self.shape.rank <= 2 and self.shape.totalElements() <= 32) {
                try writer.writeAll(" data=[");
                for (self.flat(), 0..) |v, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.4}", .{v});
                }
                try writer.writeByte(']');
            }
            try writer.writeByte('\n');
        }
    };
}

// ─── SIMD Binary Operations ────────────────────────────────────────────────

const BinaryOp = enum { add_op, mul_op };

fn simdBinaryOp(comptime T: type, a: []const T, b: []const T, out: []T, comptime op: BinaryOp) void {
    const n = @min(a.len, @min(b.len, out.len));
    const lanes = comptime VectorSize;
    const V = @Vector(lanes, T);

    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        out[i..][0..lanes].* = switch (op) {
            .add_op => va + vb,
            .mul_op => va * vb,
        };
    }
    while (i < n) : (i += 1) {
        out[i] = switch (op) {
            .add_op => a[i] + b[i],
            .mul_op => a[i] * b[i],
        };
    }
}

// ─── Convenience Aliases ───────────────────────────────────────────────────

pub const Tensor32 = Tensor(f32);
pub const Tensor64 = Tensor(f64);

// ─── Tests ──────────────────────────────────────────────────────────────────

test "Shape basics" {
    const s = Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(u8, 3), s.rank);
    try std.testing.expectEqual(@as(usize, 24), s.totalElements());

    const s2 = Shape.mat(3, 4);
    try std.testing.expectEqual(@as(u8, 2), s2.rank);
    try std.testing.expectEqual(@as(usize, 12), s2.totalElements());

    const v = Shape.vec(5);
    try std.testing.expectEqual(@as(u8, 1), v.rank);

    const sc = Shape.scalar();
    try std.testing.expectEqual(@as(u8, 0), sc.rank);
    try std.testing.expectEqual(@as(usize, 1), sc.totalElements());
}

test "Shape broadcast compatibility" {
    const a = Shape.init(&.{ 1, 3 });
    const b = Shape.init(&.{ 4, 1 });
    try std.testing.expect(a.broadcastable(&b));

    const c = Shape.init(&.{ 4, 3 });
    const d = Shape.init(&.{ 5, 3 });
    try std.testing.expect(!c.broadcastable(&d));

    const result = a.broadcastShape(&b);
    try std.testing.expectEqual(@as(usize, 4), result.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), result.dims[1]);
}

test "Tensor init and element access" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t = Tensor32.init(&data, Shape.mat(2, 3));

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), t.at(&.{ 0, 0 }), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), t.at(&.{ 1, 0 }), 0.001);
    t.set(&.{ 0, 1 }, 99.0);
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), t.at(&.{ 0, 1 }), 0.001);
}

test "Tensor alloc, fill, and free" {
    const alloc = std.testing.allocator;
    var t = try Tensor32.alloc(alloc, Shape.vec(8));
    defer t.free(alloc);

    t.fill(3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), t.at(&.{4}), 0.001);
    t.arange();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), t.at(&.{5}), 0.001);
}

test "Tensor add and mul" {
    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 10, 20, 30, 40 };
    var out_data: [4]f32 = undefined;

    const shape = Shape.vec(4);
    var a = Tensor32.init(&a_data, shape);
    const b = Tensor32.init(&b_data, shape);
    var out = Tensor32.init(&out_data, shape);

    Tensor32.add(&a, &b, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), out.data[3], 0.001);

    Tensor32.mul(&a, &b, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 160.0), out.data[3], 0.001);
}

test "Tensor reductions" {
    var data = [_]f32{ 1, 2, 3, 4, 5 };
    const t = Tensor32.init(&data, Shape.vec(5));

    try std.testing.expectApproxEqAbs(@as(f32, 15.0), t.sum(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), t.mean(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), t.max(), 0.001);
    try std.testing.expectEqual(@as(usize, 4), t.argmax());
}

test "Tensor softmax sums to 1" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    var out_data: [3]f32 = undefined;

    const shape = Shape.vec(3);
    const t = Tensor32.init(&data, shape);
    var out = Tensor32.init(&out_data, shape);

    t.softmax(&out);

    // Softmax output must sum to ~1.0
    var total: f32 = 0;
    for (out.flat()) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.001);

    // Values should be ordered: out[2] > out[1] > out[0]
    try std.testing.expect(out.data[2] > out.data[1]);
    try std.testing.expect(out.data[1] > out.data[0]);
}

test "Tensor reshape and view" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t = Tensor32.init(&data, Shape.vec(6));

    try t.reshape(Shape.mat(2, 3));
    try std.testing.expectEqual(@as(u8, 2), t.shape.rank);
    try std.testing.expectEqual(@as(usize, 2), t.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), t.shape.dims[1]);

    // Mismatched element count should error
    try std.testing.expectError(error.ShapeMismatch, t.reshape(Shape.vec(5)));
}

test "Tensor scaleInPlace" {
    var data = [_]f32{ 1, 2, 3, 4 };
    var t = Tensor32.init(&data, Shape.vec(4));
    t.scaleInPlace(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), t.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), t.data[3], 0.001);
}

test "Tensor relu" {
    var data = [_]f32{ -2, -1, 0, 1, 2 };
    var out_data: [5]f32 = undefined;
    const shape = Shape.vec(5);
    const t = Tensor32.init(&data, shape);
    var out = Tensor32.init(&out_data, shape);
    t.relu(&out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.data[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out.data[4], 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
