//! SIMD vector primitives
//!
//! Provides typed SIMD vector wrappers with proper type conversions
//! and common vector operations.

const std = @import("std");

pub fn ComputeVector(comptime width: usize) type {
    return struct {
        const Self = @This();
        pub const VectorType = @Vector(width, f32);

        data: VectorType,

        /// Initialize from array with explicit cast
        pub inline fn init(values: [width]f32) Self {
            return .{ .data = @as(VectorType, values) };
        }

        /// Initialize from slice (requires exact width match)
        pub inline fn initFromSlice(values: []const f32) !Self {
            if (values.len != width) return error.SliceLengthMismatch;
            var arr: [width]f32 = undefined;
            @memcpy(arr[0..], values[0..width]);
            return Self{ .data = @as(VectorType, arr) };
        }

        /// Dot product with another vector
        pub inline fn dot(self: Self, other: Self) f32 {
            return @reduce(.Add, self.data * other.data);
        }

        /// Vector magnitude (L2 norm)
        pub inline fn magnitude(self: Self) f32 {
            return std.math.sqrt(self.dot(self));
        }

        /// Normalize in place
        pub inline fn normalize(self: *Self) void {
            const mag = self.magnitude();
            if (mag > 0.0) {
                self.*.data = self.data / @as(VectorType, @splat(mag));
            }
        }

        /// Fused multiply-add: self = self + (a * b)
        pub inline fn fma(self: *Self, a: Self, b: Self) void {
            self.*.data = self.data + a.data * b.data;
        }

        /// Add scalar to each element
        pub inline fn addScalar(self: *Self, scalar: f32) void {
            const scalar_vec: VectorType = @as(VectorType, @splat(scalar));
            self.*.data = self.data + scalar_vec;
        }

        /// Scale by scalar
        pub inline fn scale(self: *Self, scalar: f32) void {
            const scalar_vec: VectorType = @as(VectorType, @splat(scalar));
            self.*.data = self.data * scalar_vec;
        }

        /// Copy data to slice (slice must have width elements)
        pub inline fn store(self: Self, out: []f32) void {
            std.debug.assert(out.len >= width);
            const arr: [width]f32 = @as([width]f32, self.data);
            @memcpy(out[0..width], arr[0..]);
        }
    };
}

test "ComputeVector init with array" {
    const Vec4 = ComputeVector(4);
    const v = Vec4.init([_]f32{ 1.0, 2.0, 3.0, 4.0 });
    try std.testing.expectEqual(@as(f32, 1.0), v.data[0]);
}

test "ComputeVector dot product" {
    const Vec4 = ComputeVector(4);
    const a = Vec4.init([_]f32{ 1.0, 2.0, 3.0, 4.0 });
    const b = Vec4.init([_]f32{ 2.0, 3.0, 4.0, 5.0 });
    const expected: f32 = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;
    try std.testing.expectApproxEqAbs(expected, a.dot(b), 0.001);
}
