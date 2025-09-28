//! SIMD operations and vectorized computations
//! Updated for Zig 0.16 compatibility

const std = @import("std");
const shared = @import("shared/simd.zig");
const collections = @import("core/collections.zig");

// Re-export from shared module
pub const SIMDOpts = shared.SIMDOpts;
pub const getPerformanceMonitor = shared.getPerformanceMonitor;
pub const getPerformanceMonitorDetails = shared.getPerformanceMonitorDetails;
pub const getVectorOps = shared.getVectorOps;
pub const text = shared.text;

pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: shared.SIMDOpts) f32 {
    return shared.dotProductSIMD(a, b, opts);
}

/// Additional vector operations using SIMD when available
pub const VectorOps = struct {
    /// Add two f32 vectors element-wise
    pub fn addF32(a: []const f32, b: []const f32, result: []f32) void {
        std.debug.assert(a.len == b.len and a.len == result.len);

        for (a, b, result) |a_val, b_val, *res_val| {
            res_val.* = a_val + b_val;
        }
    }

    /// Multiply two f32 vectors element-wise
    pub fn mulF32(a: []const f32, b: []const f32, result: []f32) void {
        std.debug.assert(a.len == b.len and a.len == result.len);

        for (a, b, result) |a_val, b_val, *res_val| {
            res_val.* = a_val * b_val;
        }
    }

    /// Dot product of two f32 vectors
    pub fn dotF32(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        var result: f32 = 0.0;
        for (a, b) |a_val, b_val| {
            result += a_val * b_val;
        }
        return result;
    }
};

/// SIMD-aware buffer management
pub const SIMDBuffer = struct {
    const Self = @This();

    data: collections.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .data = collections.ArrayList(f32){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.data.deinit(self.allocator);
    }

    pub fn append(self: *Self, value: f32) !void {
        try self.data.append(self.allocator, value);
    }

    pub fn items(self: *const Self) []f32 {
        return self.data.items;
    }
};

test "simd - vector operations" {
    const testing = std.testing;

    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    var result = [_]f32{ 0.0, 0.0, 0.0 };

    VectorOps.addF32(&a, &b, &result);
    try testing.expectEqual(@as(f32, 5.0), result[0]);
    try testing.expectEqual(@as(f32, 7.0), result[1]);
    try testing.expectEqual(@as(f32, 9.0), result[2]);

    const dot = VectorOps.dotF32(&a, &b);
    try testing.expectEqual(@as(f32, 32.0), dot); // 1*4 + 2*5 + 3*6 = 32
}

test "simd - buffer management" {
    const testing = std.testing;

    var buffer = SIMDBuffer.init(testing.allocator);
    defer buffer.deinit();

    try buffer.append(1.0);
    try buffer.append(2.0);

    try testing.expectEqual(@as(usize, 2), buffer.items().len);
    try testing.expectEqual(@as(f32, 1.0), buffer.items()[0]);
}
