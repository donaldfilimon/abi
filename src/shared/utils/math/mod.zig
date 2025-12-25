const std = @import("std");

pub const MinMax = struct {
    min: f64,
    max: f64,
};

pub fn clamp(value: anytype, min_value: @TypeOf(value), max_value: @TypeOf(value)) @TypeOf(value) {
    return std.math.clamp(value, min_value, max_value);
}

pub fn lerp(a: f64, b: f64, t: f64) f64 {
    return a + (b - a) * t;
}

pub fn mean(values: []const f64) f64 {
    if (values.len == 0) return 0;
    var total: f64 = 0;
    for (values) |value| {
        total += value;
    }
    return total / @as(f64, @floatFromInt(values.len));
}

pub fn sum(values: []const f64) f64 {
    var total: f64 = 0;
    for (values) |value| {
        total += value;
    }
    return total;
}

pub fn variance(values: []const f64) f64 {
    if (values.len == 0) return 0;
    const avg = mean(values);
    var acc: f64 = 0;
    for (values) |value| {
        const diff = value - avg;
        acc += diff * diff;
    }
    return acc / @as(f64, @floatFromInt(values.len));
}

pub fn stddev(values: []const f64) f64 {
    return std.math.sqrt(variance(values));
}

pub fn minMax(values: []const f64) ?MinMax {
    if (values.len == 0) return null;
    var min_value = values[0];
    var max_value = values[0];
    for (values[1..]) |value| {
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
    }
    return .{ .min = min_value, .max = max_value };
}

pub fn medianSorted(values: []const f64) f64 {
    if (values.len == 0) return 0;
    const mid = values.len / 2;
    if (values.len % 2 == 1) return values[mid];
    return (values[mid - 1] + values[mid]) / 2.0;
}

pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
    if (values.len == 0) return 0;
    const copy = try allocator.dupe(f64, values);
    defer allocator.free(copy);
    std.sort.heap(f64, copy, {}, comptime std.sort.asc(f64));
    return medianSorted(copy);
}

pub fn percentileSorted(values: []const f64, percentile_value: f64) f64 {
    if (values.len == 0) return 0;
    const clamped = clamp(percentile_value, 0.0, 1.0);
    const position = clamped * @as(f64, @floatFromInt(values.len - 1));
    const lower_index: usize = @intFromFloat(@floor(position));
    const upper_index: usize = @intFromFloat(@ceil(position));
    if (lower_index == upper_index) return values[lower_index];
    const weight = position - @as(f64, @floatFromInt(lower_index));
    return lerp(values[lower_index], values[upper_index], weight);
}

pub fn percentile(
    allocator: std.mem.Allocator,
    values: []const f64,
    percentile_value: f64,
) !f64 {
    if (values.len == 0) return 0;
    const copy = try allocator.dupe(f64, values);
    defer allocator.free(copy);
    std.sort.heap(f64, copy, {}, comptime std.sort.asc(f64));
    return percentileSorted(copy, percentile_value);
}

pub fn normalize(allocator: std.mem.Allocator, values: []const f64) ![]f64 {
    const output = try allocator.alloc(f64, values.len);
    if (values.len == 0) return output;
    const avg = mean(values);
    const deviation = stddev(values);
    if (deviation == 0) {
        @memset(output, 0);
        return output;
    }
    for (values, 0..) |value, i| {
        output[i] = (value - avg) / deviation;
    }
    return output;
}

pub fn normalizeMinMax(allocator: std.mem.Allocator, values: []const f64) ![]f64 {
    const output = try allocator.alloc(f64, values.len);
    if (values.len == 0) return output;
    const bounds = minMax(values) orelse return output;
    const range = bounds.max - bounds.min;
    if (range == 0) {
        @memset(output, 0);
        return output;
    }
    for (values, 0..) |value, i| {
        output[i] = (value - bounds.min) / range;
    }
    return output;
}

test "mean, variance, stddev, normalize" {
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean(&values), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), variance(&values), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1180), stddev(&values), 0.001);

    const normalized = try normalize(std.testing.allocator, &values);
    defer std.testing.allocator.free(normalized);
    try std.testing.expectEqual(values.len, normalized.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mean(normalized), 0.0001);
}

test "min max and median helpers" {
    const values = [_]f64{ 4.0, 1.0, 3.0, 2.0 };
    const bounds = minMax(&values) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bounds.min, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), bounds.max, 0.0001);

    const median_value = try median(std.testing.allocator, &values);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), median_value, 0.0001);

    const sorted = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), medianSorted(&sorted), 0.0001);
}

test "percentile and min-max normalization" {
    const values = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const p90 = percentileSorted(&values, 0.9);
    try std.testing.expectApproxEqAbs(@as(f64, 46.0), p90, 0.0001);

    const normalized = try normalizeMinMax(std.testing.allocator, &values);
    defer std.testing.allocator.free(normalized);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), normalized[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), normalized[4], 0.0001);
}
