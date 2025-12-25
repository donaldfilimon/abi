const std = @import("std");

pub fn clamp(value: anytype, min_value: @TypeOf(value), max_value: @TypeOf(value)) @TypeOf(value) {
    return std.math.clamp(value, min_value, max_value);
}

pub fn lerp(a: f64, b: f64, t: f64) f64 {
    return a + (b - a) * t;
}

pub fn mean(values: []const f64) f64 {
    if (values.len == 0) return 0;
    var sum: f64 = 0;
    for (values) |value| {
        sum += value;
    }
    return sum / @as(f64, @floatFromInt(values.len));
}
