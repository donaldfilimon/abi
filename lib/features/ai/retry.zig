const std = @import("std");

pub fn backoff_ms(attempt: u8, base_ms: u32, factor: f32) u32 {
    const exp: f32 = @floatFromInt(attempt);
    const mult = std.math.pow(f32, factor, exp);
    const scaled = @as(f32, @floatFromInt(base_ms)) * mult;
    return @intFromFloat(scaled);
}
