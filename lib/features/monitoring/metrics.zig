const std = @import("std");

pub fn logCall(provider: []const u8, ok: bool, latency_ms: u32) void {
    std.log.info("provider={s} ok={} latency_ms={}", .{ provider, ok, latency_ms });
}
