const std = @import("std");
const time_mod = @import("../../foundation/mod.zig").time;

/// MQTT-style pattern matching.
/// `*` matches exactly one level, `#` matches zero or more levels.
pub fn patternMatches(pattern: []const u8, topic: []const u8) bool {
    var pat_iter = std.mem.splitScalar(u8, pattern, '.');
    var top_iter = std.mem.splitScalar(u8, topic, '.');

    while (true) {
        const pat_seg = pat_iter.next();
        const top_seg = top_iter.next();

        if (pat_seg == null and top_seg == null) return true;

        if (pat_seg) |p| {
            if (std.mem.eql(u8, p, "#")) return true; // # matches rest
            if (top_seg == null) return false;
            if (std.mem.eql(u8, p, "*")) continue; // * matches one level
            if (!std.mem.eql(u8, p, top_seg.?)) return false;
        } else {
            return false; // pattern ended but topic continues
        }
    }
}

pub fn nowMs() u64 {
    const instant = time_mod.Instant.now() catch return 0;
    return @intCast(instant.nanos / std.time.ns_per_ms);
}
