pub const std = @import("std");

/// Current Unix time in milliseconds (best-effort via libc `clock_gettime`).
pub fn unixMs() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.REALTIME, &ts) != 0)
        return 0;
    const sec_ms = @as(i64, ts.sec) * std.time.ms_per_s;
    const nsec_ms = @divTrunc(@as(i64, ts.nsec), std.time.ns_per_ms);
    return sec_ms + nsec_ms;
}
