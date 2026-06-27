pub const std = @import("std");

/// Process-wide default IO. This std exposes the system clock only through an
/// `Io` handle; `std.Options.debug_io` is a target-appropriate default that is
/// usable without threading an `Io` through every timestamp call site, keeping
/// `unixMs()`/`monotonicNs()` allocation-free, libc-free, and portable across
/// macOS/Linux/Windows.
const clock_io: std.Io = std.Options.debug_io;

/// Current Unix time in milliseconds (epoch since 1970-01-01). Portable across
/// macOS/Linux/Windows via the `Io` real clock (no libc dependency); return
/// type and semantics match the previous `clock_gettime`-based implementation.
pub fn unixMs() i64 {
    return std.Io.Clock.real.now(clock_io).toMilliseconds();
}

/// Monotonic timestamp in nanoseconds, sourced from the `awake` clock (the
/// CLOCK_MONOTONIC equivalent: advances while the system is awake, unaffected
/// by wall-clock jumps). Portable and libc-free; used for coarse duration and
/// telemetry sampling.
pub fn monotonicNs() i64 {
    return @intCast(std.Io.Clock.awake.now(clock_io).toNanoseconds());
}

test {
    std.testing.refAllDecls(@This());
}

test "unixMs returns a positive timestamp" {
    const t = unixMs();
    try std.testing.expect(t > 0);
}

test "unixMs is monotonic" {
    const t1 = unixMs();
    const t2 = unixMs();
    try std.testing.expect(t2 >= t1);
}
