const std = @import("std");
const builtin = @import("builtin");

// ── Timeout Policy ──────────────────────────────────────────────────────

/// Default timeout for standard CLI test functions (30 seconds).
pub const cli_timeout_ns: u64 = 30 * std.time.ns_per_s;

/// Default timeout for TUI / interactive test functions (60 seconds).
pub const tui_timeout_ns: u64 = 60 * std.time.ns_per_s;

/// Active timeout used by the runner. TUI tests get the longer deadline.
const active_timeout_ns: u64 = tui_timeout_ns;

pub fn main(_: std.process.Init) anyerror!void {
    @disableInstrumentation();

    const test_fn_list = builtin.test_functions;
    var passed: u64 = 0;
    var skipped: u64 = 0;
    var failed: u64 = 0;
    var timed_out: u64 = 0;

    for (test_fn_list) |test_fn| {
        std.debug.print("{s}... ", .{test_fn.name});

        const start_ts = getMonotonicNs();

        if (test_fn.func()) |_| {
            const elapsed_ns = getMonotonicNs() - start_ts;
            if (elapsed_ns > active_timeout_ns) {
                const elapsed_s = elapsed_ns / std.time.ns_per_s;
                const limit_s = active_timeout_ns / std.time.ns_per_s;
                std.debug.print(
                    "TIMEOUT [{s}] exceeded {d}s limit ({d}s actual)\n",
                    .{ test_fn.name, limit_s, elapsed_s },
                );
                timed_out += 1;
            } else {
                std.debug.print("PASS\n", .{});
                passed += 1;
            }
        } else |err| {
            const elapsed_ns = getMonotonicNs() - start_ts;
            if (elapsed_ns > active_timeout_ns) {
                const elapsed_s = elapsed_ns / std.time.ns_per_s;
                const limit_s = active_timeout_ns / std.time.ns_per_s;
                std.debug.print(
                    "TIMEOUT [{s}] exceeded {d}s limit ({d}s actual, error: {s})\n",
                    .{ test_fn.name, limit_s, elapsed_s, @errorName(err) },
                );
                timed_out += 1;
            } else if (err == error.SkipZigTest) {
                std.debug.print("SKIP\n", .{});
                skipped += 1;
            } else {
                std.debug.print("FAIL ({s})\n", .{@errorName(err)});
                failed += 1;
                return err;
            }
        }
    }

    std.debug.print(
        "{d} passed, {d} skipped, {d} failed, {d} timed-out\n",
        .{ passed, skipped, failed, timed_out },
    );

    if (failed != 0 or timed_out != 0) std.process.exit(1);
}

/// Monotonic nanosecond timestamp via POSIX clock_gettime.
/// Returns 0 on failure (timing degrades gracefully).
fn getMonotonicNs() u64 {
    if (comptime builtin.os.tag == .windows) return 0;
    var ts: std.posix.timespec = undefined;
    if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) return 0;
    const sec: u64 = @intCast(if (ts.sec < 0) 0 else ts.sec);
    const nsec: u64 = @intCast(if (ts.nsec < 0) 0 else ts.nsec);
    return sec * std.time.ns_per_s + nsec;
}
