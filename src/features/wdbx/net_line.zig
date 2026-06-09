//! Newline-framed socket read helper (Transport Layer).
//!
//! Shared by the cluster_rpc and remote_compute transports. A single
//! `Stream.read` returns only whatever bytes the kernel had ready, so a request
//! or reply split across TCP segments would be truncated by a naive one-shot
//! read. `readLine` accumulates into the caller's buffer until a newline is seen
//! (or the buffer fills / the peer closes), then returns the frame without its
//! trailing CR/LF. Correct on loopback (one read) and on a routable interface
//! (multiple segments).

const std = @import("std");

const Stream = std.Io.net.Stream;

/// Read one newline-terminated frame into `buf`, reassembling across reads.
/// Returns the bytes up to (not including) the first '\n', trimmed of a trailing
/// '\r'. If the peer closes before a newline, returns whatever was received.
pub fn readLine(io: std.Io, conn: Stream, buf: []u8) ![]const u8 {
    var total: usize = 0;
    while (total < buf.len) {
        var rv: [1][]u8 = .{buf[total..]};
        const n = try conn.read(io, &rv);
        if (n == 0) break; // EOF / peer closed
        total += n;
        if (std.mem.indexOfScalar(u8, buf[0..total], '\n') != null) break;
    }
    const end = std.mem.indexOfScalar(u8, buf[0..total], '\n') orelse {
        // No newline and the buffer is full: the frame exceeds `buf` and was
        // truncated. Signal it rather than handing back a silently-cut line.
        if (total == buf.len) return error.LineTooLong;
        // Otherwise the peer closed before sending a newline (incomplete frame);
        // return what arrived so the caller's parser rejects it.
        return std.mem.trimEnd(u8, buf[0..total], "\r");
    };
    return std.mem.trimEnd(u8, buf[0..end], "\r");
}
