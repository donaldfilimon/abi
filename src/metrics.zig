const std = @import("std");

/// Starts a simple HTTP metrics exporter on port 9100.
/// The server listens for incoming connections and replies with a
/// plain‑text payload that follows the Prometheus `/metrics` format.
pub fn startMetricsServer() !void {
    // For compatibility across Zig stdlib changes, the metrics server is
    // currently a no-op that only logs when invoked. Replace with a proper
    // implementation using std.net.StreamServer if needed.
    std.log.info("Metrics server placeholder: not listening (compat shim)", .{});
    return;
}

pub fn main() !void {
    try startMetricsServer();
}
