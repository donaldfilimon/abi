//! Server Request Handlers
//!
//! HTTP endpoint handler implementations and connection context types
//! for the streaming inference server.

const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const build_options = @import("build_options");
const observability = if (build_options.feat_observability) @import("../../../observability/mod.zig") else @import("../../../observability/stub.zig");
const backends = @import("../backends/mod.zig");
const formats = @import("../formats/mod.zig");
const mod = @import("../mod.zig");
const sse = @import("../sse.zig");
const websocket = @import("../websocket.zig");
const recovery = @import("../recovery.zig");
const request_types = @import("../request_types.zig");
const helpers = @import("helpers.zig");

pub const AbiStreamRequest = request_types.AbiStreamRequest;
const parseAbiStreamRequest = request_types.parseAbiStreamRequest;
const extractJsonString = request_types.extractJsonString;
const findHeaderInBuffer = helpers.findHeaderInBuffer;

/// Connection context for streaming responses
pub const ConnectionContext = struct {
    io: std.Io,
    stream: std.Io.net.Stream,
    send_buffer: *[8192]u8,

    /// Write raw bytes to connection using the writer interface
    pub fn write(self: *ConnectionContext, data: []const u8) !void {
        // Get writer and write all bytes (handles partial writes)
        var writer = self.stream.writer(self.io, self.send_buffer);
        try writer.interface.writeAll(data);
    }

    /// Flush is a no-op for stream writers (writeAll ensures delivery)
    pub fn flush(_: *ConnectionContext) !void {
        // writeAll ensures all bytes are written, no explicit flush needed
    }

    /// Write SSE headers to start streaming response
    pub fn writeSseHeaders(self: *ConnectionContext) !void {
        const headers =
            "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: text/event-stream\r\n" ++
            "Cache-Control: no-cache\r\n" ++
            "Connection: keep-alive\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "\r\n";
        try self.write(headers);
    }
};

/// Compute percentile value from a histogram.
pub fn histogramPercentile(hist: *const observability.Histogram, p: f64) u64 {
    if (hist.bounds.len == 0) return 0;

    var total: u64 = 0;
    for (hist.buckets) |count| total += count;
    if (total == 0) return 0;

    const target = @as(u64, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(total)) * p)));
    var cumulative: u64 = 0;
    for (hist.buckets, 0..) |count, i| {
        cumulative += count;
        if (cumulative >= target) {
            if (i < hist.bounds.len) return hist.bounds[i];
            return hist.bounds[hist.bounds.len - 1];
        }
    }

    return hist.bounds[hist.bounds.len - 1];
}


test {
    std.testing.refAllDecls(@This());
}
