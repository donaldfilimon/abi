//! Lightweight tracing helpers for correlating requests across logs and metrics.

const std = @import("std");

pub const TraceContext = struct {
    trace_id: [16]u8,
    span_id: [8]u8,
    start_ns: u64,

    pub fn init(random: std.rand.Random) TraceContext {
        var trace = TraceContext{
            .trace_id = undefined,
            .span_id = undefined,
            .start_ns = std.time.nanoTimestamp(),
        };
        random.bytes(&trace.trace_id);
        random.bytes(&trace.span_id);
        return trace;
    }

    pub fn child(self: TraceContext, random: std.rand.Random) TraceContext {
        var child_ctx = TraceContext{
            .trace_id = self.trace_id,
            .span_id = undefined,
            .start_ns = std.time.nanoTimestamp(),
        };
        random.bytes(&child_ctx.span_id);
        return child_ctx;
    }

    pub fn traceHex(self: TraceContext) [32]u8 {
        var buf: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "{s}", .{std.fmt.fmtSliceHexLower(&self.trace_id)}) catch unreachable;
        return buf;
    }

    pub fn spanHex(self: TraceContext) [16]u8 {
        var buf: [16]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "{s}", .{std.fmt.fmtSliceHexLower(&self.span_id)}) catch unreachable;
        return buf;
    }

    pub fn elapsedNs(self: TraceContext) u64 {
        return std.time.nanoTimestamp() - self.start_ns;
    }
};

pub fn formatTraceId(trace: TraceContext, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}", .{std.fmt.fmtSliceHexLower(&trace.trace_id)});
}

pub fn formatSpanId(trace: TraceContext, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}", .{std.fmt.fmtSliceHexLower(&trace.span_id)});
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "trace context generates ids" {
    var prng = std.rand.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();
    const ctx = TraceContext.init(random);
    const child = ctx.child(random);
    try std.testing.expect(!std.mem.eql(u8, &ctx.span_id, &child.span_id));
    try std.testing.expectEqualSlices(u8, &ctx.trace_id, &child.trace_id);
}
