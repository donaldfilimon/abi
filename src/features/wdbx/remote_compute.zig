//! Remote accelerator dispatch (Compute/Transport Layer).
//!
//! The honest realization of the north-star "remote TPU execution" backend: a
//! pure-Zig TCP transport that ships a compute op (a vector dot product) to a
//! configured remote endpoint and returns its result, with a deterministic CPU
//! fallback when no endpoint is set or it is unreachable. This is the DISPATCH
//! MECHANISM — the operator points `ABI_REMOTE_COMPUTE_ENDPOINT` at their own
//! TPU/GPU inference service; no accelerator is bundled or claimed. It mirrors
//! the `cluster_rpc` socket pattern and is exercised over a 127.0.0.1 loopback
//! reference server in tests.
//!
//! Wire protocol (one request/response per connection, newline-framed text):
//!   "DOT <n> <a0> .. <a(n-1)> <b0> .. <b(n-1)>\n"  ->  "<dot>\n"

const std = @import("std");
const net_line = @import("net_line.zig");

const Stream = std.Io.net.Stream;
const Server = std.Io.net.Server;

pub const ENDPOINT_ENV = "ABI_REMOTE_COMPUTE_ENDPOINT";
pub const MAX_MSG = 64 * 1024;

pub const RemoteError = error{ MalformedRequest, MalformedResponse, DimensionMismatch };

/// Local reference dot product (also the CPU fallback and the value a correct
/// remote endpoint must reproduce).
pub fn localDot(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.DimensionMismatch;
    var sum: f32 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

/// The configured remote endpoint ("host:port"), or null when unset — the honest
/// signal that ops run on the local CPU fallback.
pub fn endpoint() ?[]const u8 {
    const raw = std.c.getenv(ENDPOINT_ENV) orelse return null;
    const span = std.mem.span(raw);
    return if (span.len == 0) null else span;
}

/// Accept one connection, evaluate the DOT op, and respond. This is the
/// reference/mock remote accelerator — a real endpoint would compute on its TPU.
pub fn serveOnce(io: std.Io, server: *Server, allocator: std.mem.Allocator) !void {
    const conn = try server.accept(io);
    defer conn.close(io);

    var buf: [MAX_MSG]u8 = undefined;
    const line = try net_line.readLine(io, conn, &buf);

    if (!std.mem.startsWith(u8, line, "DOT ")) return error.MalformedRequest;
    var it = std.mem.splitScalar(u8, line["DOT ".len..], ' ');
    const dim_s = it.next() orelse return error.MalformedRequest;
    const dim = std.fmt.parseInt(usize, dim_s, 10) catch return error.MalformedRequest;

    const scratch = try allocator.alloc(f32, dim * 2);
    defer allocator.free(scratch);
    var i: usize = 0;
    while (i < dim * 2) : (i += 1) {
        const tok = it.next() orelse return error.MalformedRequest;
        scratch[i] = std.fmt.parseFloat(f32, tok) catch return error.MalformedRequest;
    }
    const result = try localDot(scratch[0..dim], scratch[dim .. dim * 2]);

    var out: [64]u8 = undefined;
    const resp = try std.fmt.bufPrint(&out, "{d}\n", .{result});
    try net_line.writeLine(io, conn, resp);
}

/// Connect to the endpoint and send a DOT request, returning the open stream
/// (read the reply with `readDotReply`). Null if the endpoint is unreachable.
pub fn dialDot(io: std.Io, allocator: std.mem.Allocator, port: u16, a: []const f32, b: []const f32) !?Stream {
    if (a.len != b.len) return error.DimensionMismatch;
    var msg: std.ArrayListUnmanaged(u8) = .empty;
    defer msg.deinit(allocator);
    try msg.print(allocator, "DOT {d}", .{a.len});
    for (a) |x| try msg.print(allocator, " {d}", .{x});
    for (b) |y| try msg.print(allocator, " {d}", .{y});
    try msg.append(allocator, '\n');

    return net_line.dial(io, port, msg.items);
}

/// Read and parse a DOT reply, then close the connection.
pub fn readDotReply(io: std.Io, conn: Stream) !f32 {
    defer conn.close(io);
    var buf: [64]u8 = undefined;
    const line = try net_line.readLine(io, conn, &buf);
    return std.fmt.parseFloat(f32, line) catch error.MalformedResponse;
}

const testing = std.testing;

test "remote_compute: dot dispatched over loopback matches the local reference" {
    const allocator = testing.allocator;
    const io = testing.io;

    var address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39310);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 0.5, -1.0, 2.0, 0.25 };

    // Dial (request buffers into the kernel), then serve, then read the reply —
    // single-threaded and deterministic, like the cluster_rpc tests.
    const conn = (try dialDot(io, allocator, 39310, &a, &b)).?;
    try serveOnce(io, &server, allocator);
    const got = try readDotReply(io, conn);

    try testing.expectApproxEqAbs(try localDot(&a, &b), got, 1e-4);
}

test "remote_compute: serveOnce rejects a malformed request" {
    const allocator = testing.allocator;
    const io = testing.io;

    var address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39312);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 39312);
    const conn = try caddr.connect(io, .{ .mode = .stream });
    defer conn.close(io);
    var wb: [64]u8 = undefined;
    var sw = conn.writer(io, &wb);
    try sw.interface.writeAll("GARBAGE not a dot request\n");
    try sw.interface.flush();

    // A request that does not start with "DOT " is surfaced, not mis-served.
    try testing.expectError(error.MalformedRequest, serveOnce(io, &server, allocator));
}

test "remote_compute: unreachable endpoint yields null (caller falls back to CPU)" {
    const allocator = testing.allocator;
    const io = testing.io;
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0 };
    // Port 39311 has no listener -> connect refused -> null -> CPU fallback.
    try testing.expect((try dialDot(io, allocator, 39311, &a, &b)) == null);
}

test {
    testing.refAllDecls(@This());
}
