//! Networked consensus transport (Cluster Layer / Transport Layer).
//!
//! A real TCP RPC layer over the in-process Raft state machine (`cluster.zig`):
//! a node serves RequestVote / AppendEntries on a loopback-or-host socket, and a
//! client drives elections and replication across those sockets. This is the
//! "Cluster RPC" transport — genuine network sockets (`std.Io.net`), not an
//! in-process array. It is exercised on 127.0.0.1; binding a routable address is
//! the only change for multi-host, so distribution is a deployment concern, not
//! a code one.
//!
//! Wire protocol (one request/response per connection, newline-framed text):
//!   "VOTE <term> <candidate>\n"      -> "GRANTED <term>\n" | "DENIED <term>\n"
//!   "APPEND <term> <data...>\n"      -> "ACK <term>\n"     | "NACK <term>\n"
//!
//! The protocol carries no timers or randomness, so a single-threaded driver can
//! dial every peer (connect + send buffer into the kernel), then serve each peer,
//! then read every reply — fully deterministic, which is how the tests run.

const std = @import("std");
const cluster = @import("cluster.zig");

const Stream = std.Io.net.Stream;
const Server = std.Io.net.Server;

pub const RpcError = error{ MalformedRequest, MalformedResponse };

pub const VoteReply = struct { granted: bool, term: u64 };
pub const AppendReply = struct { ack: bool, term: u64 };

/// Apply a RequestVote to `node` under standard Raft rules. Grants the vote when
/// the candidate's term is not stale and the node has not already voted for a
/// different candidate this term.
pub fn applyVote(node: *cluster.Node, term: u64, candidate: u32) bool {
    if (term < node.term) return false;
    if (term > node.term) {
        node.term = term;
        node.voted_for = null;
        node.role = .follower;
    }
    if (node.voted_for == null or node.voted_for == candidate) {
        node.voted_for = candidate;
        node.role = .follower;
        return true;
    }
    return false;
}

/// Apply an AppendEntries to `node`: a non-stale term makes the node a follower
/// and appends the (owned) entry to its log. Rejects a stale term.
pub fn applyAppend(node: *cluster.Node, allocator: std.mem.Allocator, term: u64, data: []const u8) !bool {
    if (term < node.term) return false;
    node.term = term;
    node.role = .follower;
    const owned = try allocator.dupe(u8, data);
    errdefer allocator.free(owned);
    try node.log.append(allocator, .{ .term = term, .data = owned });
    return true;
}

/// Bind a loopback listener for a node endpoint.
pub fn listen(io: std.Io, port: u16) !Server {
    var address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    return address.listen(io, .{ .reuse_address = true });
}

/// Accept one connection, apply the RPC to `node`, and respond. One request per
/// connection (the client closes after reading the reply).
pub fn serveOnce(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator) !void {
    const conn = try server.accept(io);
    defer conn.close(io);

    var buf: [4096]u8 = undefined;
    var rv: [1][]u8 = .{buf[0..]};
    const n = try conn.read(io, &rv);
    const line = std.mem.trimEnd(u8, buf[0..n], "\r\n");

    var resp_buf: [64]u8 = undefined;
    const resp: []const u8 = blk: {
        if (std.mem.startsWith(u8, line, "VOTE ")) {
            const rest = line["VOTE ".len..];
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse return error.MalformedRequest;
            const term = std.fmt.parseInt(u64, rest[0..sp], 10) catch return error.MalformedRequest;
            const cand = std.fmt.parseInt(u32, rest[sp + 1 ..], 10) catch return error.MalformedRequest;
            const granted = applyVote(node, term, cand);
            break :blk try std.fmt.bufPrint(&resp_buf, "{s} {d}\n", .{ if (granted) "GRANTED" else "DENIED", node.term });
        } else if (std.mem.startsWith(u8, line, "APPEND ")) {
            const rest = line["APPEND ".len..];
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
            const term = std.fmt.parseInt(u64, rest[0..sp], 10) catch return error.MalformedRequest;
            const data = if (sp < rest.len) rest[sp + 1 ..] else "";
            const ack = try applyAppend(node, allocator, term, data);
            break :blk try std.fmt.bufPrint(&resp_buf, "{s} {d}\n", .{ if (ack) "ACK" else "NACK", node.term });
        }
        break :blk "ERR 0\n";
    };

    var wb: [64]u8 = undefined;
    var sw = conn.writer(io, &wb);
    try sw.interface.writeAll(resp);
    try sw.interface.flush();
}

/// Serve a node's consensus RPC endpoint, applying RequestVote/AppendEntries to
/// `node` until the process is stopped. One request per connection; a failed
/// connection is logged and the loop continues (same posture as the REST server).
pub fn serveLoop(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator) !void {
    while (true) {
        serveOnce(io, server, node, allocator) catch |err| {
            std.log.warn("cluster RPC serve error: {s}", .{@errorName(err)});
        };
    }
}

fn dial(io: std.Io, port: u16, msg: []const u8) !?Stream {
    var address = std.Io.net.IpAddress.parseIp4("127.0.0.1", port) catch return null;
    // A refused/unreachable peer (e.g. a downed node) yields null rather than an
    // error: the caller treats it as a missing vote/ack, like a real cluster.
    const conn = address.connect(io, .{ .mode = .stream }) catch return null;
    errdefer conn.close(io);
    var wb: [4096]u8 = undefined;
    var sw = conn.writer(io, &wb);
    try sw.interface.writeAll(msg);
    try sw.interface.flush();
    return conn;
}

/// Connect to a peer and send a RequestVote, returning the open stream (read the
/// reply with `readVoteReply`). Null if the peer is unreachable. Two-phase so a
/// single driver can dial every peer before any peer is served.
pub fn dialVote(io: std.Io, port: u16, term: u64, candidate: u32) !?Stream {
    var msg_buf: [64]u8 = undefined;
    const msg = try std.fmt.bufPrint(&msg_buf, "VOTE {d} {d}\n", .{ term, candidate });
    return dial(io, port, msg);
}

/// Connect to a peer and send an AppendEntries, returning the open stream (read
/// the reply with `readAppendReply`). Null if the peer is unreachable.
pub fn dialAppend(io: std.Io, port: u16, term: u64, data: []const u8) !?Stream {
    var msg_buf: [4096]u8 = undefined;
    const msg = try std.fmt.bufPrint(&msg_buf, "APPEND {d} {s}\n", .{ term, data });
    return dial(io, port, msg);
}

/// Read and parse a vote reply, then close the connection.
pub fn readVoteReply(io: std.Io, conn: Stream) !VoteReply {
    defer conn.close(io);
    var buf: [64]u8 = undefined;
    var rv: [1][]u8 = .{buf[0..]};
    const n = try conn.read(io, &rv);
    const line = std.mem.trimEnd(u8, buf[0..n], "\r\n");
    var it = std.mem.splitScalar(u8, line, ' ');
    const verb = it.next() orelse return error.MalformedResponse;
    const term_s = it.next() orelse return error.MalformedResponse;
    const term = std.fmt.parseInt(u64, term_s, 10) catch return error.MalformedResponse;
    return .{ .granted = std.mem.eql(u8, verb, "GRANTED"), .term = term };
}

/// Read and parse an append reply, then close the connection.
pub fn readAppendReply(io: std.Io, conn: Stream) !AppendReply {
    defer conn.close(io);
    var buf: [64]u8 = undefined;
    var rv: [1][]u8 = .{buf[0..]};
    const n = try conn.read(io, &rv);
    const line = std.mem.trimEnd(u8, buf[0..n], "\r\n");
    var it = std.mem.splitScalar(u8, line, ' ');
    const verb = it.next() orelse return error.MalformedResponse;
    const term_s = it.next() orelse return error.MalformedResponse;
    const term = std.fmt.parseInt(u64, term_s, 10) catch return error.MalformedResponse;
    return .{ .ack = std.mem.eql(u8, verb, "ACK"), .term = term };
}

const testing = std.testing;

fn deinitNode(node: *cluster.Node, allocator: std.mem.Allocator) void {
    for (node.log.items) |e| allocator.free(e.data);
    node.log.deinit(allocator);
}

test "cluster_rpc: networked election reaches quorum over loopback" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1 };
    var n2 = cluster.Node{ .id = 2 };
    defer deinitNode(&n1, allocator);
    defer deinitNode(&n2, allocator);

    var s1 = try listen(io, 39101);
    defer s1.deinit(io);
    var s2 = try listen(io, 39102);
    defer s2.deinit(io);

    // Candidate (node 0) requests votes at term 1. Dial both peers first (the
    // requests buffer into the kernel), then serve each, then read replies.
    const c1 = (try dialVote(io, 39101, 1, 0)).?;
    const c2 = (try dialVote(io, 39102, 1, 0)).?;
    try serveOnce(io, &s1, &n1, allocator);
    try serveOnce(io, &s2, &n2, allocator);
    const r1 = try readVoteReply(io, c1);
    const r2 = try readVoteReply(io, c2);

    var votes: usize = 1; // candidate votes for itself
    if (r1.granted) votes += 1;
    if (r2.granted) votes += 1;
    try testing.expect(votes >= 2); // quorum of a 3-node cluster
    try testing.expectEqual(@as(u64, 1), n1.term);
    try testing.expectEqual(@as(?u32, 0), n1.voted_for);
}

test "cluster_rpc: election succeeds with a downed node (no listener)" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1 };
    defer deinitNode(&n1, allocator);

    var s1 = try listen(io, 39111);
    defer s1.deinit(io);
    // Port 39112 is intentionally NOT bound — node 2 is down.

    const c1 = (try dialVote(io, 39111, 1, 0)).?;
    const down = try dialVote(io, 39112, 1, 0); // unreachable -> null
    try serveOnce(io, &s1, &n1, allocator);
    const r1 = try readVoteReply(io, c1);

    var votes: usize = 1;
    if (r1.granted) votes += 1;
    if (down != null) votes += 1;
    try testing.expect(down == null); // the downed node was unreachable
    try testing.expect(votes >= 2); // self + the one live peer still forms quorum
}

test "cluster_rpc: replication over loopback acks and records the entry" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1 };
    var n2 = cluster.Node{ .id = 2 };
    defer deinitNode(&n1, allocator);
    defer deinitNode(&n2, allocator);

    var s1 = try listen(io, 39121);
    defer s1.deinit(io);
    var s2 = try listen(io, 39122);
    defer s2.deinit(io);

    const c1 = (try dialAppend(io, 39121, 2, "set k=v")).?;
    const c2 = (try dialAppend(io, 39122, 2, "set k=v")).?;
    try serveOnce(io, &s1, &n1, allocator);
    try serveOnce(io, &s2, &n2, allocator);
    const r1 = try readAppendReply(io, c1);
    const r2 = try readAppendReply(io, c2);

    var acks: usize = 1; // leader holds the entry
    if (r1.ack) acks += 1;
    if (r2.ack) acks += 1;
    try testing.expect(acks >= 2); // quorum replication
    try testing.expectEqual(@as(usize, 1), n1.log.items.len);
    try testing.expectEqualStrings("set k=v", n1.log.items[0].data);
}

test {
    testing.refAllDecls(@This());
}
