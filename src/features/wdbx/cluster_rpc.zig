//! Networked consensus transport (Cluster Layer / Transport Layer).
//!
//! A real TCP RPC layer over the in-process Raft state machine (`cluster.zig`):
//! a node serves RequestVote / AppendEntries on a loopback-or-host socket, and a
//! client drives elections and replication across those sockets. This is the
//! "Cluster RPC" transport — genuine network sockets (`std.Io.net`), not an
//! in-process array. Both bind and dial are host-aware (`listenAddr` /
//! `dialVoteAddr` / `dialAppendAddr` take an explicit IPv4/IPv6 host, with
//! `listen`/`dialVote`/`dialAppend` as loopback convenience wrappers), so a node
//! can bind a routable address ("0.0.0.0" or a specific NIC) and peers on other
//! hosts can reach it. Multi-host clustering is then a deployment concern (choose
//! the bind host + reachable peer addresses); the transport code is routable.
//!
//! Wire protocol (one request/response per connection, newline-framed text):
//!   "VOTE <term> <candidate>\n"      -> "GRANTED <term>\n" | "DENIED <term>\n"
//!   "APPEND <term> <data...>\n"      -> "ACK <term>\n"     | "NACK <term>\n"
//!   "AUTH <token> VOTE <term> <candidate>\n"
//!   "AUTH <token> APPEND <term> <leader> <data...>\n"
//!
//! The protocol carries no timers or randomness, so a single-threaded driver can
//! dial every peer (connect + send buffer into the kernel), then serve each peer,
//! then read every reply — fully deterministic, which is how the tests run.

const std = @import("std");
const cluster = @import("cluster.zig");
const net_line = @import("net_line.zig");

pub const VoteReply = cluster.VoteReply;
pub const AppendReply = cluster.AppendReply;

const Stream = std.Io.net.Stream;
const Server = std.Io.net.Server;

pub const RpcError = error{ MalformedRequest, MalformedResponse };

pub const ClusterAuth = struct {
    token: ?[]const u8 = null,

    pub fn enabled(self: ClusterAuth) bool {
        return self.token != null;
    }
};

pub const ClusterPolicy = struct {
    auth: ClusterAuth = .{},
    peers: ?[]const u32 = null,

    fn allowsPeer(self: ClusterPolicy, id: u32) bool {
        const peers = self.peers orelse return true;
        for (peers) |peer| {
            if (peer == id) return true;
        }
        return false;
    }
};

const ParsedRequest = union(enum) {
    vote: struct { term: u64, candidate: u32 },
    append: struct { term: u64, leader: ?u32, data: []const u8 },
    unknown,
};

fn fixedWorkEql(a: []const u8, b: []const u8) bool {
    const max_len = @max(a.len, b.len);
    var diff: usize = a.len ^ b.len;
    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        const av: u8 = if (i < a.len) a[i] else 0;
        const bv: u8 = if (i < b.len) b[i] else 0;
        diff |= av ^ bv;
    }
    return diff == 0;
}

fn authMatches(auth: ClusterAuth, supplied: ?[]const u8) bool {
    const expected = auth.token orelse return supplied == null;
    const got = supplied orelse return false;
    return fixedWorkEql(expected, got);
}

fn parseVote(rest: []const u8) !ParsedRequest {
    const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse return error.MalformedRequest;
    const term = std.fmt.parseInt(u64, rest[0..sp], 10) catch return error.MalformedRequest;
    const cand = std.fmt.parseInt(u32, rest[sp + 1 ..], 10) catch return error.MalformedRequest;
    return .{ .vote = .{ .term = term, .candidate = cand } };
}

fn parseAppend(rest: []const u8, leader_required: bool) !ParsedRequest {
    const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
    const term = std.fmt.parseInt(u64, rest[0..sp], 10) catch return error.MalformedRequest;
    if (!leader_required) {
        const data = if (sp < rest.len) rest[sp + 1 ..] else "";
        return .{ .append = .{ .term = term, .leader = null, .data = data } };
    }
    if (sp >= rest.len) return .{ .append = .{ .term = term, .leader = null, .data = "" } };
    const leader_rest = rest[sp + 1 ..];
    const leader_sp = std.mem.indexOfScalar(u8, leader_rest, ' ') orelse leader_rest.len;
    const leader = std.fmt.parseInt(u32, leader_rest[0..leader_sp], 10) catch
        return .{ .append = .{ .term = term, .leader = null, .data = leader_rest } };
    const data = if (leader_sp < leader_rest.len) leader_rest[leader_sp + 1 ..] else "";
    return .{ .append = .{ .term = term, .leader = leader, .data = data } };
}

fn parseRequest(line: []const u8, auth_supplied: *?[]const u8) !ParsedRequest {
    auth_supplied.* = null;
    if (std.mem.startsWith(u8, line, "AUTH ")) {
        const rest = line["AUTH ".len..];
        const token_sp = std.mem.indexOfScalar(u8, rest, ' ') orelse return error.MalformedRequest;
        auth_supplied.* = rest[0..token_sp];
        const after_token = rest[token_sp + 1 ..];
        if (std.mem.startsWith(u8, after_token, "VOTE ")) {
            return parseVote(after_token["VOTE ".len..]);
        }
        if (std.mem.startsWith(u8, after_token, "APPEND ")) {
            return parseAppend(after_token["APPEND ".len..], true);
        }
        return .unknown;
    }
    if (std.mem.startsWith(u8, line, "VOTE ")) {
        return parseVote(line["VOTE ".len..]);
    }
    if (std.mem.startsWith(u8, line, "APPEND ")) {
        return parseAppend(line["APPEND ".len..], false);
    }
    return .unknown;
}

/// Bind a node endpoint on an explicit host address — the routable-cluster entry
/// point. `host` accepts loopback ("127.0.0.1"), all-interfaces ("0.0.0.0" /
/// "::"), or a specific routable IPv4/IPv6 address, so a node can be reached by
/// peers on other hosts. Multi-host clustering is then purely a deployment
/// concern (choose the bind host + reachable peer addresses).
pub fn listenAddr(io: std.Io, host: []const u8, port: u16) !Server {
    var address = try net_line.resolveHost(host, port);
    return address.listen(io, .{ .reuse_address = true });
}

/// Loopback convenience: `listenAddr(io, "127.0.0.1", port)`. Preserved for
/// single-host use and tests.
pub fn listen(io: std.Io, port: u16) !Server {
    return listenAddr(io, "127.0.0.1", port);
}

/// Accept one connection, apply the RPC to `node`, and respond. One request per
/// connection (the client closes after reading the reply).
pub fn serveOnce(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator) !void {
    return serveOnceAuth(io, server, node, allocator, .{});
}

/// Policy-aware form of `serveOnce`. Auth or peer-policy failures return a
/// normal DENIED/NACK response and intentionally leave the Raft node untouched.
pub fn serveOnceAuth(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator, policy: ClusterPolicy) !void {
    const conn = try server.accept(io);
    defer conn.close(io);

    var buf: [4096]u8 = undefined;
    const line = try net_line.readLine(io, conn, &buf);

    var resp_buf: [64]u8 = undefined;
    const resp: []const u8 = blk: {
        var supplied: ?[]const u8 = null;
        const parsed = try parseRequest(line, &supplied);
        switch (parsed) {
            .vote => |req| {
                if (!authMatches(policy.auth, supplied) or !policy.allowsPeer(req.candidate)) {
                    break :blk try std.fmt.bufPrint(&resp_buf, "DENIED {d}\n", .{node.term});
                }
                const granted = cluster.applyVote(node, req.term, req.candidate);
                break :blk try std.fmt.bufPrint(&resp_buf, "{s} {d}\n", .{ if (granted) "GRANTED" else "DENIED", node.term });
            },
            .append => |req| {
                if (!authMatches(policy.auth, supplied) or (supplied != null and req.leader == null) or (policy.peers != null and (req.leader == null or !policy.allowsPeer(req.leader.?)))) {
                    break :blk try std.fmt.bufPrint(&resp_buf, "NACK {d}\n", .{node.term});
                }
                const ack = try cluster.applyAppend(node, allocator, req.term, req.data);
                break :blk try std.fmt.bufPrint(&resp_buf, "{s} {d}\n", .{ if (ack) "ACK" else "NACK", node.term });
            },
            .unknown => break :blk "ERR 0\n",
        }
    };

    try net_line.writeLine(io, conn, resp);
}

/// Serve a node's consensus RPC endpoint, applying RequestVote/AppendEntries to
/// `node` until the process is stopped. One request per connection; a failed
/// connection is logged and the loop continues (same posture as the REST server).
pub fn serveLoop(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator) !void {
    return serveLoopAuth(io, server, node, allocator, .{});
}

/// Policy-aware form of `serveLoop`.
pub fn serveLoopAuth(io: std.Io, server: *Server, node: *cluster.Node, allocator: std.mem.Allocator, policy: ClusterPolicy) !void {
    while (true) {
        serveOnceAuth(io, server, node, allocator, policy) catch |err| {
            std.log.warn("cluster RPC serve error: {s}", .{@errorName(err)});
        };
    }
}

/// Connect to a peer and send a RequestVote, returning the open stream (read the
/// reply with `readVoteReply`). Null if the peer is unreachable. Two-phase so a
/// single driver can dial every peer before any peer is served.
pub fn dialVote(io: std.Io, port: u16, term: u64, candidate: u32) !?Stream {
    return dialVoteAddr(io, "127.0.0.1", port, term, candidate);
}

/// Routable RequestVote: dial a peer at an explicit `host` (loopback or any
/// reachable IPv4/IPv6) for multi-host clusters.
pub fn dialVoteAddr(io: std.Io, host: []const u8, port: u16, term: u64, candidate: u32) !?Stream {
    return dialVoteAddrAuth(io, host, port, term, candidate, null);
}

/// Authenticated RequestVote dialer. When `token` is null, sends the legacy
/// unauthenticated frame for local loopback demos and existing tests.
pub fn dialVoteAddrAuth(io: std.Io, host: []const u8, port: u16, term: u64, candidate: u32, token: ?[]const u8) !?Stream {
    var msg_buf: [64]u8 = undefined;
    const msg = if (token) |t|
        try std.fmt.bufPrint(&msg_buf, "AUTH {s} VOTE {d} {d}\n", .{ t, term, candidate })
    else
        try std.fmt.bufPrint(&msg_buf, "VOTE {d} {d}\n", .{ term, candidate });
    return net_line.dialAddr(io, host, port, msg);
}

/// Connect to a peer and send an AppendEntries, returning the open stream (read
/// the reply with `readAppendReply`). Null if the peer is unreachable.
pub fn dialAppend(io: std.Io, port: u16, term: u64, data: []const u8) !?Stream {
    return dialAppendAddr(io, "127.0.0.1", port, term, data);
}

/// Routable AppendEntries: dial a peer at an explicit `host` for multi-host
/// clusters.
pub fn dialAppendAddr(io: std.Io, host: []const u8, port: u16, term: u64, data: []const u8) !?Stream {
    return dialAppendAddrAuth(io, host, port, term, null, data, null);
}

/// Authenticated AppendEntries dialer. Authenticated frames include a leader id
/// so a peer allowlist can reject forged appenders before mutating the node.
pub fn dialAppendAddrAuth(io: std.Io, host: []const u8, port: u16, term: u64, leader: ?u32, data: []const u8, token: ?[]const u8) !?Stream {
    var msg_buf: [4096]u8 = undefined;
    const msg = if (token) |t| blk: {
        const l = leader orelse return error.MalformedRequest;
        break :blk try std.fmt.bufPrint(&msg_buf, "AUTH {s} APPEND {d} {d} {s}\n", .{ t, term, l, data });
    } else try std.fmt.bufPrint(&msg_buf, "APPEND {d} {s}\n", .{ term, data });
    return net_line.dialAddr(io, host, port, msg);
}

/// Read and parse a vote reply, then close the connection.
pub fn readVoteReply(io: std.Io, conn: Stream) !VoteReply {
    defer conn.close(io);
    var buf: [64]u8 = undefined;
    const line = try net_line.readLine(io, conn, &buf);
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
    const line = try net_line.readLine(io, conn, &buf);
    var it = std.mem.splitScalar(u8, line, ' ');
    const verb = it.next() orelse return error.MalformedResponse;
    const term_s = it.next() orelse return error.MalformedResponse;
    const term = std.fmt.parseInt(u64, term_s, 10) catch return error.MalformedResponse;
    return .{ .ack = std.mem.eql(u8, verb, "ACK"), .term = term };
}

pub const MultiNodeLoopResult = struct {
    node_count: usize,
    vote_quorum: bool,
    append_quorum: bool,
    votes: usize,
    append_acks: usize,
    logs_verified: usize,
};

/// Drive one deterministic local multi-node round: leader/candidate 0 asks every
/// loopback peer for a vote, then appends one entry to every peer with auth and
/// allowlist policy enforced. This is a test/runtime helper, not a production
/// membership or sharding loop.
pub fn runAuthenticatedLoopbackRound(
    allocator: std.mem.Allocator,
    io: std.Io,
    ports: []const u16,
    token: []const u8,
    data: []const u8,
) !MultiNodeLoopResult {
    const peer_count = ports.len;
    const total_nodes = peer_count + 1; // candidate/leader 0 plus served peers.
    const quorum = total_nodes / 2 + 1;

    var peer_ids = try allocator.alloc(u32, total_nodes);
    defer allocator.free(peer_ids);
    peer_ids[0] = 0;
    for (peer_ids[1..], 0..) |*id, i| id.* = @intCast(i + 1);
    const policy = ClusterPolicy{ .auth = .{ .token = token }, .peers = peer_ids };

    var nodes = try allocator.alloc(cluster.Node, peer_count);
    defer allocator.free(nodes);
    var nodes_initialized: usize = 0;
    defer {
        var i: usize = 0;
        while (i < nodes_initialized) : (i += 1) deinitNode(&nodes[i], allocator);
    }
    for (nodes, 0..) |*node, i| {
        node.* = .{ .id = @intCast(i + 1) };
        nodes_initialized += 1;
    }

    var servers = try allocator.alloc(Server, peer_count);
    defer allocator.free(servers);
    var servers_initialized: usize = 0;
    defer {
        var i: usize = 0;
        while (i < servers_initialized) : (i += 1) servers[i].deinit(io);
    }
    for (ports, 0..) |port, i| {
        servers[i] = try listen(io, port);
        servers_initialized += 1;
    }

    var vote_conns = try allocator.alloc(?Stream, peer_count);
    defer allocator.free(vote_conns);
    for (ports, 0..) |port, i| {
        vote_conns[i] = try dialVoteAddrAuth(io, "127.0.0.1", port, 1, 0, token);
    }

    var votes: usize = 1;
    for (vote_conns, 0..) |conn, i| {
        if (conn == null) continue;
        try serveOnceAuth(io, &servers[i], &nodes[i], allocator, policy);
    }
    for (vote_conns) |conn| {
        if (conn) |stream| {
            const reply = try readVoteReply(io, stream);
            if (reply.granted) votes += 1;
        }
    }

    var append_conns = try allocator.alloc(?Stream, peer_count);
    defer allocator.free(append_conns);
    for (ports, 0..) |port, i| {
        append_conns[i] = try dialAppendAddrAuth(io, "127.0.0.1", port, 2, 0, data, token);
    }

    var append_acks: usize = 1;
    for (append_conns, 0..) |conn, i| {
        if (conn == null) continue;
        try serveOnceAuth(io, &servers[i], &nodes[i], allocator, policy);
    }
    for (append_conns) |conn| {
        if (conn) |stream| {
            const reply = try readAppendReply(io, stream);
            if (reply.ack) append_acks += 1;
        }
    }

    var logs_verified: usize = 0;
    for (nodes) |node| {
        if (node.log.items.len == 1 and std.mem.eql(u8, node.log.items[0].data, data)) logs_verified += 1;
    }

    return .{
        .node_count = total_nodes,
        .vote_quorum = votes >= quorum,
        .append_quorum = append_acks >= quorum,
        .votes = votes,
        .append_acks = append_acks,
        .logs_verified = logs_verified,
    };
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

test "cluster_rpc: routable bind on 0.0.0.0 accepts a RequestVote over the host-aware path" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1 };
    defer deinitNode(&n1, allocator);

    // Bind all interfaces (the routable-cluster bind), then reach it over the
    // host-aware client path — proving multi-host bind/dial works end-to-end
    // (a peer on another host would dial this node's routable address the same
    // way; here the loopback route validates the 0.0.0.0 listener accepts it).
    var s1 = try listenAddr(io, "0.0.0.0", 39131);
    defer s1.deinit(io);

    const c1 = (try dialVoteAddr(io, "127.0.0.1", 39131, 1, 0)).?;
    try serveOnce(io, &s1, &n1, allocator);
    const r1 = try readVoteReply(io, c1);

    try testing.expect(r1.granted);
    try testing.expectEqual(@as(u64, 1), n1.term);
    try testing.expectEqual(@as(?u32, 0), n1.voted_for);
}

test "cluster_rpc: shared secret permits authenticated vote and append" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1 };
    var n2 = cluster.Node{ .id = 2 };
    defer deinitNode(&n1, allocator);
    defer deinitNode(&n2, allocator);

    var s1 = try listen(io, 39141);
    defer s1.deinit(io);
    var s2 = try listen(io, 39142);
    defer s2.deinit(io);

    const policy = ClusterPolicy{ .auth = .{ .token = "cluster-secret" } };
    const vote = (try dialVoteAddrAuth(io, "127.0.0.1", 39141, 1, 0, "cluster-secret")).?;
    try serveOnceAuth(io, &s1, &n1, allocator, policy);
    const vote_reply = try readVoteReply(io, vote);
    try testing.expect(vote_reply.granted);
    try testing.expectEqual(@as(?u32, 0), n1.voted_for);

    const append = (try dialAppendAddrAuth(io, "127.0.0.1", 39142, 2, 0, "set secure=true", "cluster-secret")).?;
    try serveOnceAuth(io, &s2, &n2, allocator, policy);
    const append_reply = try readAppendReply(io, append);
    try testing.expect(append_reply.ack);
    try testing.expectEqual(@as(usize, 1), n2.log.items.len);
    try testing.expectEqualStrings("set secure=true", n2.log.items[0].data);
}

test "cluster_rpc: missing or wrong shared secret rejects without mutating state" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1, .term = 5 };
    var n2 = cluster.Node{ .id = 2, .term = 7 };
    defer deinitNode(&n1, allocator);
    defer deinitNode(&n2, allocator);

    var s1 = try listen(io, 39151);
    defer s1.deinit(io);
    var s2 = try listen(io, 39152);
    defer s2.deinit(io);

    const policy = ClusterPolicy{ .auth = .{ .token = "cluster-secret" } };
    const missing = (try dialVote(io, 39151, 6, 0)).?;
    try serveOnceAuth(io, &s1, &n1, allocator, policy);
    const missing_reply = try readVoteReply(io, missing);
    try testing.expect(!missing_reply.granted);
    try testing.expectEqual(@as(u64, 5), n1.term);
    try testing.expectEqual(@as(?u32, null), n1.voted_for);

    const wrong = (try dialAppendAddrAuth(io, "127.0.0.1", 39152, 8, 0, "set forged=true", "wrong-secret")).?;
    try serveOnceAuth(io, &s2, &n2, allocator, policy);
    const wrong_reply = try readAppendReply(io, wrong);
    try testing.expect(!wrong_reply.ack);
    try testing.expectEqual(@as(u64, 7), n2.term);
    try testing.expectEqual(@as(usize, 0), n2.log.items.len);
}

test "cluster_rpc: peer allowlist rejects unknown candidates and leaders" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1, .term = 3 };
    var n2 = cluster.Node{ .id = 2, .term = 4 };
    defer deinitNode(&n1, allocator);
    defer deinitNode(&n2, allocator);

    var s1 = try listen(io, 39161);
    defer s1.deinit(io);
    var s2 = try listen(io, 39162);
    defer s2.deinit(io);

    const peers = [_]u32{ 0, 1, 2 };
    const policy = ClusterPolicy{ .auth = .{ .token = "cluster-secret" }, .peers = &peers };

    const bad_vote = (try dialVoteAddrAuth(io, "127.0.0.1", 39161, 5, 99, "cluster-secret")).?;
    try serveOnceAuth(io, &s1, &n1, allocator, policy);
    const bad_vote_reply = try readVoteReply(io, bad_vote);
    try testing.expect(!bad_vote_reply.granted);
    try testing.expectEqual(@as(u64, 3), n1.term);
    try testing.expectEqual(@as(?u32, null), n1.voted_for);

    const bad_append = (try dialAppendAddrAuth(io, "127.0.0.1", 39162, 6, 99, "set forged=true", "cluster-secret")).?;
    try serveOnceAuth(io, &s2, &n2, allocator, policy);
    const bad_append_reply = try readAppendReply(io, bad_append);
    try testing.expect(!bad_append_reply.ack);
    try testing.expectEqual(@as(u64, 4), n2.term);
    try testing.expectEqual(@as(usize, 0), n2.log.items.len);
}

test "cluster_rpc: authenticated append requires an explicit leader id" {
    const allocator = testing.allocator;
    const io = testing.io;

    var n1 = cluster.Node{ .id = 1, .term = 9 };
    defer deinitNode(&n1, allocator);

    var s1 = try listen(io, 39171);
    defer s1.deinit(io);

    const peers = [_]u32{ 0, 1, 2 };
    const policy = ClusterPolicy{ .auth = .{ .token = "cluster-secret" }, .peers = &peers };
    const missing_leader = (try net_line.dial(io, 39171, "AUTH cluster-secret APPEND 10 set missing-leader=true\n")).?;
    try serveOnceAuth(io, &s1, &n1, allocator, policy);
    const reply = try readAppendReply(io, missing_leader);
    try testing.expect(!reply.ack);
    try testing.expectEqual(@as(u64, 9), n1.term);
    try testing.expectEqual(@as(usize, 0), n1.log.items.len);
}

test "cluster_rpc: authenticated multi-node loop reaches quorum and verifies peer logs" {
    const ports = [_]u16{ 39181, 39182, 39183 };
    const result = try runAuthenticatedLoopbackRound(testing.allocator, testing.io, &ports, "cluster-secret", "set loop=true");

    try testing.expectEqual(@as(usize, 4), result.node_count);
    try testing.expect(result.vote_quorum);
    try testing.expect(result.append_quorum);
    try testing.expectEqual(@as(usize, 4), result.votes);
    try testing.expectEqual(@as(usize, 4), result.append_acks);
    try testing.expectEqual(@as(usize, ports.len), result.logs_verified);
}

test {
    testing.refAllDecls(@This());
}
