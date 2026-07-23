//! Cluster auth + peer allowlist policy (Approach-1 leaf from cluster_rpc).
//! Not production multi-host membership, sharding, or mTLS.

const std = @import("std");

pub const ClusterAuth = struct {
    token: ?[]const u8 = null,

    pub fn enabled(self: ClusterAuth) bool {
        return self.token != null;
    }
};

pub const ClusterPolicy = struct {
    auth: ClusterAuth = .{},
    peers: ?[]const u32 = null,

    pub fn allowsPeer(self: ClusterPolicy, id: u32) bool {
        const peers = self.peers orelse return true;
        for (peers) |peer| {
            if (peer == id) return true;
        }
        return false;
    }

    /// Return a new policy with an updated peer allowlist. This is the
    /// membership-reload primitive: a server loop can call `withPeers` on a
    /// fresh peer list and pass the new policy to the next `serveOnceAuth`,
    /// so a node admitted or rejected at startup can be reconfigured without
    /// restarting the listener. Single-host / loopback-tested; NOT production
    /// dynamic membership, NOT sharding, NOT mTLS.
    pub fn withPeers(self: ClusterPolicy, new_peers: ?[]const u32) ClusterPolicy {
        return .{ .auth = self.auth, .peers = new_peers };
    }
};

/// Length-independent equality used for shared-secret compares.
pub fn fixedWorkEql(a: []const u8, b: []const u8) bool {
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

pub fn authMatches(auth: ClusterAuth, supplied: ?[]const u8) bool {
    const expected = auth.token orelse return supplied == null;
    const got = supplied orelse return false;
    return fixedWorkEql(expected, got);
}

test "cluster_policy: null peer allowlist permits any peer id" {
    const open = ClusterPolicy{ .auth = .{ .token = "t" }, .peers = null };
    try std.testing.expect(open.allowsPeer(99));
    const closed = ClusterPolicy{ .auth = .{ .token = "t" }, .peers = &[_]u32{ 1, 2 } };
    try std.testing.expect(closed.allowsPeer(1));
    try std.testing.expect(!closed.allowsPeer(99));
}

test "cluster_policy: withPeers reloads allowlist" {
    var policy = ClusterPolicy{ .auth = .{ .token = "t" }, .peers = &[_]u32{1} };
    try std.testing.expect(!policy.allowsPeer(2));
    policy = policy.withPeers(&[_]u32{ 1, 2 });
    try std.testing.expect(policy.allowsPeer(2));
}

test {
    std.testing.refAllDecls(@This());
}
