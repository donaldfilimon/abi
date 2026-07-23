const std = @import("std");
const headers = @import("headers.zig");

pub fn hasBearerToken(raw: []const u8, token: []const u8) bool {
    const value = headers.headerValue(raw, "Authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, value, prefix)) return false;
    return fixedWorkEql(value[prefix.len..], token);
}

/// Length-independent equality used for bearer / shared-secret compares.
/// Same algorithm as `cluster_rpc.fixedWorkEql` (kept local so foundation
/// does not import WDBX).
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

test "Authorization bearer parser" {
    const raw =
        "POST / HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST / HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST / HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "local-tok"));
    try std.testing.expect(!hasBearerToken(raw, "local-token-extra"));
}

test "fixedWorkEql matches equal and rejects unequal" {
    try std.testing.expect(fixedWorkEql("abc", "abc"));
    try std.testing.expect(!fixedWorkEql("abc", "abd"));
    try std.testing.expect(!fixedWorkEql("abc", "ab"));
    try std.testing.expect(fixedWorkEql("", ""));
}

test {
    std.testing.refAllDecls(@This());
}
