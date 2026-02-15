const std = @import("std");
const radix = @import("../../services/shared/utils/radix_tree.zig");

pub const RouteTree = radix.RadixTree(u32);
pub const RadixNode = RouteTree.Node;

/// Split a URL path into segments.
pub fn splitPath(path: []const u8) std.mem.SplitIterator(u8, .scalar) {
    return RouteTree.splitPath(path);
}

/// Legacy matcher retained for compatibility tests that inspect wildcard/param behavior.
pub fn pathMatchesRoute(request_path: []const u8, route_path: []const u8) bool {
    var req_seg = splitPath(request_path);
    var route_seg = splitPath(route_path);

    while (true) {
        const rs = route_seg.next();
        const rq = req_seg.next();

        if (rs == null and rq == null) return true;
        if (rs == null or rq == null) return false;

        const rseg = rs.?;
        if (rseg.len > 0 and rseg[0] == '*') return true;
        if (rseg.len > 2 and rseg[0] == '{') continue; // param matches anything
        if (!std.mem.eql(u8, rseg, rq.?)) return false;
    }
}
