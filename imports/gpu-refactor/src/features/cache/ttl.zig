const std = @import("std");
const time = @import("../../foundation/mod.zig").time;
const state_mod = @import("state.zig");

const CacheState = state_mod.CacheState;
const InternalEntry = state_mod.InternalEntry;
const NodeIndex = state_mod.NodeIndex;
const SENTINEL = state_mod.SENTINEL;

/// Check whether an entry has exceeded its TTL.
pub fn isExpired(entry: *const InternalEntry) bool {
    if (entry.ttl_ms == 0) return false;
    const now_ns = (time.Instant.now() catch return false).nanos;
    const ttl_ns = @as(u128, entry.ttl_ms) * std.time.ns_per_ms;
    return now_ns > entry.created_at_ns + ttl_ns;
}

/// Walk the eviction list from tail, removing up to `max_check` expired entries.
pub fn expireSweep(s: *CacheState, max_check: u32) u32 {
    var expired_count: u32 = 0;
    var cur = s.list_tail;
    var checked: u32 = 0;

    while (cur != SENTINEL and checked < max_check) : (checked += 1) {
        const prev = s.slab.getEntry(cur).prev;
        if (isExpired(s.slab.getEntry(cur))) {
            s.listRemove(cur);
            s.removeEntry(cur);
            expired_count += 1;
            _ = s.stat_expired.fetchAdd(1, .monotonic);
        }
        cur = prev;
    }
    return expired_count;
}
