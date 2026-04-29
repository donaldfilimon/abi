const std = @import("std");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const CacheState = state_mod.CacheState;
const NodeIndex = state_mod.NodeIndex;
const SENTINEL = state_mod.SENTINEL;

/// Evict one entry according to the configured eviction policy.
pub fn evictOne(s: *CacheState) void {
    const victim_idx: ?NodeIndex = switch (s.config.eviction_policy) {
        .lru, .fifo => s.listPopBack(),
        .lfu => findLfuVictim(s),
        .random => findRandomVictim(s),
    };

    if (victim_idx) |idx| {
        s.removeEntry(idx);
        _ = s.stat_evictions.fetchAdd(1, .monotonic);
    }
}

fn findLfuVictim(s: *CacheState) ?NodeIndex {
    var min_freq: u32 = std.math.maxInt(u32);
    var min_idx: NodeIndex = SENTINEL;
    var cur = s.list_tail;
    var checked: u32 = 0;

    while (cur != SENTINEL and checked < 64) : (checked += 1) {
        const entry = s.slab.getEntry(cur);
        if (entry.active and entry.frequency < min_freq) {
            min_freq = entry.frequency;
            min_idx = cur;
        }
        cur = entry.prev;
    }

    if (min_idx != SENTINEL) {
        s.listRemove(min_idx);
    }
    return if (min_idx != SENTINEL) min_idx else null;
}

fn findRandomVictim(s: *CacheState) ?NodeIndex {
    if (s.key_map.count() == 0) return null;

    // xorshift64 RNG
    var x = s.rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s.rng_state = x;

    const target = x % s.key_map.count();
    var cur = s.list_head;
    var i: u64 = 0;
    while (cur != SENTINEL and i < target) : (i += 1) {
        cur = s.slab.getEntry(cur).next;
    }

    if (cur != SENTINEL) {
        s.listRemove(cur);
        return cur;
    }
    return null;
}
