//! Telemetry Feature
//!
//! Lightweight event emission and structured observability hooks. Provides
//! cheap, always-available (when enabled) primitives for recording named
//! events and counters from scheduler, wdbx, ai, and connectors.
//!
//! Implementation: a process-wide, fixed-capacity, lock-guarded counter table.
//! `record`/`increment` keep their fire-and-forget (allocation-free) signatures
//! and accumulate into the table; `counterValue`/`totalEvents`/`reset` read it
//! back. Over-long names or a full table increment `droppedEvents` rather than
//! truncating into a colliding bucket, so the numbers never silently lie.
//! Complements the opt-in `metrics` feature (which is per-instance and richer).

pub const types = @import("types.zig");

const std = @import("std");
const sync = @import("../../foundation/sync.zig");

pub const TelemetryError = types.TelemetryError;
pub const Error = types.Error;

/// Maximum distinct counter names retained process-wide.
pub const SLOT_CAPACITY = 128;
/// Maximum bytes of a counter name retained verbatim.
pub const NAME_CAPACITY = 64;

const Slot = struct {
    used: bool = false,
    name: [NAME_CAPACITY]u8 = undefined,
    name_len: usize = 0,
    value: u64 = 0,
};

var g_lock: sync.SpinLock = .{};
var g_slots: [SLOT_CAPACITY]Slot = emptySlots();
var g_total: u64 = 0;
var g_dropped: u64 = 0;

fn emptySlots() [SLOT_CAPACITY]Slot {
    var slots: [SLOT_CAPACITY]Slot = undefined;
    for (&slots) |*slot| slot.* = .{};
    return slots;
}

pub fn isEnabled() bool {
    return true;
}

/// Record a named event (fire-and-forget) — equivalent to `increment(name, 1)`.
pub fn record(name: []const u8) void {
    bump(name, 1);
}

/// Increment a named counter by `delta`. A zero/empty name or zero delta is
/// ignored; an over-long name or a full table is counted in `droppedEvents`.
pub fn increment(name: []const u8, delta: u64) void {
    if (name.len == 0 or delta == 0) return;
    bump(name, delta);
}

fn bump(name: []const u8, delta: u64) void {
    if (name.len == 0 or name.len > NAME_CAPACITY) {
        @atomicStore(u64, &g_dropped, @atomicLoad(u64, &g_dropped, .monotonic) + 1, .monotonic);
        return;
    }
    g_lock.lock();
    defer g_lock.unlock();

    if (findSlot(name)) |slot| {
        slot.value += delta;
        g_total += delta;
        return;
    }
    if (claimSlot(name)) |slot| {
        slot.value = delta;
        g_total += delta;
        return;
    }
    g_dropped += 1;
}

fn findSlot(name: []const u8) ?*Slot {
    for (&g_slots) |*slot| {
        if (slot.used and std.mem.eql(u8, slot.name[0..slot.name_len], name)) return slot;
    }
    return null;
}

fn claimSlot(name: []const u8) ?*Slot {
    for (&g_slots) |*slot| {
        if (!slot.used) {
            slot.used = true;
            slot.name_len = name.len;
            @memcpy(slot.name[0..name.len], name);
            slot.value = 0;
            return slot;
        }
    }
    return null;
}

/// Current value of `name`'s counter, or 0 if never recorded.
pub fn counterValue(name: []const u8) u64 {
    g_lock.lock();
    defer g_lock.unlock();
    if (findSlot(name)) |slot| return slot.value;
    return 0;
}

/// Sum of all recorded deltas across every counter since the last reset.
pub fn totalEvents() u64 {
    g_lock.lock();
    defer g_lock.unlock();
    return g_total;
}

/// Number of distinct counter names currently retained.
pub fn distinctCounters() usize {
    g_lock.lock();
    defer g_lock.unlock();
    var count: usize = 0;
    for (g_slots) |slot| {
        if (slot.used) count += 1;
    }
    return count;
}

/// Events dropped because a name exceeded NAME_CAPACITY or the table was full.
pub fn droppedEvents() u64 {
    g_lock.lock();
    defer g_lock.unlock();
    return g_dropped;
}

/// Clear all counters. Primarily for test isolation and process re-init.
pub fn reset() void {
    g_lock.lock();
    defer g_lock.unlock();
    g_slots = emptySlots();
    g_total = 0;
    g_dropped = 0;
}

test {
    std.testing.refAllDecls(@This());
}

test "telemetry accumulates events and counters when enabled" {
    reset();
    defer reset();

    try std.testing.expect(isEnabled());

    record("scheduler.task.submitted");
    record("scheduler.task.submitted");
    increment("tasks.total", 5);
    increment("", 9); // ignored: empty name
    increment("noop", 0); // ignored: zero delta

    try std.testing.expectEqual(@as(u64, 2), counterValue("scheduler.task.submitted"));
    try std.testing.expectEqual(@as(u64, 5), counterValue("tasks.total"));
    try std.testing.expectEqual(@as(u64, 0), counterValue("never.seen"));
    try std.testing.expectEqual(@as(usize, 2), distinctCounters());
    try std.testing.expectEqual(@as(u64, 7), totalEvents());
}

test "telemetry drops over-long names instead of colliding" {
    reset();
    defer reset();

    var long_name: [NAME_CAPACITY + 1]u8 = undefined;
    @memset(&long_name, 'x');
    record(&long_name);
    try std.testing.expectEqual(@as(u64, 0), counterValue(&long_name));
    try std.testing.expectEqual(@as(u64, 1), droppedEvents());
    try std.testing.expectEqual(@as(usize, 0), distinctCounters());
}
