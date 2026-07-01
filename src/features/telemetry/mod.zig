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
        // Count the drop under the same lock every other g_dropped access uses.
        // (Previously a non-atomic load+store outside the lock — a lost-update
        // race with the full-table path and readers.)
        g_lock.lock();
        defer g_lock.unlock();
        g_dropped += 1;
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

/// Point-in-time aggregate view (`total`/`distinct`/`dropped`) of the table.
/// `dropped > 0` or `distinct == SLOT_CAPACITY` means the fixed table is
/// undersized for the workload and some events are being lost.
pub fn summary() types.Summary {
    g_lock.lock();
    defer g_lock.unlock();
    var distinct: usize = 0;
    for (g_slots) |slot| {
        if (slot.used) distinct += 1;
    }
    return .{ .total = g_total, .distinct = distinct, .dropped = g_dropped };
}

/// Owned copy of every retained counter. The lock is held only long enough to
/// copy the fixed table into stack scratch, so no allocation happens under the
/// spinlock. Caller frees each `name` and the outer slice.
pub fn snapshot(allocator: std.mem.Allocator) ![]types.CounterSnapshot {
    var scratch: [SLOT_CAPACITY]Slot = undefined;
    var used: usize = 0;
    {
        g_lock.lock();
        defer g_lock.unlock();
        for (g_slots) |slot| {
            if (slot.used) {
                scratch[used] = slot;
                used += 1;
            }
        }
    }

    var list: std.ArrayListUnmanaged(types.CounterSnapshot) = .empty;
    errdefer {
        for (list.items) |item| allocator.free(item.name);
        list.deinit(allocator);
    }
    try list.ensureTotalCapacity(allocator, used);
    for (scratch[0..used]) |slot| {
        const name = try allocator.dupe(u8, slot.name[0..slot.name_len]);
        errdefer allocator.free(name);
        list.appendAssumeCapacity(.{ .name = name, .value = slot.value });
    }
    return list.toOwnedSlice(allocator);
}

/// Render the table as Prometheus-style text exposition. Counter names are
/// sanitized to the Prometheus charset (`.` and anything outside `[a-zA-Z0-9_:]`
/// become `_`); the `abi_telemetry_*` meta lines surface table health so a full
/// or lossy table is observable. Caller owns the returned bytes.
pub fn writeText(allocator: std.mem.Allocator) ![]u8 {
    const counters = try snapshot(allocator);
    defer {
        for (counters) |c| allocator.free(c.name);
        allocator.free(counters);
    }
    const s = summary();

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "# abi telemetry (prometheus text exposition)\n");
    try appendMeta(allocator, &buf, "abi_telemetry_events_total", "counter", s.total);
    try appendMeta(allocator, &buf, "abi_telemetry_distinct_counters", "gauge", @intCast(s.distinct));
    try appendMeta(allocator, &buf, "abi_telemetry_dropped_events_total", "counter", s.dropped);

    for (counters) |c| {
        var name_buf: [NAME_CAPACITY + 1]u8 = undefined;
        const name = sanitizeName(c.name, &name_buf);
        const line = try std.fmt.allocPrint(allocator, "{s} {d}\n", .{ name, c.value });
        defer allocator.free(line);
        try buf.appendSlice(allocator, line);
    }

    return buf.toOwnedSlice(allocator);
}

fn appendMeta(
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    name: []const u8,
    kind: []const u8,
    value: u64,
) !void {
    const line = try std.fmt.allocPrint(allocator, "# TYPE {s} {s}\n{s} {d}\n", .{ name, kind, name, value });
    defer allocator.free(line);
    try buf.appendSlice(allocator, line);
}

/// Sanitize `name` to a valid Prometheus metric name. Prometheus names must not
/// only use the `[a-zA-Z0-9_:]` charset but also *start* with `[a-zA-Z_:]`, so a
/// leading digit (or an empty name) is prefixed with `_`. Worst case is a
/// full-length name with a `_` prefix, hence the `NAME_CAPACITY + 1` buffer.
fn sanitizeName(name: []const u8, out: *[NAME_CAPACITY + 1]u8) []const u8 {
    if (name.len == 0) {
        out[0] = '_';
        return out[0..1];
    }
    var w: usize = 0;
    // Only a leading digit is an illegal start: every other input char maps to
    // a letter, `_`, `:`, or `_` (via the else branch), all valid first chars.
    if (name[0] >= '0' and name[0] <= '9') {
        out[0] = '_';
        w = 1;
    }
    const n = @min(name.len, NAME_CAPACITY);
    for (name[0..n]) |ch| {
        out[w] = switch (ch) {
            'a'...'z', 'A'...'Z', '0'...'9', '_', ':' => ch,
            else => '_',
        };
        w += 1;
    }
    return out[0..w];
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

test "telemetry summary and snapshot reflect recorded counters" {
    reset();
    defer reset();

    increment("scheduler.tasks.completed", 3);
    record("plugin.loaded");

    const s = summary();
    try std.testing.expectEqual(@as(u64, 4), s.total);
    try std.testing.expectEqual(@as(usize, 2), s.distinct);
    try std.testing.expectEqual(@as(u64, 0), s.dropped);

    const snap = try snapshot(std.testing.allocator);
    defer {
        for (snap) |c| std.testing.allocator.free(c.name);
        std.testing.allocator.free(snap);
    }
    try std.testing.expectEqual(@as(usize, 2), snap.len);
}

test "telemetry writeText renders sanitized prometheus exposition" {
    reset();
    defer reset();

    increment("scheduler.tasks.completed", 2);

    const text = try writeText(std.testing.allocator);
    defer std.testing.allocator.free(text);

    // Dots are sanitized to underscores for the Prometheus charset.
    try std.testing.expect(std.mem.indexOf(u8, text, "scheduler_tasks_completed 2") != null);
    // Self-observability meta lines are always present.
    try std.testing.expect(std.mem.indexOf(u8, text, "abi_telemetry_events_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "abi_telemetry_dropped_events_total 0") != null);
}

test "telemetry writeText prefixes leading-digit names for prometheus validity" {
    reset();
    defer reset();

    increment("3d.render.count", 5);

    const text = try writeText(std.testing.allocator);
    defer std.testing.allocator.free(text);

    // A leading digit is illegal as a Prometheus name start, so it is prefixed.
    try std.testing.expect(std.mem.indexOf(u8, text, "_3d_render_count 5") != null);
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
