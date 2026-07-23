//! Long-lived MCP process state.
//!
//! The transport and tool handlers share one scheduler and one WDBX store for
//! the lifetime of the MCP server. Keeping that ownership here prevents the
//! dispatch layer from also being the lifecycle coordinator.

const std = @import("std");
const abi = @import("abi");

var g_mcp_scheduler: ?abi.scheduler.Scheduler = null;
var g_scheduler_initialized = std.atomic.Value(bool).init(false);
var g_mcp_session: ?abi.features.wdbx.durable_store.Session = null;
var g_wdbx_initialized = std.atomic.Value(bool).init(false);
var g_wdbx_lock: abi.foundation.sync.SpinLock = .{};
// Dedicated init locks: lazy construction is serialized here (double-checked,
// publishing the `initialized` flag only AFTER the optional is constructed) so a
// concurrent reader can never observe `initialized==true` while the backing
// optional is still null. Distinct from g_wdbx_lock (the per-operation store lock).
var g_scheduler_init_lock: abi.foundation.sync.SpinLock = .{};
var g_wdbx_init_lock: abi.foundation.sync.SpinLock = .{};
/// IO handle for durable-store persistence, set once by `main` before the
/// transport loops start. When unset (e.g. contract tests that invoke handlers
/// directly), the store opens in-memory so test behavior is unchanged.
var g_io: ?std.Io = null;

pub fn setIo(io: std.Io) void {
    g_io = io;
}

pub fn statDelta(after: usize, before: usize) usize {
    return if (after >= before) after - before else 0;
}

fn ensureScheduler() void {
    if (g_scheduler_initialized.load(.acquire)) return;
    g_scheduler_init_lock.lock();
    defer g_scheduler_init_lock.unlock();
    if (g_scheduler_initialized.load(.acquire)) return; // double-check under the lock
    g_mcp_scheduler = abi.scheduler.Scheduler.init(std.heap.page_allocator);
    // Publish readiness only after construction: a peer that observes the flag
    // via acquire is guaranteed a non-null g_mcp_scheduler (release/acquire pair).
    g_scheduler_initialized.store(true, .release);
}

pub fn getScheduler() *abi.scheduler.Scheduler {
    ensureScheduler();
    return &g_mcp_scheduler.?;
}

pub fn deinitScheduler() void {
    if (g_mcp_scheduler) |*s| {
        s.deinit();
        g_mcp_scheduler = null;
    }
    g_scheduler_initialized.store(false, .release);
}

fn envTruthy(key: []const u8) bool {
    const v = abi.foundation.env.get(key) orelse return false;
    return std.ascii.eqlIgnoreCase(v, "1") or std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "yes");
}

fn ensureWdbxStore() void {
    if (g_wdbx_initialized.load(.acquire)) return;
    g_wdbx_init_lock.lock();
    defer g_wdbx_init_lock.unlock();
    if (g_wdbx_initialized.load(.acquire)) return; // double-check under the lock
    const durable = abi.features.wdbx.durable_store;
    g_mcp_session = if (g_io) |io|
        durable.Session.open(io, std.heap.page_allocator) catch |err| blk: {
            if (envTruthy(abi.foundation.env.WDBX_ALLOW_MEMORY_FALLBACK_ENV)) {
                std.log.warn("durable WDBX open failed ({s}); ABI_WDBX_ALLOW_MEMORY_FALLBACK set — using empty in-memory store", .{@errorName(err)});
                break :blk durable.Session.openInMemory(std.heap.page_allocator);
            }
            // Fail closed: leave uninitialized so tools surface the open error
            // instead of silently serving an empty RAM store over a corrupt disk.
            std.log.err("durable WDBX open failed ({s}); refusing in-memory fallback (set ABI_WDBX_ALLOW_MEMORY_FALLBACK=1 to override)", .{@errorName(err)});
            return;
        }
    else
        durable.Session.openInMemory(std.heap.page_allocator);
    // Publish readiness only after the session is constructed (see ensureScheduler).
    g_wdbx_initialized.store(true, .release);
}

pub fn getWdbxStore() !*abi.features.wdbx.Store {
    ensureWdbxStore();
    if (g_mcp_session) |*session| return session.storePtr();
    return error.WdbxUnavailable;
}

pub fn lockWdbxStore() void {
    g_wdbx_lock.lock();
}

pub fn unlockWdbxStore() void {
    g_wdbx_lock.unlock();
}

pub fn deinitWdbxStore() void {
    lockWdbxStore();
    defer unlockWdbxStore();
    if (g_mcp_session) |*session| {
        session.deinit(); // checkpoints to disk for persistent sessions
        g_mcp_session = null;
    }
    g_wdbx_initialized.store(false, .release);
}

test "stat delta saturates at zero" {
    try std.testing.expectEqual(@as(usize, 3), statDelta(5, 2));
    try std.testing.expectEqual(@as(usize, 0), statDelta(2, 5));
}

test "getWdbxStore available without setIo uses in-memory" {
    deinitWdbxStore();
    const store = try getWdbxStore();
    try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
    deinitWdbxStore();
}

test {
    std.testing.refAllDecls(@This());
}
