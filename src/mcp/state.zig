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
    if (g_scheduler_initialized.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) {
        g_mcp_scheduler = abi.scheduler.Scheduler.init(std.heap.page_allocator);
    }
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

fn ensureWdbxStore() void {
    if (g_wdbx_initialized.load(.acquire)) return;
    if (g_wdbx_initialized.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) {
        const durable = abi.features.wdbx.durable_store;
        g_mcp_session = if (g_io) |io|
            durable.Session.open(io, std.heap.page_allocator) catch |err| blk: {
                std.log.warn("durable WDBX open failed ({s}); using in-memory store", .{@errorName(err)});
                break :blk durable.Session.openInMemory(std.heap.page_allocator);
            }
        else
            durable.Session.openInMemory(std.heap.page_allocator);
    }
}

pub fn getWdbxStore() *abi.features.wdbx.Store {
    ensureWdbxStore();
    return g_mcp_session.?.storePtr();
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

test {
    std.testing.refAllDecls(@This());
}
