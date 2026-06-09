//! Long-lived MCP process state.
//!
//! The transport and tool handlers share one scheduler and one WDBX store for
//! the lifetime of the MCP server. Keeping that ownership here prevents the
//! dispatch layer from also being the lifecycle coordinator.

const std = @import("std");
const abi = @import("abi");

var g_mcp_scheduler: ?abi.scheduler.Scheduler = null;
var g_scheduler_initialized = std.atomic.Value(bool).init(false);
var g_mcp_wdbx_store: ?abi.features.wdbx.Store = null;
var g_wdbx_initialized = std.atomic.Value(bool).init(false);
var g_wdbx_lock: abi.foundation.sync.SpinLock = .{};

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
        g_mcp_wdbx_store = abi.features.wdbx.Store.init(std.heap.page_allocator);
    }
}

pub fn getWdbxStore() *abi.features.wdbx.Store {
    ensureWdbxStore();
    return &g_mcp_wdbx_store.?;
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
    if (g_mcp_wdbx_store) |*store| {
        store.deinit();
        g_mcp_wdbx_store = null;
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
