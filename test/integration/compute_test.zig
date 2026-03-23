//! Integration Tests: Compute (Distributed Mesh)
//!
//! Verifies the compute module's public types, context lifecycle,
//! mesh networking types, and compute node structures through
//! the `abi.compute` facade.

const std = @import("std");
const abi = @import("abi");

const compute = abi.compute;

// ── Type availability ──────────────────────────────────────────────────

test "compute: core types are accessible" {
    _ = compute.ComputeError;
    _ = compute.Error;
    _ = compute.Context;
}

test "compute: mesh sub-module is accessible" {
    _ = compute.mesh;
    _ = compute.mesh.ComputeNode;
    _ = compute.mesh.MeshOrchestrator;
}

// ── ComputeNode ────────────────────────────────────────────────────────

test "compute: ComputeNode BackendType enum values" {
    const metal = compute.mesh.ComputeNode.BackendType.metal;
    const cuda = compute.mesh.ComputeNode.BackendType.cuda;
    const rocm = compute.mesh.ComputeNode.BackendType.rocm;
    const vulkan = compute.mesh.ComputeNode.BackendType.vulkan;
    const cpu = compute.mesh.ComputeNode.BackendType.cpu;

    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(metal));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(cuda));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(rocm));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(vulkan));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(cpu));
}

test "compute: ComputeNode struct layout" {
    const node = compute.mesh.ComputeNode{
        .id = [_]u8{0} ** 16,
        .address = std.mem.zeroes(std.c.sockaddr.in),
        .is_local = true,
        .available_vram_mb = 8192,
        .backend = .metal,
        .last_seen_ms = 0,
    };

    try std.testing.expect(node.is_local);
    try std.testing.expectEqual(@as(u64, 8192), node.available_vram_mb);
    try std.testing.expectEqual(compute.mesh.ComputeNode.BackendType.metal, node.backend);
}

// ── Context lifecycle ──────────────────────────────────────────────────

test "compute: Context init and deinit" {
    var ctx = compute.Context.init(std.testing.allocator);
    try std.testing.expect(ctx.initialized);
    try std.testing.expect(compute.isInitialized());

    ctx.deinit();
    try std.testing.expect(!ctx.initialized);
    try std.testing.expect(!compute.isInitialized());
}

// ── Module lifecycle ───────────────────────────────────────────────────

test "compute: isEnabled returns true" {
    try std.testing.expect(compute.isEnabled());
}

test "compute: init-deinit cycle does not leak" {
    // First cycle
    var ctx1 = compute.Context.init(std.testing.allocator);
    try std.testing.expect(compute.isInitialized());
    ctx1.deinit();
    try std.testing.expect(!compute.isInitialized());

    // Second cycle — ensures clean state after deinit
    var ctx2 = compute.Context.init(std.testing.allocator);
    try std.testing.expect(compute.isInitialized());
    ctx2.deinit();
    try std.testing.expect(!compute.isInitialized());
}

// ── Compute error set ──────────────────────────────────────────────────

test "compute: error set members" {
    const mesh_unavailable: compute.ComputeError = error.MeshUnavailable;
    const node_unreachable: compute.ComputeError = error.NodeUnreachable;
    const task_failed: compute.ComputeError = error.TaskFailed;
    const oom: compute.ComputeError = error.OutOfMemory;

    try std.testing.expect(mesh_unavailable != node_unreachable);
    try std.testing.expect(task_failed != oom);
}

// ── Types sub-module ───────────────────────────────────────────────────

test "compute: types sub-module re-exports" {
    _ = compute.types;
    _ = compute.types.ComputeError;
    _ = compute.types.Error;
    _ = compute.types.Context;
}

test {
    std.testing.refAllDecls(@This());
}
